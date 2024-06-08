"""
    FineTuneBLIP
"""
from datasets import Dataset
import json
import os
from PIL import Image
import io
import base64

from transformers import AutoProcessor, BlipForConditionalGeneration, InstructBlipForConditionalGeneration, InstructBlipProcessor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F

import torch
import numpy as np
import random

import wandb

"""## Load model and processor"""
model_teacher = InstructBlipForConditionalGeneration.from_pretrained("./models/instructBlip")
processor_teacher = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

model_student = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# processor_teacher = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_teacher.to(device)
model_student.to(device)

grountTruths = [
    './outputs/groundTruth/surrealism/out.txt',
    './outputs/groundTruth/art_nouveau/out.txt'
]

images = [
    './data/artbench-10-imagefolder-split/train/surrealism',
    './data/artbench-10-imagefolder-split/train/art_nouveau'
]

mapped = zip(grountTruths, images)

# ------------------ EXPERIMENT HYPERPARAMETERS -------------------

experiment = 25

sample_size = 500
learning_rate = 5e-6
batch_size = 2
num_epochs = 1
kl_loss_weight = 0.3
accumulation_steps = 8
train_split_size = 0.8

# --------------------- SAVE HYPERPARAMS --------------------------

directory = f'./finetunedModel/exp{experiment}'
os.makedirs(directory, exist_ok=True)

mp = {}

mp["lr"] = learning_rate
mp["sample_size"] = sample_size
mp["epochs"] = num_epochs
mp["batch_size"] = batch_size
mp["num_training_samples"] = int(sample_size * train_split_size)
mp["data_split"] = train_split_size
mp["kl_div"] = True
mp["kl_div_weight"] = kl_loss_weight
mp["accumulation_steps"] = accumulation_steps

with open(f'{directory}/hyperparams.txt', 'a') as convert_file: 
    convert_file.write(json.dumps(mp))
    convert_file.write("\n")

# -----------------------------------------------------------------

def load_data_and_encode_images(dataset, sample_size=None):
    data = {"image": [], "text": []}

    for json_file, image_folder in dataset:
        with open(json_file, 'r') as file:
            for line in file:
                try:
                    item = json.loads(line)
                    image_path = os.path.join(image_folder, item["filename"])
                    if os.path.exists(image_path):
                        #encoded_image = encode_image_to_base64(image_path)
                        #data["image"].append(encoded_image)
                        image = Image.open(image_path)
                        data["image"].append(image)
                        data["text"].append(item["output"])
                    else:
                        print(f"Image file not found: {image_path}")
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error {e} in line: {line}")

    if sample_size:
        indices = random.sample([i for i in range(10000)], sample_size)
        data["image"] = [data["image"][i] for i in indices]
        data["text"] = [data["text"][i] for i in indices]

    return data

# Load your data
my_data = load_data_and_encode_images(mapped, sample_size)

class ImageTextDataset(Dataset):
    def __init__(self, image_list, text_list, transform=None):
        self.images = image_list
        self.texts = text_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "text": text}
    
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = ImageTextDataset(my_data['image'], my_data['text'], transform=transform)

val_size = int(len(my_data["image"]) * (1.0 - train_split_size))
train_size = len(my_data["image"]) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

"""Now that we have loaded the processor, let's load the dataset and the dataloader:"""

# train_dataset_student = ImageCaptioningDataset(train_dataset, processor_teacher)
# train_dataloader_student = DataLoader(train_dataset_student, shuffle=True, batch_size=batch_size)

# train_dataset_teacher = ImageCaptioningDataset(train_dataset, processor_teacher)
# train_dataloader_teacher = DataLoader(train_dataset_teacher, shuffle=True, batch_size=batch_size)

# val_dataset_student = ImageCaptioningDataset(val_dataset, processor_teacher)
# val_dataloader_student = DataLoader(val_dataset_student, shuffle=False, batch_size=batch_size)

# val_dataset_teacher = ImageCaptioningDataset(val_dataset, processor_teacher)
# val_dataloader_teacher = DataLoader(val_dataset_teacher, shuffle=False, batch_size=batch_size)

# print(train_dataset_teacher)


train_dataset = ImageCaptioningDataset(train_dataset, processor_teacher)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

val_dataset = ImageCaptioningDataset(val_dataset, processor_teacher)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

# ------------------------------ TRAINING --------------------------

model_save_name = "finetuned_model.pth"

# path to your model directory
model_save_dir = f'model_path'
os.makedirs(model_save_dir, exist_ok=True)

model_save_path = os.path.join(model_save_dir, model_save_name)

# you can modify the config to add additional key value pairs
config = {
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "sample_size": sample_size,
    "gradient_acc_steps": accumulation_steps,
    "train_split_size": train_split_size
}

# Initialize wandb for experiment visualisation
wandb.init(project="project name", entity="<your wandb username>", config=config)

# optimizer working on student model's parameters
optimizer = torch.optim.AdamW(model_student.parameters(), lr=learning_rate)

# put the teacher model in evaluation mode
model_teacher.eval()

for epoch in range(num_epochs):
    model_student.train()
    train_loss = 0

    optimizer.zero_grad()

    for i, batch in enumerate(train_dataloader):
        # print(batch)
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        outputs_student = model_student(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
        
        print(outputs_student.shape)

        print(outputs_student)

        with torch.no_grad():
            outputs_teacher = model_teacher(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)

        kl_loss = F.kl_div(F.log_softmax(outputs_student[2], dim=1), F.softmax(outputs_teacher[2], dim=1), reduction='batchmean')

        loss = (1 - kl_loss_weight) * outputs_student.loss + kl_loss_weight * kl_loss
        train_loss = loss.item()

        loss = loss / accumulation_steps

        loss.backward()

        if i % 20 == 0:
            wandb.log({"train_loss": train_loss})
            wandb.log({"diversion_loss": kl_loss.item()})
            wandb.log({"model_alignment_loss": outputs_student.loss.item()})

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    if len(train_dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    model_student.eval() 
    val_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model_student(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)

            loss = outputs.loss
            val_loss += loss.item()

            if i % 20 == 0:
                wandb.log({"val_loss": loss.item()})

    avg_train_loss = train_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)

    wandb.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss})

    torch.save({'epoch': epoch,
                'model_state_dict': model_student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 
                model_save_path)

wandb.finish()