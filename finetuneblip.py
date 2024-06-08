"""
    FineTuneBLIP
"""
from datasets import Dataset
import json
import os
from PIL import Image
import io
import base64

from transformers import AutoProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import torch
import numpy as np
import random

import wandb

"""## Load model and processor"""
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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

experiment = 14

sample_size = 10000
learning_rate = 5e-7
batch_size = 2
num_epochs = 2

# --------------------- SAVE HYPERPARAMS --------------------------

directory = f'./finetunedModel/exp{experiment}'
os.makedirs(directory, exist_ok=True)

mp = {}

mp["lr"] = learning_rate
mp["sample_size"] = sample_size
mp["epochs"] = num_epochs
mp["batch_size"] = batch_size
mp["num_training_samples"] = int(sample_size * 0.8)
mp["data_split"] = "80:20"

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

val_size = int(len(my_data["image"]) * 0.2)  # using 20% of the dataset for validation
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

train_dataset = ImageCaptioningDataset(train_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

val_dataset = ImageCaptioningDataset(val_dataset, processor)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

# ------------------------------ TRAINING --------------------------

model.train()

model_save_name = "finetuned_model.pth"
model_save_path = os.path.join(directory, model_save_name)

# Initialize wandb for hyperparameter tuning
wandb.init(project=f'finetune-blip-exp{experiment}', entity="pvashisht")

wandb.config = {
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size,
}

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for i, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)

        loss = outputs.loss
        train_loss += loss.item()

        if i % 20 == 0:
            wandb.log({"train_loss": loss.item()})

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.eval() 
    val_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 
                model_save_path)

wandb.finish()