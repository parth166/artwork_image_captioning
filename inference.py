from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from peft import PeftModel, PeftConfig
import json
import os
from glob import glob
import re
import numpy as np
import torch
import time

def remove_trailing_content_after_last_period(text):
    # This regular expression matches any text up to and including the last period.
    match = re.search(r'^(.*\.)', text)
    return match.group(0) if match else text

experiment = 5
experiment_name = "peft"
classname = "art_nouveau"

model_directory = f'./finetunedModel/{experiment_name}/exp{experiment}'
peft_config = PeftConfig.from_pretrained(model_directory)

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path
)

model = PeftModel.from_pretrained(model, model_directory)
model.print_trainable_parameters()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.eval()

folderPath = f'./data/artbench-10-imagefolder-split/test/{classname}/*'
files = glob(folderPath, recursive = False)

for file_path in files:
    filename = file_path.split("/")[-1]
    image = Image.open(file_path).convert('RGB')
    text = "The picture describes"

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, 
        do_sample=False,
        num_beams=10,
        max_length=100,
        min_length=40,
        top_p=0.9,
        repetition_penalty=1.5,
        temperature=1
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    output = remove_trailing_content_after_last_period(generated_text)
    mp = {}

    mp["output"] = output
    mp["filename"] = filename


    with open(f'{model_directory}/out_{classname}_{experiment}.txt', 'a') as convert_file: 
        convert_file.write(json.dumps(mp))
        convert_file.write("\n")
