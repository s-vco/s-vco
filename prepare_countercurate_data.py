import json
from PIL import Image
import os
from tqdm import tqdm
import shutil

flickr30k_path = "./flickr30k/flickr30k-images"
train_data_path = "./Flickr30k-Counterfactuals/train_data"

with open(os.path.join(train_data_path, "up_down.json"), 'r') as f:
    ud = json.load(f)
with open(os.path.join(train_data_path, "counting.json"), 'r') as f:
    ct = json.load(f)
with open(os.path.join(train_data_path, "left_right.json"), 'r') as f:
    lr = json.load(f)

for index in tqdm(ct.keys()):
    shutil.copy(os.path.join(flickr30k_path, f"{index}.jpg"), os.path.join(train_data_path, "counting", "pos", f"{index}.jpg"))

for index in tqdm(ud.keys()):
    shutil.copy(os.path.join(flickr30k_path, f"{index}.jpg"), os.path.join(train_data_path, "up_down", "pos", f"{index}.jpg"))

for index in tqdm(lr.keys()):
    shutil.copy(os.path.join(flickr30k_path, f"{index}.jpg"), os.path.join(train_data_path, "left_right", "pos", f"{index}.jpg"))
    with Image.open(os.path.join(flickr30k_path, f"{index}.jpg")) as img:
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    with open(os.path.join(train_data_path, "left_right", "neg", f"{index}.jpg"), 'w') as f:
        flipped.save(f)
