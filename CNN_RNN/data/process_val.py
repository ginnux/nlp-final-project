import json
import os
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import sent_tokenize

data = dict()

annotations = []
images = []

caption_id = 0
Train_data = pd.read_csv(r'Val.csv',sep=',')
Train_data = Train_data.values
img_name = r'./images/val/(.*)'
img_id = r'./images/val/(.*)\.'

for i in range(0,len(Train_data)):
    image_name = re.findall(img_name,Train_data[i][0])[0]
    image_id = re.findall(img_id,Train_data[i][0])[0]
    caption = Train_data[i][1]
    images.append({
        "id": image_id,
        "filename": image_name,
    })

    segment_caption = sent_tokenize(caption)[0]
    annotations.append({
    "id": caption_id,
    "image_id": image_id,
    "caption": caption,
    "segment_caption": segment_caption
    })
    caption_id += 1



data['images'] = images
data['annotations'] = annotations

with open("val_data.json", "w+", encoding="utf-8") as file:
    json.dump(data, fp = file, ensure_ascii=False, indent=1)

print("Done")
