import pandas as pd
from ali import *
from glob import glob
from tqdm import tqdm

images = sorted(glob('../img_captioning_CNNRNN/images/train/*.jpg'))
captions = []
i=0

bar = tqdm(images)
for image in bar:
    i+=1
    if i % 100 == 0:
        df = pd.DataFrame({'image': images[:i], 'caption': captions})
        df.to_csv(f'../captions_{i}.csv', index=False)
    caption = ali_caption(image)
    captions.append(caption)
    bar.set_description(f'Captioning {image}')

df = pd.DataFrame({'image': images, 'caption': captions})
df.to_csv('../captions.csv', index=False)
