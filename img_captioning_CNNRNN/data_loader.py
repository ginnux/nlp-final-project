import torch
import torch.utils.data as data
import os
import pandas as pd
import nltk
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)

class CSVDataset(data.Dataset):
    """CSV Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, csv, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            csv: csv file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.csv = csv
        self.vocab = vocab
        self.transform = transform
        self.data = pd.read_csv(csv)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        row = self.data.iloc[index]
        caption = row['caption']
        path = row['image']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, type='CSV'):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    if type == 'COCO':
        # COCO caption dataset
        dataset = CocoDataset(root=root,
                           json=json,
                           vocab=vocab,
                           transform=transform)

        # Data loader for COCO dataset
        # This will return (images, captions, lengths) for each iteration.
        # images: a tensor of shape (batch_size, 3, 224, 224).
        # captions: a tensor of shape (batch_size, padded_length).
        # lengths: a list indicating valid length for each caption. length is (batch_size).
    elif type == 'CSV':
        # CSV caption dataset
        dataset = CSVDataset(root=root,
                            csv=json,
                            vocab=vocab,
                            transform=transform)

    else:
        raise Exception('Data loader type not supported: {}'.format(type))
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             collate_fn=collate_fn)
    return dataloader