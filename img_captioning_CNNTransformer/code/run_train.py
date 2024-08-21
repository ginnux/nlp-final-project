from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision.transforms import Normalize, Compose
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.cnn_encoder import ImageEncoder
from models.IC_encoder_decoder.transformer import Transformer

from dataset.dataloader import HDF5Dataset, collate_padd
from torchtext.vocab import Vocab

from trainer import Trainer
from utils.train_utils import parse_arguments, seed_everything, load_json
from utils.gpu_cuda_helper import select_device

import pandas as pd
import nltk
from PIL import Image
import os
from torchvision import transforms
import pickle
from build_vocab import Vocabulary
from torch.nn.utils.rnn import pad_sequence

def get_datasets(dataset_dir: str, pid_pad: float):
    # Setting some pathes
    dataset_dir = Path(dataset_dir)
    images_train_path = dataset_dir / "train_images.hdf5"
    images_val_path = dataset_dir / "val_images.hdf5"
    captions_train_path = dataset_dir / "train_captions.json"
    captions_val_path = dataset_dir / "val_captions.json"
    lengthes_train_path = dataset_dir / "train_lengthes.json"
    lengthes_val_path = dataset_dir / "val_lengthes.json"

    # images transfrom
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])

    train_dataset = HDF5Dataset(hdf5_path=images_train_path,
                                captions_path=captions_train_path,
                                lengthes_path=lengthes_train_path,
                                pad_id=pid_pad,
                                transform=transform)

    val_dataset = HDF5Dataset(hdf5_path=images_val_path,
                              captions_path=captions_val_path,
                              lengthes_path=lengthes_val_path,
                              pad_id=pid_pad,
                              transform=transform)

    return train_dataset, val_dataset


# ------------csv dataset------------#
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

        self.max_len = 52
        self.pad_id = 0

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
        #if len(target) < self.max_len:
        #    pad_size = self.max_len - len(target)
        #    target = torch.cat((target, torch.full((pad_size,), self.pad_id, dtype=torch.long)))

        # If target is longer than max_len, truncate it
        #target = target[:self.max_len]
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
    #targets = torch.zeros(len(captions), max(lengths)).long()
    # 获取原始长度再补齐
    targets = torch.zeros(len(captions), 52).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]


    return images, targets, lengths


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, type='CSV'):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    #if type == 'COCO':
        # COCO caption dataset
        #dataset = CocoDataset(root=root,
        #                      json=json,
        #                      vocab=vocab,
        #                      transform=transform)

        # Data loader for COCO dataset
        # This will return (images, captions, lengths) for each iteration.
        # images: a tensor of shape (batch_size, 3, 224, 224).
        # captions: a tensor of shape (batch_size, padded_length).
        # lengths: a list indicating valid length for each caption. length is (batch_size).
    if type == 'CSV':
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
                                             collate_fn=collate_fn) #collate_padd doesn't work
    return dataloader


if __name__ == "__main__":

    # parse command arguments
    args = parse_arguments()
    dataset_dir = args.dataset_dir  # mscoco hdf5 and json files
    resume = args.resume
    if resume == "":
        resume = None

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")

    # load confuguration file
    config = load_json(args.config_path)

    # load vocab
    min_freq = config["min_freq"]
    #vocab: Vocab = torch.load('../vocab.pkl')


    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    # --------------- dataloader --------------- #
    print("loading dataset...")
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_params = config["dataloader_parms"]
    max_len = config["max_len"]

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open('../vocab.pkl', 'rb') as f:
        vocab: Vocab = pickle.load(f)

    pad_id = 0#vocab.stoi["<pad>"]
    vocab_size = len(vocab)

    train_iter = get_loader(root = '/root/autodl-tmp',
                             json ='../captionsblip2.csv',
                             vocab = vocab,
                             transform = transform,
                             batch_size = 128,
                             shuffle=True, num_workers=2)
    val_iter = get_loader(root = '/root/autodl-tmp',
                             json ='../captionsvalblip2.csv',
                             vocab = vocab,
                             transform = transform,
                             batch_size = 128,
                             shuffle=True, num_workers=2)
    """train_ds, val_ds = get_datasets(dataset_dir, pad_id)
    train_iter = DataLoader(train_ds,
                            collate_fn=collate_padd(max_len, pad_id),
                            pin_memory=True,
                            **loader_params)
    val_iter = DataLoader(val_ds,
                          collate_fn=collate_padd(max_len, pad_id),
                          batch_size=1,
                          pin_memory=True,
                          num_workers=1,
                          shuffle=True)#
    print("loading dataset finished.")"""
    print(f"number of vocabualry is {len(vocab)}\n")

    # --------------- Construct models, optimizers --------------- #
    print("constructing models")
    # prepare some hyperparameters
    image_enc_hyperparms = config["hyperparams"]["image_encoder"]
    image_seq_len = int(image_enc_hyperparms["encode_size"]**2)

    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = pad_id
    transformer_hyperparms["img_encode_size"] = image_seq_len
    transformer_hyperparms["max_len"] = max_len - 1

    # construct models
    image_enc = ImageEncoder(**image_enc_hyperparms)
    image_enc.fine_tune(True)
    transformer = Transformer(**transformer_hyperparms)

    # load pretrained embeddings
    print("loading pretrained glove embeddings...")
    #weights = 0#vocab.vectors
    #transformer.decoder.cptn_emb.from_pretrained(weights,
    #                                             freeze=True,
    #                                             padding_idx=pad_id)
    #list(transformer.decoder.cptn_emb.parameters())[0].requires_grad = False

    # Optimizers and schedulers
    image_enc_lr = config["optim_params"]["encoder_lr"]
    parms2update = filter(lambda p: p.requires_grad, image_enc.parameters())
    image_encoder_optim = Adam(params=parms2update, lr=image_enc_lr)
    gamma = config["optim_params"]["lr_factors"][0]
    image_scheduler = StepLR(image_encoder_optim, step_size=1, gamma=gamma)

    transformer_lr = config["optim_params"]["transformer_lr"]
    parms2update = filter(lambda p: p.requires_grad, transformer.parameters())
    transformer_optim = Adam(params=parms2update, lr=transformer_lr)
    gamma = config["optim_params"]["lr_factors"][1]
    transformer_scheduler = StepLR(transformer_optim, step_size=1, gamma=gamma)

    # --------------- Training --------------- #
    print("start training...\n")
    train = Trainer(optims=[image_encoder_optim, transformer_optim],
                    schedulers=[image_scheduler, transformer_scheduler],
                    device=device,
                    pad_id=pad_id,
                    resume=resume,
                    checkpoints_path=config["pathes"]["checkpoint"],
                    **config["train_parms"])
    train.run(image_enc, transformer, [train_iter, val_iter], SEED)

    print("done")
