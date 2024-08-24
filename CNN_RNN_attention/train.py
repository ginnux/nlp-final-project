import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model_att import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # For logging
from torch.optim.lr_scheduler import StepLR  # Import the scheduler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(encoder, decoder, criterion, data_loader, epoch, writer):
    decoder.eval()
    encoder.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    avg_perplexity = np.exp(avg_loss)
    writer.add_scalar('Validation Loss', avg_loss, epoch)
    writer.add_scalar('Validation Perplexity', avg_perplexity, epoch)
    print(f'Validation Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}')
    decoder.train()
    encoder.train()
    return avg_loss, avg_perplexity

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Set up TensorBoard logging
    writer = SummaryWriter(log_dir=args.log_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loaders
    train_loader = get_loader(args.train_image_dir, args.train_caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(args.valid_image_dir, args.valid_caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    if args.decoder_type == 'RNN':
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.hidden_size).to(device)
    else:
        raise ValueError('Decoder type not supported')
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define separate parameters for encoder and decoder
    encoder_params = list(encoder.embed.parameters()) + list(encoder.bn.parameters())
    decoder_params = list(decoder.parameters())
    
    # Create separate optimizers for encoder and decoder
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=args.learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder_params, lr=args.learning_rate)
    
    # Initialize the learning rate schedulers for both optimizers
    encoder_scheduler = StepLR(encoder_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    decoder_scheduler = StepLR(decoder_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Train the models
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(train_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_value = 5.0  # You can adjust this value
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_value)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_value)

            decoder_optimizer.step()
            encoder_optimizer.step()

            # Print log info and write to TensorBoard
            if i % args.log_step == 0:
                perplexity = np.exp(loss.item())
                print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}')
                writer.add_scalar('Train Loss', loss.item(), epoch * total_step + i)
                writer.add_scalar('Train Perplexity', perplexity, epoch * total_step + i)
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, f'decoder-{epoch+1}-{i+1}.ckpt'))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, f'encoder-{epoch+1}-{i+1}.ckpt'))
        
        # Update learning rate scheduler
        encoder_scheduler.step()
        decoder_scheduler.step()

        # Validate the model after each epoch
        val_loss, val_perplexity = validate(encoder, decoder, criterion, val_loader, epoch, writer)
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_att_all/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--train_image_dir', type=str, default='..', help='directory for train resized images')
    parser.add_argument('--train_caption_path', type=str, default='captionsblip2.csv' , help='path for train annotation json file')
    parser.add_argument('--valid_image_dir', type=str, default='..' ,help='directory for validation resized images')
    parser.add_argument('--valid_caption_path', type=str, default='captionsvalblip2.csv', help='path for validation annotation json file')
    parser.add_argument('--log_path', type=str, default='logs/crnn_attn_all/', help='directory for saving logs')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')
    parser.add_argument('--decoder_type', type=str, default='RNN', help='RNN or Transformer')
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_step_size', type=int, default=20, help='number of epochs before learning rate is updated')
    parser.add_argument('--lr_gamma', type=float, default=0.7, help='factor by which the learning rate is reduced')
    args = parser.parse_args()
    print(args)
    main(args)
