import torch
import pandas as pd
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, DecoderTransformer2
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.gleu_score import sentence_gleu

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

def remove_tokens(caption, tokens_to_remove):
    words = caption.split()
    cleaned_words = [word for word in words if word not in tokens_to_remove]
    cleaned_caption = ' '.join(cleaned_words)
    
    return cleaned_caption

def generate_caption(encoder, decoder, transform, vocab, image_path):
    image = load_image(image_path, transform)
    image_tensor = image.to(device)
    
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
    
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    return sentence

def compute_metrics(reference, hypothesis):
    """
    Computes BLEU, METEOR, and GLEU metrics.
    
    Args:
        reference (list of str): List of reference sentences (tokenized).
        hypothesis (list of str): Hypothesis sentence (tokenized).

    Returns:
        dict: A dictionary containing BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, and GLEU scores.
    """
    smoothing = SmoothingFunction().method4
    
    
    # Tokenize reference and hypothesis
    reference = [reference]  # NLTK expects reference to be a list of tokenized sentences
    hypothesis = hypothesis  # Hypothesis should be a list of tokens
    
    
    scores = {
        'BLEU-1': sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        'BLEU-2': sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        'BLEU-3': sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing),
        'BLEU-4': sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing),
        'METEOR': meteor_score(reference, hypothesis),
        'GLEU': sentence_gleu(reference, hypothesis)
    }
    return scores


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    reference_df = pd.read_csv(args.caption_csv_path)
    reference_captions = reference_df['caption'].tolist()

    encoder = EncoderCNN(args.embed_size).eval()
    if args.decoder_type == 'RNN':
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    elif args.decoder_type == 'Transformer':
        decoder = DecoderTransformer2(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    else:
        raise ValueError('Decoder type not supported')
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    total_bleu1, total_bleu2, total_bleu3, total_bleu4, total_meteor, total_gleu = 0, 0, 0, 0, 0, 0
    num_samples = len(reference_captions)
    results = []
    
    for idx, image_name in enumerate(sorted(os.listdir(args.image_dir))):
        image_path = os.path.join(args.image_dir, image_name)
        if os.path.isfile(image_path):
            caption = generate_caption(encoder, decoder, transform, vocab, image_path)
            tokens_to_remove = ['<start>', '<end>', '<pad>', '<unk>']
            cleaned_caption = remove_tokens(caption, tokens_to_remove)
            results.append({'image_name': image_name, 'caption': cleaned_caption})
            
            reference = reference_captions[idx].split()
            hypothesis = cleaned_caption.split()
            metrics = compute_metrics(reference, hypothesis)

            total_bleu1 += metrics['BLEU-1']
            total_bleu2 += metrics['BLEU-2']
            total_bleu3 += metrics['BLEU-3']
            total_bleu4 += metrics['BLEU-4']
            total_meteor += metrics['METEOR']
            total_gleu += metrics['GLEU']

    avg_bleu1 = total_bleu1 / num_samples
    avg_bleu2 = total_bleu2 / num_samples
    avg_bleu3 = total_bleu3 / num_samples
    avg_bleu4 = total_bleu4 / num_samples
    avg_meteor = total_meteor / num_samples
    avg_gleu = total_gleu / num_samples

    print(f'Average BLEU-1: {avg_bleu1:.4f}')
    print(f'Average BLEU-2: {avg_bleu2:.4f}')
    print(f'Average BLEU-3: {avg_bleu3:.4f}')
    print(f'Average BLEU-4: {avg_bleu4:.4f}')
    print(f'Average METEOR: {avg_meteor:.4f}')
    print(f'Average GLEU: {avg_gleu:.4f}')

    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f'Results saved to {args.output_csv}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="../images/val", help='directory containing images for generating captions')
    parser.add_argument('--output_csv', type=str, default="output.csv", help='path for saving results to CSV')
    parser.add_argument('--encoder_path', type=str, default='models_orignal/encoder-22-10.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models_orignal/decoder-22-10.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--decoder_type', type=str, default='RNN', help='RNN or Transformer')
    parser.add_argument('--caption_csv_path', type=str, default="captionsvalblip2.csv", help='path to CSV file containing reference captions')
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()

    main(args)
