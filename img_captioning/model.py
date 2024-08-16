import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderTransformer, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.feature_embed = nn.Linear(1, embed_size)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(self.transformer_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.word_embed(captions)
        features = self.feature_embed(features.unsqueeze(-1))
        hiddens = self.transformer(embeddings, features)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        # TODO: NEED CODE REVIEW
        features = self.feature_embed(features.unsqueeze(-1))
        if states is None:
            states = torch.zeros(features.size(0), 1, features.size(1)).to(features.device)
        for i in range(self.max_seg_length):
            hiddens = self.transformer(states, features)
            outputs = self.linear(hiddens)
            _, predicted = outputs.max(1)
            states = states.repeat(1, 1, 1)
            states[:, i:i + 1, :] = self.embed(predicted)
        return states

class transformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=8)
        self.transformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=8)
        self.transformerEncoder = nn.TransformerEncoder(self.transformerEncoderLayer, num_layers=num_layers)
        self.transformerDecoder = nn.TransformerDecoder(self.transformerDecoderLayer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        encoderhidden = self.transformerEncoder(features)
        hiddens = self.transformerDecoder(embeddings, encoderhidden)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None):
        features = self.transformerEncoder(features)
        if states is None:
            states = torch.zeros(features.size(0), 1, features.size(1)).to(features.device)
        for i in range(self.max_seg_length):
            hiddens = self.transformerDecoder(states, features)
            outputs = self.linear(hiddens)
            _, predicted = outputs.max(1)
            states = states.repeat(1, 1, 1)
            states[:, i:i + 1, :] = self.embed(predicted)
        return states