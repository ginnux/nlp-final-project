import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

# Encoder using a pre-trained ResNet-152
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=False)
        model_weights_path = '../resnet152-394f9c45.pth'
        resnet.load_state_dict(torch.load(model_weights_path))
        for param in resnet.parameters():
            param.requires_grad_(False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove last two layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.attention = nn.Linear(feature_dim + hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden):
        # Reshape hidden to match the features size for concatenation
        hidden = hidden.unsqueeze(1).repeat(1, features.size(1), 1)
        combined = torch.cat((features, hidden), dim=2)
        energy = torch.tanh(self.attention(combined))
        attention_weights = torch.softmax(self.v(energy), dim=1)
        attention_applied = torch.sum(attention_weights * features, dim=1)
        return attention_applied, attention_weights

# Decoder with Attention
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(embed_size, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.max_seg_length = 20

    def forward(self, features, captions, lengths):
        embeddings = self.embedding(captions)
        attention_features = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        hidden = (torch.zeros(1, features.size(0), self.hidden_size).to(features.device),
                  torch.zeros(1, features.size(0), self.hidden_size).to(features.device))

        packed_embeddings = []
        for t in range(embeddings.size(1)):
            attention_output, _ = self.attention(attention_features, hidden[0][-1])
            lstm_input = torch.cat((embeddings[:, t, :], attention_output), dim=1)
            lstm_input = lstm_input.unsqueeze(1)
            packed_embeddings.append(lstm_input)

        packed_embeddings = torch.cat(packed_embeddings, dim=1)
        packed = pack_padded_sequence(packed_embeddings, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.lstm(packed, hidden)
        outputs = self.fc(lstm_out[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search with attention."""
        # embeddings = self.embedding(features)
        attention_features = features.unsqueeze(1).repeat(1, 512, 1)
        sampled_ids = []
        # inputs = features.unsqueeze(1)  # Start with the image features
        hidden = (torch.zeros(1, features.size(0), self.hidden_size).to(features.device),
              torch.zeros(1, features.size(0), self.hidden_size).to(features.device))
    
        for i in range(self.max_seg_length):
            # Compute the attention output
            attention_output, _ = self.attention(attention_features, hidden[0][-1])
        
            # Combine attention output with the current input (embedding)
            lstm_input = torch.cat((attention_features.squeeze(1), attention_output), dim=1)
            lstm_input = lstm_input.unsqueeze(1)
        
            # LSTM forward step
            hiddens, hidden = self.lstm(lstm_input, hidden)
        
            # Predict the next word
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
        
            sampled_ids.append(predicted)
        
            # Prepare the next input (embedding of the predicted word)
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
        
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    
# Combine Encoder and Decoder into a Captioning model
class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, num_layers=1):
        super(CaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, attention_dim, num_layers)

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs



if __name__ == '__main__':
    # Example of how to initialize and use the model
    embed_size = 256
    hidden_size = 512
    vocab_size = 5000  # Example vocabulary size
    attention_dim = 256

    model = CaptioningModel(embed_size, hidden_size, vocab_size, attention_dim)

    # Example input
    images = torch.randn(10, 3, 224, 224)  # Example batch of images
    captions = torch.randint(0, vocab_size, (10, 15))  # Example batch of captions
    lengths = torch.tensor([15, 14, 13, 12, 11, 10, 9, 8, 7, 6])  # Example caption lengths

    outputs = model(images, captions, lengths)
    print(outputs.shape)  # Should output (batch_size*sum(lengths), vocab_size)
