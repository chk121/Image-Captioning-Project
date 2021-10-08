import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        # Load the pretrained ResNet-152 && replace top fc layer
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        
        modules = list(resnet.children())[:-1]  # Delete the last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()


    def init_weights(self):
        ## Initialize the weights
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        ## Extract the image feature vectors.
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)  ## Flatten multi-dimentional tensor to 1d
        cnn_features = features
        features = self.bn(self.linear(features))
        return features, cnn_features

    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        #sizes of the model's blocks
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding Layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        # LSTM Unit
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, dropout=0.4, num_layers=self.num_layers)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.attention = nn.Linear(hidden_size+embed_size, 2048)
        self.attended = nn.Linear(2048+embed_size, embed_size)
        self.softmax = nn.Softmax()
        self.init_weights()
        
    def init_weights(self):
        ## Initialize weights.
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.attention.weight.data.uniform_(-0.1, 0.1)
        self.attention.bias.data.fill_(0)
        self.attended.weight.data.uniform_(-0.1, 0.1)
        self.attended.bias.data.fill_(0)
        
    
    def forward(self, features, cnn_features, captions, lengths):
        ## Decode image feature vectors && generate captions.

        embeddings = self.embed(captions)
        # vals = torch.cat((features.unsqueeze(1), captions_embed), dim=1)
        # outputs, (self.hidden_state, self.cell_state) = self.lstm(vals, (self.hidden_state, self.cell_state))
        # pass through the linear unit
        #outputs = self.fc_out(outputs)
        
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)[0]
        batch_sizes = pack_padded_sequence(embeddings, lengths, batch_first=True)[1]
        

        hiddenStates = None
        start = 0
        
        for batch_size in batch_sizes:
            in_vector = packed[start:start+batch_size].view(batch_size, 1, -1)
            start += batch_size
            if hiddenStates is None:
                hiddenStates, (h_n, c_n) = self.lstm(in_vector)
                hiddenStates = torch.squeeze(hiddenStates)
            else:
                h_n, c_n = h_n[:,0:batch_size,:], c_n[:,0:batch_size,:]
                info_vector = torch.cat((in_vector, h_n.view(batch_size, 1, -1)), dim=2)
                attention_weights = self.attention(info_vector.view(batch_size, -1))
                attention_weights = self.softmax(attention_weights)
                attended_weights = cnn_features[:batch_size] * attention_weights
                attended_info_vec = torch.cat((in_vector.view(batch_size, -1), attended_weights), dim=1)
                attended_in_vec = self.attended(attended_info_vec)
                attended_in_vec = attended_in_vec.view(batch_size, 1, -1)
                out, (h_n, c_n) = self.lstm(attended_in_vec, (h_n, c_n))
                hiddenStates = torch.cat((hiddenStates, out.view(batch_size, -1)))  ## ??
        
        hiddenStates = self.linear(hiddenStates)
        
        return hiddenStates



    def sample(self, inputs, cnn_features, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        # Initialize the output
        output = []
        batch_size = inputs.shape[0]
        inputs = inputs.unsqueeze(1)

        # # Initialize hidden state
        # self.hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        # self.cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        
        for i in range(max_len):
            if states is None:
                hiddens, states = self.lstm(inputs, states)
            else:
                h_n, c_n = states
                info_vector = torch.cat((inputs, h_n.view(batch_size, 1, -1)), dim=2)
                attention_weights = self.attention(info_vector.view(batch_size, -1))
                attention_weights = self.softmax(attention_weights)
                attended_weights = cnn_features[:batch_size] * attention_weights
                attended_info_vec = torch.cat((inputs.view(batch_size, -1), attended_weights), dim=1)
                attended_in_vec = self.attended(attended_info_vec)
                attended_in_vec = attended_in_vec.view(batch_size, 1, -1)
                hiddens, states = self.lstm(attended_in_vec, states)

            # pass through linear unit
            outputs = self.linear(hiddens.squeeze(1))   # (batch_size, vocab_size)
            
            # predict the most likely next word
            predicted = outputs.max(1)[1]
            # outputs = outputs.squeeze(1)
            # _, max_indice = torch.max(outputs, dim=1)
            
            # store the word predicted
            # output.append(predicted.cpu().numpy()[0].item())
            output.append(predicted)
            
            # # if predicted the end token
            # if predicted == 1
            #     break
        
            ## embed the last predicted word to be the new input of the lstm
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)        # (batch_size, 1, embed_size)
        output = torch.cat(output, 0)   # (batch_size, 20)
        
        return output.squeeze()