import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
## Function to initialize weights with xavier initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
    

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

        # output Fully Connected Layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        
        #initialize the weights
#         self = self.apply(weights_init)
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        # setup the device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        batch_size = features.size(0)
        
        # init the hidden and cell states to zero
        self.hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        self.cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        
        # embed the captions
        captions_embed = self.embed(captions)
        
        # pass through lstm unit(s)
        vals = torch.cat((features.unsqueeze(1), captions_embed), dim=1)
        outputs, (self.hidden_state, self.cell_state) = self.lstm(vals, (self.hidden_state, self.cell_state))
        
        # pass through the linear unit
        outputs = self.fc_out(outputs)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        # Initialize the output
        output = []
        batch_size = inputs.shape[0]
#         batch_size = 1
        
        # Initialize hidden state
        self.hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        self.cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        
        while True:
            # pass through lstm unit(s)
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(inputs, (self.hidden_state, self.cell_state))
            
            # pass through linear unit
            outputs = self.fc_out(lstm_out)
            
            # predict the most likely next word
            outputs = outputs.squeeze(1)
            _, max_indice = torch.max(outputs, dim=1)
            
            # store the word predicted
            output.append(max_indice.cpu().numpy()[0].item())
            
            # if reached the max len or predicted the end token
            if max_indice == 1 or len(output) >= max_len:
                break
        
            ## embed the last predicted word to be the new input of the lstm
            inputs = self.embed(max_indice)
            inputs = inputs.unsqueeze(1)
            
            
        return output