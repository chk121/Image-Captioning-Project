import os
import os.path as osp
import cv2
import sys
sys.path.append('/mnt/ssd2/kcheng/gpu205/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import json
import pickle
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from data_loader import get_loader
from torchvision import transforms
from model import EncoderCNN, DecoderRNN

def load_image(im_path, transform=None):
    image = Image.open(im_path).convert("RGB")
    image = image.resize([256, 256], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)  # convert [3, 224, 224] -> [1, 3, 224, 224], where 1 is batch_size
        
    return image


def clean_sentence(output, vocab):
    sentence = ''
    for x in output:
        word = vocab.idx2word[x]
        if word == '<end>':
            break
        sentence += ' ' + word
    
    sentence = sentence.strip()[len('<start>')+1:]
    if sentence.endswith('.'):
        sentence = sentence[:-2]
    
    return sentence


def caption_generation(img):
    # Define a transform to pre-process the testing images.
    transform_test = transforms.Compose([ 
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))])


    ## Load vocabulary wrapper
    with open('./vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    print('Vocabulary successfully loaded from vocab.pkl file!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify the saved models to load.
    encoder_file = 'encoder-7-12941.pkl'
    decoder_file = 'decoder-7-12941.pkl'

    # Select appropriate values for the Python variables below.
    embed_size = 256
    hidden_size = 512

    # The size of the vocabulary.
    vocab_size = len(vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    encoder.eval()    # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    # decoder.eval()

    # Load the trained weights.
    encoder.load_state_dict(torch.load(os.path.join('./models-attention', encoder_file)))
    decoder.load_state_dict(torch.load(os.path.join('./models-attention', decoder_file)))

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)

    
    im = load_image(img, transform_test)
    # Move image Pytorch Tensor to GPU if CUDA is available.
    image_tensor = im.to(device)

    # Generate caption from image.
    try:
        feature, cnn_features = encoder(image_tensor)

        # Pass the embedded image features through the model to get a predicted caption.
        output = decoder.sample(feature, cnn_features)
        output = output.cpu().data.numpy()
    except:
        print('Error.')
    
    # Decode word_ids to words
    sentence = clean_sentence(output, vocab)

    cap_dict = {}
    cap_dict[img.name] = sentence

    
    return cap_dict