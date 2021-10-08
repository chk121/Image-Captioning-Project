import torch
import torch.nn as nn
import torchvision.models as models
# from torchvision import transforms
# import sys
# sys.path.append('/mnt/ssd2/kcheng/gpu205/cocoapi/PythonAPI')
# from pycocotools.coco import COCO
# from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import math


def main():
    print('1')
    resnet = models.resnet50(pretrained=False)
    modules = list(resnet.children())[:-1]
    print(modules)


if __name__ == '__main__':
    main()