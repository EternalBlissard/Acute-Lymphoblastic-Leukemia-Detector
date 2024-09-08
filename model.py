
import torch
import torchvision
from torch import nn
from helper import setAllSeeds

def getViT(seed,classNames,DEVICE):
  setAllSeeds(seed)
  vitWeights = torchvision.models.ViT_B_16_Weights.DEFAULT
  vitTransforms = vitWeights.transforms()
  vit = torchvision.models.vit_b_16(weights=vitWeights).to(DEVICE)
  for param in vit.parameters():
    param.requires_grad = False
  vit.heads = nn.Linear(in_features=768, out_features=len(classNames)).to(DEVICE)
  return vit,vitTransforms

def getEffNetModel(seed,numClasses):
  setAllSeeds(seed)
  effNetWeights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  effNetTransforms = effNetWeights.transforms()
  effNet = torchvision.models.efficientnet_b2(weights=effNetWeights)
  for param in effNet.parameters():
    param.requires_grad = False
  effNet.classifier = nn.Sequential(
    nn.Dropout(p=0.3,inplace=True),
    nn.Linear(1408,numClasses,bias=True)
  )
  return effNet,effNetTransforms