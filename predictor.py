### Imports for Modules ### 
import gradio as gr
import os
import torch
from typing import Tuple, Dict
from timeit import default_timer as timer

### Functional Imports
from model import getViT, getEffNetModel

classNames = ["Benign", "Early", "Pre","Pro"]
ViTModel, VitTransforms = getViT(42,classNames,torch.device("cpu"))
ViTModel.load_state_dict(torch.load(f="ViTModel.pt",map_location=torch.device("cpu")))
EffNetModel, EffNetTransforms = getEffNetModel(42,len(classNames))
# EffNetModel.load_state_dict(torch.load(f="EffNetModel.pt",map_location=torch.device("cpu")))

def predictionMakerViT(img):
  startTime = timer()
  img = VitTransforms(img).unsqueeze(0)
  ViTModel.eval()
  with torch.inference_mode():
    predProbs = torch.softmax(ViTModel(img),dim=1)
  predDict = {classNames[i]: float(predProbs[0][i]) for i in range(len(classNames))}
  endTime = timer()
  predTime = round(endTime-startTime,4)
  return predDict,predTime

def predictionMakerEfficientNet(img):
  startTime = timer()
  img = EffNetTransforms(img).unsqueeze(0)
  EffNetModel.eval()
  with torch.inference_mode():
    predProbs = torch.softmax(EffNetModel(img),dim=1)
  predDict = {classNames[i]: float(predProbs[0][i]) for i in range(len(classNames))}
  endTime = timer()
  predTime = round(endTime-startTime,4)
  return predDict,predTime

def predictionMaker(Model_Choice="ViT",img=None):
  if(Model_Choice == "ViT"):
    predDict,predTime = predictionMakerViT(img)
    return predDict,predTime
  if(Model_Choice == "EfficientNet"):
    predDict,predTime = predictionMakerEfficientNet(img)
    return predDict,predTime
  return None,None
  predDictViT,predTimeViT = predictionMakerViT(img)
  predDictEffNet,predTimeEffNet = predictionMakerEfficientNet(img)
  return predDictViT,predDictEffNet,predTimeViT,predTimeEffNet