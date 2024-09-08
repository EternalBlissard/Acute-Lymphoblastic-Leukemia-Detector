### Imports for Modules ### 
import gradio as gr
import os
import torch
from typing import Tuple, Dict
from timeit import default_timer as timer

### Functional Imports
from predictor import predictionMakerViT as predictionMaker

exampleList = [["examples/" + example] for example in os.listdir("examples")]

title = "Acute-Lymphoblastic-Leukemia-Detector"
description = "Trained a Vision Transformer to classify images of Based on [Acute-Lymphoblastic-Leukemia-Dataset](https://www.kaggle.com/datasets/mehradaria/leukemia)."


# Create the Gradio demo
demo = gr.Interface(fn=predictionMaker, 
                    inputs=[gr.Image(type="pil")], 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=exampleList, 
                    title=title,
                    description=description,)

# Launch the demo!
demo.launch() 




