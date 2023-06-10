import pandas as pd
import numpy as np
import neattext.functions as nfx
data = pd.read_csv("./emotion-dataset.csv")

#Emotion text count
data['Emotion'].value_counts()
