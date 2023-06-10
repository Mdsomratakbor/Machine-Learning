

# Load EDA Pkgs
import pandas as pd
import numpy as np
import neattext.functions as nfx
# Load Dataset
df = pd.read_csv("./emotion-dataset.csv")

df.head()

# Value Counts
df['Emotion'].value_counts()



# Load Text Cleaning Pkgs


# Data Cleaning
dir(nfx)

# User handles
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

# Stopwords
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

df

# Load ML Pkgs
# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Features & Labels
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

#  Split Data
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)

# Build Pipeline
from sklearn.pipeline import Pipeline

# LogisticRegression Pipeline
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])

# Train and Fit Data
pipe_lr.fit(x_train,y_train)

pipe_lr

# Check Accuracy
pipe_lr.score(x_test,y_test)

# Make A Prediction
sample1 = "This chocholate was very sweet it made me happy"

pipe_lr.predict([sample1])