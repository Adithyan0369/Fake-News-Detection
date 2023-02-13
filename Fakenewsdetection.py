import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
import seaborn as sns

st.markdown("<h1 style='text-align:center; color:white;'>Fake News Prediction</h1>", unsafe_allow_html=True)
data=pd.read_csv("E:/Projects/FakeNewsDetectionML/news.csv")

categories=['REAL','FAKE']

x=data[['label','text']]

from sklearn.model_selection import train_test_split

train, test = train_test_split(x, test_size = 0.20, random_state = 0)


val = st.text_input("Enter text to check")

# g = sns.countplot(x='label', data=data)
# g.set_ylabel('Count', fontsize=14)
# st.pyplot(g)
# fig = plt.figure(figsize=(10,4))
# sns.countplot(x='label', data=data)
# st.pyplot(fig)

st.markdown("<h1 style='color:white;'>Multinomial Model</h1>", unsafe_allow_html=True)

#MultinomialNB

model1=make_pipeline(TfidfVectorizer(),MultinomialNB())
model1.fit(train.text,train.label)
pred=model1.predict(test.text)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
c1=confusion_matrix(test.label,pred)
st.write("Confusion matrix fot test sample1")
st.write(c1)
st.write("Accuracy fot test sample1")
ac1 = accuracy_score(test['label'].to_numpy(),pred)
st.write(ac1)
cr1 = classification_report(test['label'].to_numpy(),pred)
print(cr1)

model2=make_pipeline(TfidfVectorizer(),BernoulliNB())
model2.fit(train.text,train.label)
pred2=model2.predict(test.text)

# train.label

st.markdown("<h1 style='color:white;'>Bernoulli Model</h1>", unsafe_allow_html=True)

c2=confusion_matrix(test.label,pred2)
st.write("Confusion matrix fot test sample2")
st.write(c2)

ac2 = accuracy_score(test['label'].to_numpy(),pred2)
st.write("Accuracy fot test sample2")
st.write(ac2)
from sklearn.metrics import classification_report
rep = classification_report(test['label'].to_numpy(), pred2)

print(rep)


def predict(val): 
    result = model2.predict([val])
    st.write(result)


def predict2(val):
    result = model1.predict([val])
    st.write(result)
st.write("MultinomialNB Prediction")
predict(val)
st.write("BernoulliNB Prediction")
predict2(val)