import string
from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import words
import numpy as np
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

classifier = open("DACT_Classfication.model", "rb")
model = pickle.load(classifier)

def bodyLength(x):
    return len(x) - x.count(" ")

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100

def make_array(question):
    array = (question)
    k = 125 - len(array)
    zero_array = np.zeros((k,))
    # final_arr = np.concatenate(array, zero_array)
    final_array = np.array((array) + (zero_array.tolist()), dtype=float)
    # print(len(final_array))
    # fin_array = np.append(final_array, question)
    return final_array

wn = nltk.WordNetLemmatizer()

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.findall("\S+", text)
    text = [wn.lemmatize(word) for word in tokens if word not in words.words()]
    return text

@app.get ("/")
def index():
    return {'API Running'}



@app.post("/predict")
async def post_question(question: str):
    body_len = float((bodyLength(question)))
    puct_perc = float(count_punct(question))
    text = clean_text(question)
    tf_idf_vect = TfidfVectorizer()
    vect_fit = tf_idf_vect.fit([question])
    vectorizer = tf_idf_vect.transform([question])
    question_vect = pd.concat([pd.DataFrame([body_len]), pd.DataFrame([puct_perc]), pd.DataFrame(vectorizer.toarray())], axis=1)
    # question_vect_arary = make_array(question_vect)
    array_question = question_vect.to_numpy()
    list_ques = array_question[0].tolist()
    fin_array = make_array(list_ques)
    prediction = model.predict([fin_array])

    return {"Prediction": prediction[0]}

if __name__ == '__main__':
    uvicorn.run(app, host= '127.0.0.1', port= 8000)