import random
import json
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                      for word in sentence_words]
    return sentence_words


def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    # print(res)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res)
               if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
    print(return_list)
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# clear screen
def clear():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
clear()
print("Chatbot is up!")

while True:
    message = input("You>> ")
    ints = predict_class(message)
    if(len(ints) > 0):
        res = get_response(ints, intents)
        print("ChatBot>> " + res)
    else:
        print("ChatBot>> I don't understand your question")
