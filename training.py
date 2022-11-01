import random
import json
import pickle
from statistics import mode
import numpy as np
import nltk
import tensorflow as tf

from tensorflow.keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
# print(intents)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    print("Intent: {}".format(intent))
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        print("Word List: {}".format(word_list))
        words.extend(word_list)

        documents.append(((word_list), intent["tag"]))
        # print(documents)
        if(intent["tag"] not in classes):
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in ignore_letters]
# print(words)
words = sorted(list(set(words)))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

training_x = list(training[:, 0])
training_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(training_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(training_y[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd, metrics=["accuracy"])
history = model.fit(np.array(training_x), np.array(
    training_y), epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.h5", history)

print("Model Trained")
