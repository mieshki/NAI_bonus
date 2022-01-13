"""
ChatBot
Inspired on: https://www.youtube.com/watch?v=wypVcNIH6D4&t=1s
Authors:
Reiter, Aleksander <https://github.com/block439>
Dziadowiec, Mieszko <https://github.com/mieshki>
How to run:
Required python version 3.6
(optional): `pip install -r requirements.txt`
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
from translate import intent_translator
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from json import dump

stemmer = LancasterStemmer()


with open("intents-adv.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

"""
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
"""
model.fit(training, output, n_epoch=1, batch_size=8, show_metric=True)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    """

    :return:
    """
    print("Start talking with the bot (type quit to stop)!")
    output = {}
    output["input"] = []
    output["bot_answer"] = []
    while True:
        inp = input("You: ")

        if inp.lower() == "quit":
            break

        print(f'[DEBUG] Input before: {inp}')
        inp = inp.lower().replace('ą', 'a').replace('ć', 'c').replace('ę', 'e').replace('ł', 'l').replace('ń', 'n').replace('ó', 'o').replace('ś', 's').replace('ź', 'z').replace('ż', 'z')
        print(f'[DEBUG] Input after: {inp}')

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        output["input"].append(inp)
        if results[results_index] > 0.85:
            for tg in data["intents"]:
                # pl-greeting
                # en-greeting
                # if tg['tag'] == f'{language}-tag':
                if tg['tag'] == tag:
                    responses = tg['responses']
            answer = random.choice(responses)
            print(answer)
            output["bot_answer"].append(answer)
        else:
            print("I don't know :C")
            output["bot_answer"].append("Don't know")

        with open('output_data.txt', 'a') as f:
            dump(output, f)
            f.write(',')

#intent_translator()
chat()