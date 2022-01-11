
"""
import numpy as np
import json
import re
import tensorflow as tf
import random
import spacy

# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')




with open('data/intent.json') as f:
    intents = json.load(f)



def preprocessing(line):
    line = re.sub(r'[^a-zA-z.?!\']', ' ', line)
    line = re.sub(r'[ ]+', ' ', line)
    return line


# get text and intent title from json data
inputs, targets = [], []
classes = []
intent_doc = {}

for intent in intents['intents']:
    if intent['intent'] not in classes:
        classes.append(intent['intent'])
    if intent['intent'] not in intent_doc:
        intent_doc[intent['intent']] = []

    for text in intent['text']:
        inputs.append(preprocessing(text))
        targets.append(intent['intent'])

    for response in intent['responses']:
        intent_doc[intent['intent']].append(response)


def tokenize_data(input_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')

    tokenizer.fit_on_texts(input_list)

    input_seq = tokenizer.texts_to_sequences(input_list)

    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='pre')

    return tokenizer, input_seq


# preprocess input data
tokenizer, input_tensor = tokenize_data(inputs)

def create_categorical_target(targets):
    word = {}
    categorical_target = []
    counter = 0
    for trg in targets:
        if trg not in word:
            word[trg] = counter
            counter += 1
        categorical_target.append(word[trg])

    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    return categorical_tensor, dict((v, k) for k, v in word.items())


# preprocess output data
target_tensor, trg_index_word = create_categorical_target(targets)

print('input shape: {} and output shape: {}'.format(input_tensor.shape, target_tensor.shape))







# hyperparameters
epochs=50
vocab_size=len(tokenizer.word_index) + 1
embed_dim=512
units=128
target_length=target_tensor.shape[1]


# build RNN Model with tensorflow
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, dropout=0.2)),
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(target_length, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(lr=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()





early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# train the model
model.fit(input_tensor, target_tensor, epochs=epochs, callbacks=[early_stop])


def response(sentence):
    sent_seq = []
    doc = nlp(repr(sentence))

    # split the input sentences into words
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])

        # handle the unknown words error
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])

    sent_seq = tf.expand_dims(sent_seq, 0)
    # predict the category of input sentences
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)

    # choice a random response for predicted sentence
    return random.choice(intent_doc[trg_index_word[pred_class[0]]]), trg_index_word[pred_class[0]]


# chat with bot
print("Note: Enter 'quit' to break the loop.")
while True:
    input_ = input('You: ')
    if input_.lower() == 'quit':
        break
    res, typ = response(input_)
    print('Bot: {}'.format(res))
    print()
"""


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

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
model.fit(training, output, n_epoch=1500, batch_size=8, show_metric=True)

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
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I don't know :C")



chat()