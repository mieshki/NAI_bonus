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
import numpy
import tflearn
import random
import json
import pickle
from config import *


def trainModel():
    """
    This function uses prepared json file to train chatbot model.
    Input file contains intents with responses and answers in json format.
    Here we decode all words and create arrays with tags and words.

    :return:
    model this is our AI model that will chat with human,
    all_tags array will all tags from our intent file,
    data loaded and formatted json file,
    all_intents_words array with stemmed words  without duplicates
    """

    stemmer = LancasterStemmer()

    with open(f'data\\{INTENT_FILE_TO_TRAIN}') as file:
        data = json.load(file)

    try:
        with open(f'model\\{PICKLE_FILE_NAME}', 'rb') as f:
            all_intents_words, all_tags, training, output = pickle.load(f)
    except:
        all_intents_words = []
        all_tags = []
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                all_intents_words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in all_tags:
                all_tags.append(intent["tag"])

        all_intents_words = [stemmer.stem(w.lower()) for w in all_intents_words if w != "?"]
        all_intents_words = sorted(list(set(all_intents_words)))

        all_tags = sorted(all_tags)

        training = []
        output = []

        out_empty = [0 for _ in range(len(all_tags))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in all_intents_words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[all_tags.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = numpy.array(training)
        output = numpy.array(output)

        with open(f'model\\{PICKLE_FILE_NAME}', 'wb') as f:
            pickle.dump((all_intents_words, all_tags, training, output), f)

    """ Load existing model or train and save new one """
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
        with open(f'.\\model\\{MODEL_FILE_NAME}.index', 'r'):
            print("Available")
        model.load(f'.\\model\\{MODEL_FILE_NAME}')

    except:
        # 10% of training data used for validation
        #model.fit(X, Y, validation_set=0.1)
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True, validation_set=0.1)
        model.save(f'model\\{MODEL_FILE_NAME}')

    return [model, all_tags, data, all_intents_words]


def bag_of_words(input, words):
    """
    This function checks how many words from input string are available in our words list.
    It returns array of 0 and 1 where 1 menas that words from input exist in words list and 0 it isn't
    :param input: Input string form user
    :param words: Array with stemmed words without duplicates
    """
    stemmer = LancasterStemmer()

    bag = [0 for _ in range(len(words))]

    input_words = nltk.word_tokenize(input)
    input_words = [stemmer.stem(word.lower()) for word in input_words]

    for input_word in input_words:
        for i, word in enumerate(words):
            if word == input_word:
                bag[i] = 1

    return numpy.array(bag)


def normalize_polish_characters(str):
    """
    This function replaces all polish characters from user's input
    :param str: Message from user
    :return: String without polish characters
    """
    return str.replace('ą', 'a').replace('ć', 'c').replace('ę', 'e').replace('ł', 'l').replace('ń', 'n').replace('ó', 'o').replace(
        'ś', 's').replace('ź', 'z').replace('ż', 'z')


def chat(model, tags, data, words):
    """
    This function is responsible for chat with users. Answer is predicted basing on our trained model.
    It saves all conversation to provide more patterns and responses for future development.
    :param model: Trained model used to predict answers
    :param tags: Array with all tags available in intents source
    :param data: loaded and formatted json file with all intents raw data
    :param words: array with stemmed words without duplicates
    """
    print("Start talking with the bot (type quit to stop)!")

    while True:
        with open(f'logs\\{CHAT_HISTORY_LOG_FILE_NAME}', 'a') as f:
            inp = input('You: ')

            if inp.lower() == 'quit':
                break

            #print(f'[DEBUG] Input before: {inp}')
            inp = normalize_polish_characters(inp.lower())
            #print(f'[DEBUG] Input after: {inp}')

            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = tags[results_index]

            f.write(f'Input: {inp}\n')

            if results[results_index] > CONFIDENCE_LEVEL:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                answer = random.choice(responses)
                print(answer)
                f.write(f'bot_answer: {answer}\n')
            else:
                print('I don\'t know :C')
                f.write('bot_answer: Don\'t know\n')


if __name__ == '__main__':
    chat(*trainModel())
