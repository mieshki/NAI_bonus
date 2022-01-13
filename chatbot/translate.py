from json import load, dump
from googletrans import Translator

from deep_translator import GoogleTranslator
#

def translate(input):
    #service_urls=['translate.googleapis.com']
    #https://translate.google.pl/?hl=pl
    #translator = Translator()
    #translator = Translator(service_urls=['translate.google.pl/?hl=pl'])
    print(f'Input: {input}')
    #output = translator.translate(input, src='en', dest='pl')
    output = GoogleTranslator(source='en', target='pl').translate(input)

    print(f'Translating sentence:')
    print(f'\t [EN]: {input}')
    print(f'\t [PL]: {output}')
    print('=' * 20)

    return output


def intent_translator():
    file = open('intents-eng.json')
    json_data = load(file)

    data = {}
    data['intents'] = []

    all_intents = []

    for intent in json_data['intents']:
        temp_intent = {}
        temp_intent['tag'] = intent['tag']
        temp_intent['patterns'] = intent['patterns']
        temp_intent['responses'] = intent['responses']
        temp_intent['context'] = intent['context']

        all_intents.append(temp_intent)

    for intent in all_intents:
        intent['tag'] = 'pl-' + intent['tag']

        for i, pattern in enumerate(intent['patterns']):
            intent['patterns'][i] = translate(pattern)
            #translate(pattern)
            #break

        for i, response in enumerate(intent['responses']):
            intent['responses'][i] = translate(response)
            #response = translate(response)
            #break

    print(all_intents)

    with open('intents-adv-pl-test.json', 'w') as f:
        dump(all_intents, f)

    # json_data = json.dumps(data)