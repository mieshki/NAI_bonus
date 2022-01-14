from json import load, dump


def formater():
    """
    This function allow to format files to needed json format
    It creates Json input file used to train chatbot model
    """
    file = open('../data/intent.json')
    json_data = load(file)

    """
    {
        "intents": [
            {"tag": "greeting",
             "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day", "Whats up"],
             "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
             "context_set": ""
            },
    """

    """
    
    {
      "intents": [
            {
                  "intent": "Greeting",
                  "text": [
                        "Hi",
                  ],
                  "responses": [
                        "Hi human, please tell me your GeniSys user",
                  ],
                  "extension":  {
                        "function": "",
                        "entities": false,
                        "responses": []
                  },
                  "context":  {
                        "in": "",
                        "out": "GreetingUserRequest",
                        "clear": false
                  },
    
    """


    # intent -> tag
    # patterns -> text
    # responses -> responses
    # context -> context

    data = {'intents': []}

    all_intents = []

    for intent in json_data['intents']:
        temp_intent = {}
        temp_intent['tag'] = intent['intent']
        temp_intent['patterns'] = intent['text']
        temp_intent['responses'] = intent['responses']
        temp_intent['context'] = intent['context']

        all_intents.append(temp_intent)

    print(all_intents)

    with open('intents-translated.json', 'w') as f:
        dump(all_intents, f)

    #json_data = json.dumps(data)
