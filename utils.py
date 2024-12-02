# in this notebook we will do some preprocessing on the data and tokenization
import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

def clean_string(input_string):
    """
    Cleans the input string by removing special characters, and unnecessary punctuation.

    Args:
        input_string: The string to be cleaned.

    Returns:
        The cleaned string.
    """
    # Remove special characters and unnecessary punctuation
    # TODO: Add more special characters as needed to be excluded
    cleaned_string = re.sub(r"[^\w\s'\-,.]", " ", input_string)
    cleaned_string = cleaned_string.lower()
    # Remove extra whitespace
    cleaned_string = re.sub(r"\s+", " ", cleaned_string).strip()
    return cleaned_string

def tokenize_string(input_string):
    """
    Tokenizes the input string into tokens.
    
    Args:
        input_string: The string to be tokenized.

    Returns:
        A list of tokens.
    """
    tokens = nltk.word_tokenize(input_string)
    # stemming
    stemmer = PorterStemmer()
    cleaned_tokens = []
    entity_intent = ["NUMBER", "SIZE", "TOPPING", "STYLE", "DRINKTYPE", "CONTAINERTYPE", "VOLUME", "QUANTITY", "NOT", "PIZZAORDER", "DRINKORDER", "COMPLEX_TOPPING", "ORDER"]
    for token in tokens:
        if token in entity_intent:
            cleaned_tokens.append(token)
            continue
        stem_word = stemmer.stem(token) 
        cleaned_tokens.append(stem_word)
    return cleaned_tokens

def label_tokens1(input_tokens, structure_tokens):
    """
    Labels the input text based on a structured representation and a list of attributes.

    Args:
        input_tokens: The tokenized input text.
        structure_text: The structured text containing attributes and their values as tokens.

    Returns:
        A list of tuples where each token in the input text is paired with its corresponding label.
    """
    attribute_values = {"NUMBER", "SIZE", "TOPPING", "STYLE", "DRINKTYPE", "CONTAINERTYPE", "VOLUME", "QUANTITY"}
    execluded = ["PIZZAORDER", "DRINKORDER", "COMPLEX_TOPPING"]
    token_label = []
    curr_attr = "NONE"
    is_not_topping = False
    not_parentheses = 0
    is_begin = True
    for struct_token in structure_tokens:
        if struct_token == "NOT":
            is_not_topping = True
            continue
        if struct_token == "(" and is_not_topping:
            not_parentheses += 1
        if struct_token == ")" and is_not_topping:
            not_parentheses -= 1
        if not_parentheses == 0:
            is_not_topping = False

        if struct_token in attribute_values:
            curr_attr = struct_token
            is_begin = True
        elif struct_token not in {"(", ")"} and struct_token not in execluded:
            if curr_attr == "NONE":
                continue
            label = curr_attr
            if is_not_topping:
                label = "NOT_" + curr_attr
            if is_begin:
                label="B_" + label
                is_begin = False
            else:
                label="I_" + label
            token_label.append((struct_token, label))
    
    token_label_counter = 0 
    entity_to_num = {"I_NUMBER": 0, "I_SIZE": 1, "I_TOPPING": 2, "I_STYLE": 3, "I_DRINKTYPE": 4, "I_CONTAINERTYPE": 5, "I_VOLUME": 6, "I_QUANTITY": 7, "B_NUMBER": 8, "B_SIZE": 9, "B_TOPPING": 10, "B_STYLE": 11, "B_DRINKTYPE": 12, "B_CONTAINERTYPE": 13, "B_VOLUME": 14, "B_QUANTITY": 15, "I_NOT_TOPPING": 16, "B_NOT_TOPPING": 17,"I_NOT_STYLE": 18, "B_NOT_STYLE": 19, "B_NOT_QUANTITY": 20, "I_NOT_QUANTITY": 21, "NONE": 22}
    label_input=[]
    label_input_nums = []
    for in_token in input_tokens:
        if token_label_counter >= len(token_label):
            label_input.append((in_token,"NONE"))
            label_input_nums.append(entity_to_num["NONE"])
            continue
        if token_label[token_label_counter][0] == in_token:
            label_input.append((in_token,token_label[token_label_counter][1]))
            label_input_nums.append(entity_to_num[token_label[token_label_counter][1]])
            token_label_counter += 1
        else:
            label_input.append((in_token,"NONE"))
            label_input_nums.append(entity_to_num["NONE"])
    return label_input, label_input_nums
            
def label_tokens2(input_tokens, structure_tokens):
    """
    Labels the input text based on a structured representation and a list of attributes.

    Args:
        input_tokens: The tokenized input text.
        structure_text: The structured text containing attributes and their values.

    Returns:
        A list of tuples where each token in the input text is paired with its corresponding label.
    """
    attributes = ["PIZZAORDER", "DRINKORDER", "COMPLEX_TOPPING"]
    execluded = {"ORDER","NUMBER", "SIZE", "TOPPING", "STYLE", "DRINKTYPE", "CONTAINERTYPE", "VOLUME", "QUANTITY", "NOT"}
    curr = "NONE"
    # I will also keep tracking "(" and ")" to know when to change the current attribute to NONE
    parentheses =0
    is_begin = True
    labels_mapping = []
    for token in structure_tokens:
        if token in attributes:
            curr = token
            is_begin = True
        elif token == "(":
            parentheses += 1
        elif token == ")":
            parentheses -= 1
            if parentheses == 1:
                curr = "NONE"
        elif token not in execluded:
            if curr == "NONE":
                labels_mapping.append((token,curr))
            elif is_begin:
                labels_mapping.append((token, "B_" + curr))
                is_begin = False
            else:
                labels_mapping.append((token,"I_" + curr))
    labeled_output = []
    labeled_output_nums =[]
    labeled_output_counter = 0
    intent_to_num = {"I_PIZZAORDER": 0, "I_DRINKORDER": 1, "I_COMPLEX_TOPPING": 2, "B_PIZZAORDER": 3, "B_DRINKORDER": 4, "B_COMPLEX_TOPPING": 5, "NONE": 6}
    for token in input_tokens:
        if labeled_output_counter >= len(labels_mapping):
            labeled_output.append((token, "NONE"))
            labeled_output_nums.append(intent_to_num["NONE"])
            continue
        if labels_mapping[labeled_output_counter][0] == token:
            labeled_output.append((token, labels_mapping[labeled_output_counter][1]))
            labeled_output_nums.append(intent_to_num[labels_mapping[labeled_output_counter][1]])
            labeled_output_counter += 1
        else:
            labeled_output.append((token, "NONE"))
            labeled_output_nums.append(intent_to_num["NONE"])
    return labeled_output, labeled_output_nums

def label_input(input_text, structure_text1, structure_text2):
    """
    It is a similar function to the previous one, but it is used for adding another layer for the input
    which is the preprocessing of the input text and then tokenizing it.

    Args:
        input_text: The raw input text.
        structure_text1: The structured text containing attributes and their values. (train.TOP-DECOUPLED)
        structure_text2: The structured text containing attributes and their values. (train.TOP)
    
    Returns:
        2 lists of tuples where each token in the input text is paired with its corresponding label.
    """
    cleaned_text = clean_string(input_text)
    input_tokens = tokenize_string(cleaned_text)
    structure1_tokens = tokenize_string(structure_text1)
    labeled_output1, _ = label_tokens1(input_tokens, structure1_tokens)
    structure2_tokens = tokenize_string(structure_text2)
    labeled_output2 , _= label_tokens2(input_tokens, structure2_tokens)
    return labeled_output1, labeled_output2

def label_complete_input (input_list, structure_text1_list, structure_text2_list):
    """
    It is a similar function to the previous one, but it takes inputs as lists of tokens instead of strings.

    Args:
        input_text: The raw input text.
        structure_text1: The structured text containing attributes and their values. (train.TOP-DECOUPLED)
        structure_text2: The structured text containing attributes and their values. (train.TOP)
    
    Returns:
        2 lists of tuples where each token in the input text is paired with its corresponding label.
        1 list of tokens for input text.
    """
    labeled_output1 = []
    labeled_output2 = []
    list_of_tokens = []
    for text, struct1, struct2 in zip(input_list, structure_text1_list, structure_text2_list):
        cleaned_text = clean_string(text)
        input_tokens = tokenize_string(cleaned_text)
        list_of_tokens.append(input_tokens)
        structure1_tokens = tokenize_string(struct1)
        _, labels = label_tokens1(input_tokens, structure1_tokens)
        labeled_output1.append(labels)
        structure2_tokens = tokenize_string(struct2)
        _, labels = label_tokens2(input_tokens, structure2_tokens)
        labeled_output2.append(labels)
    return labeled_output1, labeled_output2, list_of_tokens


def get_train_dataset(data):
    """
    Builds a training corpus from a JSON-like dataset.
    Extracts the "train.SRC" field from each item in the dataset.

    Args:
        data: List of dictionaries, where each dictionary contains a "train.SRC" key.

    Returns:
        A list of strings representing the training corpus.
    """
    src, top, decoupled = [], [], []
    for d in data:
        src.append(d["train.SRC"])
        top.append(d["train.TOP"])
        decoupled.append(d["train.TOP-DECOUPLED"])
    return src, top, decoupled

def get_dev_dataset(data):
    """
    Builds a development corpus from a JSON-like dataset.
    Extracts the "dev.SRC" and "dev.TOP" field from each item in the dataset.

    Args:
        data: List of dictionaries, where each dictionary contains a "dev.SRC" key and a "dev.TOP" key.

    Returns:
        2 lists of strings representing the development corpus.
    """
    src = []
    top = []
    for d in data:
        src.append(d["dev.SRC"])
        top.append(d["dev.TOP"])
    return src, top

def read_data(path):
    """
    Reads a JSON file and loads its content into a Python object.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data as a Python object.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def label_tokens_dev(input_tokens, structure_tokens):
    """
    Labels the input text based on a structured representation and a list of attributes.

    Args:
        input_tokens: The tokenized input text.
        structure_text: The structured text containing attributes and their values as tokens.

    Returns:
        A list of tuples where each token in the input text is paired with its corresponding label.
    """
    attribute_values = {"NUMBER", "SIZE", "TOPPING", "STYLE", "DRINKTYPE", "CONTAINERTYPE", "VOLUME", "QUANTITY"}
    execluded = ["PIZZAORDER", "DRINKORDER", "COMPLEX_TOPPING"]
    token_label = []
    curr_attr = "NONE"
    is_not_topping = False
    not_parentheses = 0
    is_begin = True
    for struct_token in structure_tokens:
        if struct_token == "NOT":
            is_not_topping = True
            continue
        if struct_token == "(" and is_not_topping:
            not_parentheses += 1
        if struct_token == ")" and is_not_topping:
            not_parentheses -= 1
        elif struct_token == ")" :
            curr_attr = "NONE"
            is_begin = True

        if not_parentheses == 0:
            is_not_topping = False

        if struct_token in attribute_values:
            curr_attr = struct_token
            is_begin = True
        elif struct_token not in {"(", ")"} and struct_token not in execluded:
            if curr_attr == "NONE":
                continue
            label = curr_attr
            if is_not_topping:
                label = "NOT_" + curr_attr
            if is_begin:
                label="B_" + label
                is_begin = False
            else:
                label="I_" + label
            token_label.append((struct_token, label))
    
    token_label_counter = 0 
    entity_to_num = {"I_NUMBER": 0, "I_SIZE": 1, "I_TOPPING": 2, "I_STYLE": 3, "I_DRINKTYPE": 4, "I_CONTAINERTYPE": 5, "I_VOLUME": 6, "I_QUANTITY": 7, "B_NUMBER": 8, "B_SIZE": 9, "B_TOPPING": 10, "B_STYLE": 11, "B_DRINKTYPE": 12, "B_CONTAINERTYPE": 13, "B_VOLUME": 14, "B_QUANTITY": 15, "I_NOT_TOPPING": 16, "B_NOT_TOPPING": 17,"I_NOT_STYLE": 18, "B_NOT_STYLE": 19, "B_NOT_QUANTITY": 20, "I_NOT_QUANTITY": 21, "NONE": 22}
    label_input=[]  
    label_input_nums = []
    for in_token in input_tokens:
        if token_label_counter >= len(token_label):
            label_input.append((in_token,"NONE"))
            label_input_nums.append(entity_to_num["NONE"])
            continue
        if token_label[token_label_counter][0] == in_token:
            label_input.append((in_token,token_label[token_label_counter][1]))
            label_input_nums.append(entity_to_num[token_label[token_label_counter][1]])
            token_label_counter += 1
        else:
            label_input.append((in_token,"NONE"))
            label_input_nums.append(entity_to_num["NONE"])
    return label_input, label_input_nums

def label_complete_dev (input_list, structure_text_list):
    """
    This function is used for labeling the development data.

    Args:
        input_list: The raw input text.
        structure_text1: The structured text containing attributes and their values. (dev.TOP)
    
    Returns:
        1 list of tuples where each token in the input text is paired with its corresponding label.
        1 list of tokens for input text.
    """
    labeled_output = []
    list_of_tokens = []
    for text, struct in zip(input_list, structure_text_list):
        cleaned_text = clean_string(text)
        input_tokens = tokenize_string(cleaned_text)
        list_of_tokens.append(input_tokens)
        structure1_tokens = tokenize_string(struct)
        _, labels = label_tokens_dev(input_tokens, structure1_tokens)
        labeled_output.append(labels)
    return labeled_output, list_of_tokens

def calc_accuracy(corpus, model_out, gold_labels, NUM_CLASSES=23):
    """
    Calculates the accuracy of the model.

    Args:
        preds: The predicted labels.
        labels: The true labels.

    Returns:
        Confusion matrix
        The accuracy of the model.
    """
    confusion_matrix = [[0 for i in range(NUM_CLASSES)] for j in range(NUM_CLASSES)]
    # each row in model_out is a sequence of labels for a sentence, loop over all sequences and for each sequence loop over all labels
    for i in range(len(model_out)):
        do_print = False
        for j in range(len(model_out[i])):
            confusion_matrix[model_out[i][j]][gold_labels[i][j]] += 1
            if model_out[i][j] != gold_labels[i][j]:
                do_print = True
                print("Wrong prediction in", i, "th sentence at", j, "th token")
        if do_print:
            print("Sentence:", corpus[i])
            print("Pred:", model_out[i])
            print("True:", gold_labels[i])

    correct = 0
    total = 0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                correct += confusion_matrix[i][j]
            total += confusion_matrix[i][j]
    return confusion_matrix, 1.0*correct / total
