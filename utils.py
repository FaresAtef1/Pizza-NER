# in this notebook we will do some preprocessing on the data and tokenization
import re
import nltk
import json

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
    cleaned_string = re.sub(r"[^\w\s'-]", "", input_string)  # Keeps only alphanumeric characters and spaces and apostrophes and hyphens
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
    return tokens

def label_tokens1(input_tokens, structure_text):
    """
    Labels the input text based on a structured representation and a list of attributes.

    Args:
        input_tokens: The tokenized input text.
        structure_text: The structured text containing attributes and their values.

    Returns:
        A list of tuples where each token in the input text is paired with its corresponding label.
    """
    attribute_values = {"NUMBER", "SIZE", "TOPPING", "STYLE", "DRINKTYPE", "CONTAINERTYPE", "VOLUME", "QUANTITY"}
    structure_map = {}
    for attribute in attribute_values:
        # Match the attribute and its value in the structure text
        pattern = r"\(\s*"+ attribute + r"\s+([^\)]*)\s*\)"
        matches = re.finditer(pattern, structure_text)
        for match in matches:
            value = match.group(1).strip()
            # Special handling for TOPPING with "not" before it
            if attribute == "TOPPING":
                preceding_text = structure_text[:match.start()]
                if re.search(r"\bNOT\b\s*\($", preceding_text, re.IGNORECASE):
                    attribute = "NOT_TOPPING"
            structure_map[value] = attribute
    labeled_output = []
    labeled_output_nums = []
    entity_to_num = {"I_NUMBER": 0, "I_SIZE": 1, "I_TOPPING": 2, "I_STYLE": 3, "I_DRINKTYPE": 4, "I_CONTAINERTYPE": 5, "I_VOLUME": 6, "I_QUANTITY": 7, "B_NUMBER": 8, "B_SIZE": 9, "B_TOPPING": 10, "B_STYLE": 11, "B_DRINKTYPE": 12, "B_CONTAINERTYPE": 13, "B_VOLUME": 14, "B_QUANTITY": 15, "I_NOT_TOPPING": 16, "B_NOT_TOPPING": 17, "NONE": 18}
    for token in input_tokens:
        label = "NONE"
        if token in structure_map:
            label = structure_map[token]
            label = "B_" + label
        # else check if it is part of the key
        else:
            for key in structure_map.keys():
                if token in key.split():
                    label = structure_map[key]
                    if token == key.split()[0]:
                        label = "B_" + label
                    else:
                        label = "I_" + label
                    break
        labeled_output.append((token, label))
        labeled_output_nums.append(entity_to_num[label])
    return labeled_output, labeled_output_nums

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
    execluded = {"NUMBER", "SIZE", "TOPPING", "STYLE", "DRINKTYPE", "CONTAINERTYPE", "VOLUME", "QUANTITY"}
    curr = "NONE"
    # I will also keep tracking "(" and ")" to know when to change the current attribute to NONE
    parentheses =0
    is_begin = True
    labels_mapping = {}
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
                labels_mapping[token] = curr
            elif is_begin:
                labels_mapping[token] = "B_" + curr
                is_begin = False
            else:
                labels_mapping[token] = "I_" + curr
    labeled_output = []
    labeled_output_nums =[]
    intent_to_num = {"I_PIZZAORDER": 0, "I_DRINKORDER": 1, "I_COMPLEX_TOPPING": 2, "B_PIZZAORDER": 3, "B_DRINKORDER": 4, "B_COMPLEX_TOPPING": 5, "NONE": 6}
    for token in input_tokens:
        label = "NONE"
        if token in labels_mapping:
            label = labels_mapping[token]
        labeled_output.append((token, label))
        labeled_output_nums.append(intent_to_num[label])
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
    labeled_output1 = label_tokens1(input_tokens, structure_text1)
    structure2_tokens = tokenize_string(structure_text2)
    labeled_output2 = label_tokens2(input_tokens, structure2_tokens)
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
        List of tokens for each input text.
    """
    labeled_output1 = []
    labeled_output2 = []
    list_of_tokens = []
    for text, struct1, struct2 in zip(input_list, structure_text1_list, structure_text2_list):
        cleaned_text = clean_string(text)
        input_tokens = tokenize_string(cleaned_text)
        list_of_tokens.append(input_tokens)
        _, labels = label_tokens1(input_tokens, struct1)
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
