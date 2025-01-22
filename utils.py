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
    # TODO: Add more special characters as needed to be excluded
    cleaned_string = re.sub(r"[^\w\s'\-,.:]", " ", input_string)
    cleaned_string = cleaned_string.lower()
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

def tokenize_string_bert(input_string):
    """
    Tokenizes the input string into tokens.
    
    Args:
        input_string: The string to be tokenized.

    Returns:
        A list of tokens.
    """
    tokens = nltk.word_tokenize(input_string)
    return tokens

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
            elif parentheses == 2 and curr == "COMPLEX_TOPPING":
                curr = "PIZZAORDER"
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


def label_complete_input_bert (input_list, structure_text1_list, structure_text2_list):
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
        input_tokens = tokenize_string_bert(cleaned_text)
        list_of_tokens.append(input_tokens)
        structure1_tokens = tokenize_string_bert(struct1)
        _, labels = label_tokens1(input_tokens, structure1_tokens)
        labeled_output1.append(labels)
        structure2_tokens = tokenize_string_bert(struct2)
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
        1 list of NER labels.
        1 list of IS labels.
        1 list of tokens for input text.
    """
    ner_labeled_output = []
    is_labeled_output = []
    list_of_tokens = []
    for text, struct in zip(input_list, structure_text_list):
        cleaned_text = clean_string(text)
        input_tokens = tokenize_string(cleaned_text)
        list_of_tokens.append(input_tokens)
        structure1_tokens = tokenize_string(struct)
        _, ner_labels = label_tokens_dev(input_tokens, structure1_tokens)
        ner_labeled_output.append(ner_labels)
        _,is_labels = label_tokens2(input_tokens, structure1_tokens)
        is_labeled_output.append(is_labels)
    return ner_labeled_output,is_labeled_output, list_of_tokens

def label_complete_dev_bert (input_list, structure_text_list):
    """
    This function is used for labeling the development data.

    Args:
        input_list: The raw input text.
        structure_text1: The structured text containing attributes and their values. (dev.TOP)
    
    Returns:
        1 list of tuples where each token in the input text is paired with its corresponding label.
        1 list of tokens for input text.
    """
    ner_labeled_output = []
    is_labeled_output = []
    list_of_tokens = []
    for text, struct in zip(input_list, structure_text_list):
        cleaned_text = clean_string(text)
        input_tokens = tokenize_string_bert(cleaned_text)
        list_of_tokens.append(input_tokens)
        structure1_tokens = tokenize_string_bert(struct)
        _, ner_labels = label_tokens_dev(input_tokens, structure1_tokens)
        ner_labeled_output.append(ner_labels)
        _,is_labels = label_tokens2(input_tokens, structure1_tokens)
        is_labeled_output.append(is_labels)
    return ner_labeled_output,is_labeled_output, list_of_tokens

def intent_post_processing(model_out):
    for out in model_out:
        for i, label in enumerate(out):
            if i ==0 and out[i] == 3:
                if i+1 < len(out) and  out[i+1] in [4,6]:
                    out[i] = 6
                elif i+1 < len(out) and  out[i+1] == 2:
                    out[i] = 5
                elif i+1 < len(out) and  out[i+1] == 1:
                    out[i] = 4  
            if i>0 and i < len(out)-1:
                if out[i-1] == out[i+1] and out[i-1]==0 and out[i] not in [3,0]:
                    out[i] = out[i-1]
                elif out[i-1] == out[i+1] and out[i-1]==1 and out[i] not in [4,1]:
                    out[i] = out[i-1]
                elif out[i-1] == out[i+1] and out[i-1]==2 and out[i] not in [5,2]:
                    out[i] = out[i-1]
                elif out[i-1] == out[i+1] and out[i-1]==3 and out[i] != 0:
                    out[i] = 0
                elif out[i-1] == out[i+1] and out[i-1]==5 and out[i] != 2:
                    out[i] = 2
                elif out[i-1] == out[i+1] and out[i-1]==6 and out[i] not in [4, 6]:
                    out[i] = 6
                # elif label == 6: ###### check this
                #     if i+1 < len(out) and out[i+1] == 5:
                #         out[i] = 0
    return model_out

## only done if we need highr EM, it may decrease the word level accuracy
def intent_post_processing_extra(model_out):
    for out in model_out:
        for i, label in enumerate(out):
            if i>0:
                if label == 0 and out[i-1] not in [0,3,2,5]:
                    out[i] = 6
    return model_out   

## only done if we need highr EM, it may decrease the word level accuracy (pineapple case)
# this function will use the entity model output to correct the intent model output
def intent_post_processing2(intent_model_out, entity_model_out):
    for i, entity_out in enumerate(entity_model_out):
        for j,label in enumerate(entity_out):
            if label == 0 or label==1:
                if intent_model_out[i][j] not in [0,1]:
                    if j>0 and intent_model_out[i][j-1] in [0,3]:
                        intent_model_out[i][j-1]=0
                    elif j>0 and intent_model_out[i][j-1] in [1,4]:
                        intent_model_out[i][j-1]=1
            if  label==3 or label==18:
                intent_model_out[i][j]=0
            if label == 4 or label == 5 or label==6:
                intent_model_out[i][j]=1
            if label==7 or label==21:
                intent_model_out[i][j]=2

            if label == 10 or label == 11:
                if intent_model_out[i][j] not in [0,3,2,5]:
                    intent_model_out[i][j]=0 # or 3

            if label in [12,13,14]:
                if intent_model_out[i][j] not in [1,4]:
                    intent_model_out[i][j]=1 # or 4
            
            if label in [15, 20,21]:
                if intent_model_out[i][j] not in [2,5]:
                    intent_model_out[i][j]=2 # or 5 

            if label in [16,17]:
                if intent_model_out[i][j] not in [0,3,2,5]:
                    intent_model_out[i][j]=0 # or 3 

            if label in [18,19]:
                if intent_model_out[i][j] not in [0,3]:
                    intent_model_out[i][j]=0 # or 3 

            if label ==9 or label == 8:
                if intent_model_out[i][j] not in [0,1,3,4]:
                    if j>0 and intent_model_out[i][j-1] in [0,3]:
                        intent_model_out[i][j]=0
                    elif j>0 and intent_model_out[i][j-1] in [1,4]:
                        intent_model_out[i][j]=1
    return intent_model_out

def calc_accuracy(corpus, model_out, gold_labels, NUM_CLASSES=23):
    """
    Calculates the accuracy of the model.

    Args:
        preds: The predicted labels.
        labels: The true labels.

    Returns:
        Confusion matrix
        The accuracy of the model
        The exact match accuracy
    """
    confusion_matrix = [[0 for i in range(NUM_CLASSES)] for j in range(NUM_CLASSES)]
    exat_match = 0
    for i in range(len(model_out)):
        do_print = False
        for j in range(len(model_out[i])):
            confusion_matrix[model_out[i][j]][gold_labels[i][j]] += 1
            if model_out[i][j] != gold_labels[i][j]:
                do_print = True
                print("Wrong prediction in", i, "th sentence at", j, "th token")
        if do_print:
            exat_match += 1
            print("Sentence:", corpus[i])
            print("Pred:", model_out[i])
            print("True:", gold_labels[i])
            print("-------------------------------------------------")

    correct = 0
    total = 0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                correct += confusion_matrix[i][j]
            total += confusion_matrix[i][j]
    return confusion_matrix, 1.0*correct / total, (len(model_out)-exat_match)/len(model_out)

def total_EM(ner_out, is_out, gold_ner, gold_is):
    """
    Calculates the exact match accuracy of the model.

    Args:
        ner_out: The predicted NER labels.
        is_out: The predicted IS labels.
        gold_ner: The true NER labels.
        gold_is: The true IS labels.

    Returns:
        The exact match accuracy of the model
    """
    total = 0
    correct = 0
    for i in range(len(ner_out)):
        if ner_out[i] == gold_ner[i] and is_out[i] == gold_is[i]:
            correct += 1
        total += 1
    return 1.0*correct/total

def convert_to_json (file_name,input_tokens, entity_labels, intent_labels):
    json_map = {"ORDER":{"PIZZAORDER":[], "DRINKORDER":[]}}
    empty_pizza_order = {"ALLTOPPING":[], "STYLE":[]}
    empty_drink_order = {}
    curr_pizza_order = {"ALLTOPPING":[], "STYLE":[]}
    curr_drink_order = {}
    input_size = len(input_tokens)
    i = 0
    while(i<input_size):
        if intent_labels[i] == 3:
            if curr_pizza_order != empty_pizza_order:
                json_map["ORDER"]["PIZZAORDER"].append(curr_pizza_order)
            curr_pizza_order={"ALLTOPPING":[], "STYLE":[]}
        elif intent_labels[i]== 4:
            if curr_drink_order != empty_drink_order:
                json_map["ORDER"]["DRINKORDER"].append(curr_drink_order)
            curr_drink_order = {}
        if intent_labels[i] == 5:
            curr_topping = {"NOT":False, "QUANTITY":None}
            beg = True
            while i<input_size and (intent_labels[i] == 2  or (beg and intent_labels[i] == 5)):
                beg = False
                if entity_labels[i] == 15:
                    quantity = input_tokens[i]
                    while i+1<input_size and entity_labels[i+1] == 7:
                        i+=1
                        quantity += " " + input_tokens[i]
                    curr_topping["QUANTITY"] = quantity
                elif entity_labels[i] == 20:
                    curr_topping["NOT"] = True
                    quantity = input_tokens[i]
                    while i+1<input_size and entity_labels[i+1] == 21:
                        i+=1
                        quantity += " " + input_tokens[i]
                    curr_topping["QUANTITY"] = quantity
                if entity_labels[i] == 10:
                    topping = input_tokens[i]
                    while i+1<input_size and entity_labels[i+1] == 2:
                        i+=1
                        topping += " " + input_tokens[i]
                    curr_topping["TOPPING"] = topping
                elif entity_labels[i] == 17:
                    curr_topping["NOT"] = True
                    topping = input_tokens[i]
                    while i+1<input_size and entity_labels[i+1] == 16:
                        i+=1
                        topping += " " + input_tokens[i]
                    curr_topping["TOPPING"] = topping
                i+=1

            curr_pizza_order["ALLTOPPING"].append(curr_topping)
            continue

        # this may happen, if the complex topping is the last part of the order
        if i >= input_size:
            break
            
        if entity_labels[i]==8:
            curr_number = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==0):
                i+=1
                curr_number += " " + input_tokens[i]

            if intent_labels[i] in [3,0]:
                curr_pizza_order["NUMBER"] = curr_number
            else:
                curr_drink_order["NUMBER"] = curr_number
        elif entity_labels[i]==9:
            curr_size = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==1):
                i+=1
                curr_size += " " + input_tokens[i]
            if intent_labels[i] in [3,0]:
                curr_pizza_order["SIZE"] = curr_size
            else:
                curr_drink_order["SIZE"] = curr_size
        elif entity_labels[i]==10:
            curr_topping = {"NOT":False, "QUANTITY":None}
            topping = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==2):
                i+=1
                topping+=" " + input_tokens[i]
            curr_topping["TOPPING"] = topping
            curr_pizza_order["ALLTOPPING"].append(curr_topping)
        elif entity_labels[i]==11:
            curr_style = {"NOT":False, "TYPE":None}
            curr_style["TYPE"] = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==3):
                i+=1
                curr_style["TYPE"] += " " + input_tokens[i]
            curr_pizza_order["STYLE"].append(curr_style)
        elif entity_labels[i]==19:
            curr_style = {"NOT":True, "TYPE":None}
            curr_style["TYPE"] = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==18):
                i+=1
                curr_style["TYPE"] += " " + input_tokens[i]
            curr_pizza_order["STYLE"].append(curr_style)
        elif entity_labels[i]==17:
            curr_topping = {"NOT":True, "QUANTITY":None}
            topping = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==16):
                i+=1
                topping+=" " + input_tokens[i]
            curr_topping["TOPPING"] = topping
            curr_pizza_order["ALLTOPPING"].append(curr_topping)
        elif entity_labels[i]==12:
            curr_drink_order["DRINKTYPE"] = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==4):
                i+=1
                curr_drink_order["DRINKTYPE"] += " " + input_tokens[i]
        elif entity_labels[i]==13:
            curr_drink_order["CONTAINERTYPE"] = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==5):
                i+=1
                curr_drink_order["CONTAINERTYPE"] += " " + input_tokens[i]
        elif entity_labels[i]==14:
            curr_drink_order["VOLUME"] = input_tokens[i]
            while(i+1<input_size and entity_labels[i+1]==6):
                i+=1
                curr_drink_order["VOLUME"] += " " + input_tokens[i]
        i += 1
    if curr_pizza_order != empty_pizza_order:
        json_map["ORDER"]["PIZZAORDER"].append(curr_pizza_order)
    if curr_drink_order != empty_drink_order:
        json_map["ORDER"]["DRINKORDER"].append(curr_drink_order)
    with(open(file_name,'w')) as json_file:
        json.dump(json_map, json_file, indent=4)
    return json_map

def normalize_json(data):
    """
    Recursively sorts JSON data to normalize for comparison.
    """
    if isinstance(data, dict):
        return {k: normalize_json(v) for k, v in sorted(data.items())}
    elif isinstance(data, list):
        return sorted([normalize_json(item) for item in data], key=str)
    else:
        return data

def compare_json_files(file1, file2):
    """
    Compares two JSON files for exact match with modulo sibling order.
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    normalized_json1 = normalize_json(json1)
    normalized_json2 = normalize_json(json2)

    return normalized_json1 == normalized_json2

def total_EM_from_json(input_tokens,ner_out, is_out, gold_ner, gold_is):
    """
    Calculates the exact match accuracy of the model.

    Args:
        ner_out: The predicted NER labels.
        is_out: The predicted IS labels.
        gold_ner: The true NER labels.
        gold_is: The true IS labels.

    Returns:
        The exact match accuracy of the model
    """
    total = 0
    correct = 0
    for i in range(len(ner_out)):
        file1 = "file1.json"
        file2 = "file2.json"
        convert_to_json (file1,input_tokens[i], ner_out[i], is_out[i])
        convert_to_json (file2,input_tokens[i], gold_ner[i], gold_is[i])
        correct+=compare_json_files(file1, file2)
        total += 1
    return 1.0*correct/total

# src = "i would like to order a medium pizza with italian sausage a small pizza with beef and mushrooms a large combination pizza and four large pepsis"
# top = "(ORDER i would like to order (PIZZAORDER (NUMBER a ) (SIZE medium ) pizza with (TOPPING italian sausage ) ) (PIZZAORDER (NUMBER a ) (SIZE small ) pizza with (TOPPING beef ) and (TOPPING mushrooms ) ) (PIZZAORDER (NUMBER a ) (SIZE large ) (STYLE combination ) pizza ) and (DRINKORDER (NUMBER four ) (SIZE large ) (DRINKTYPE pepsis ) ) )"
# ner_labeled_output,is_labeled_output, list_of_tokens=label_complete_dev([src], [top])
# convert_to_json("new.json",tokenize_string_bert(src), ner_labeled_output[0], is_labeled_output[0])
