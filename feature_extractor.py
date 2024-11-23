from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import nltk
import json


def list_of_lists(sentences):
    tokenized_sentences=[]
    for sentence in sentences:
        tokenized_sentences.append(nltk.word_tokenize(sentence))
    return tokenized_sentences

def train_gensim_model(corpus):
    tokenized_sentences=list_of_lists(corpus)
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    print(tokenized_sentences[:10])
    model.build_vocab(tokenized_sentences)
    model.train(corpus_iterable=tokenized_sentences,total_examples=model.corpus_count,epochs=30)
    return model

def read_data(path):
    with open(path,'r') as file:
        data = json.load(file)
    return data

def fix_json_file(path):
    fixed_file=open("fixed_"+path,"a")
    fixed_file.write("[\n")
    with open(path,'r') as file:
        for line in file:
            fixed_file.write(line[:-1]+",\n")
    fixed_file.seek(fixed_file.tell()-3)
    fixed_file.truncate()
    fixed_file.write("]")
    fixed_file.close()