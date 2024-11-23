from gensim.models import Word2Vec
import gensim
import nltk
import json
from transformers import BertTokenizer, BertModel
import torch

VECTOR_SIZE = 50
WINDOW_SIZE = 5
THREADS = 4
CUTOFF_FREQ = 1
EPOCHS = 100


# Takes a list of sentences and return a list of tokenized sentences.
# Used for gensim
def list_of_lists(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(nltk.word_tokenize(sentence))
    return tokenized_sentences


# takes a list of sentences
# trains word to vector embedder and returns the model
def train_gensim_w2v_model(corpus):
    tokenized_sentences = list_of_lists(corpus)
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=CUTOFF_FREQ,
        workers=THREADS,
    )
    model.build_vocab(tokenized_sentences)
    model.train(
        corpus_iterable=tokenized_sentences,
        total_examples=model.corpus_count,
        epochs=EPOCHS,
    )
    return model


# reads json file
def read_data(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


# current data files are corrupted
# this makes it a proper json file
def fix_json_file(path):
    fixed_file = open("fixed_" + path, "a")
    fixed_file.write("[\n")
    with open(path, "r") as file:
        for line in file:
            fixed_file.write(line[:-1] + ",\n")
    fixed_file.seek(fixed_file.tell() - 3)
    fixed_file.truncate()
    fixed_file.write("]")
    fixed_file.close()


# loads pretrained models
def load_pretrained_model(model_name):
    model = gensim.downloader.load(model_name)
    return model


# returns word embedding
def embed_gensim(model, word):
    return model.wv[word]


# used for bert initialization
def init_bert():
    model = BertModel.from_pretrained(
        "bert-base-uncased",
        output_hidden_states=True,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


# formats text to be processable by bert
def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokenized_text, tokens_tensor, segments_tensors


# returns list of token embeddings from bert model
def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2][1:]
    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
    return list_token_embeddings


# return word embedding within text
def get_word_bert_embedding(word, text, tokenizer, model):
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(
        text, tokenizer
    )
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    word_index = tokenized_text.index(word)
    word_embedding = list_token_embeddings[word_index]
    return word_embedding
