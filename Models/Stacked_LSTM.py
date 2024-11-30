import utils
import feature_extractor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pickle

NUM_CLASSES=23
BATCH_SIZE=64
EPOCHS=10
HIDDEN_SIZE=512
NUM_LAYERS=4
VECTOR_SIZE = 50  # Size of word vectors
WINDOW_SIZE = 5  # Context window size
THREADS = 4  # Number of threads to use for training
CUTOFF_FREQ = 1  # Minimum frequency for a word to be included in vocabulary
TRAINING_SIZE = 100000  
TEST_SIZE = 10
DROP_OUT=0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

complete_data=utils.read_data("../data/fixed_PIZZA_train.json")
data = complete_data[:TRAINING_SIZE]
corpus, top, decoupled = utils.get_train_dataset(data)
entites_output_as_number_labels,intents_output_as_number_labels, input_as_tokenized_string=utils.label_complete_input(corpus, decoupled, top)

emb_model = feature_extractor.train_gensim_w2v_model(input_as_tokenized_string, VECTOR_SIZE)

class LargeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data 
        self.labels = labels 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    #I believe we can transform words into embeddings here
    embeddings=[]
    # print(sequences)
    for seq in sequences:
        x=[]
        for token in seq:
            x.append(emb_model.wv[token])
        embeddings.append(x)
    sequences=embeddings
    labels = [torch.tensor(label, dtype=torch.long) for label in labels]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    sequences = [torch.tensor(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return padded_sequences, padded_labels, lengths

class LargeWordRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LargeWordRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,num_layers=NUM_LAYERS, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.fc(out)
        return out

entites_labels = entites_output_as_number_labels

dataset = LargeDataset(input_as_tokenized_string, entites_labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=0)

model = LargeWordRNN(input_size=VECTOR_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(EPOCHS): 
    for padded_sequences, padded_labels, lengths in dataloader:
        padded_sequences=padded_sequences.to(device)
        padded_labels=padded_labels.to(device)
        lengths=lengths.to(device)
        optimizer.zero_grad()
        outputs = model(padded_sequences, lengths)
        loss = criterion(outputs.view(-1, NUM_CLASSES), padded_labels.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
pickle.dump(model , open('model_500k.pk1' , 'wb'))

# Testing
test_data=complete_data[TRAINING_SIZE:TRAINING_SIZE+TEST_SIZE]
print(test_data)
print("---------------------------------------------")
test_corpus,_,_= utils.get_train_dataset(test_data)
print(test_corpus)
test_as_tokenized_string=feature_extractor.list_of_lists(test_corpus)
print(test_as_tokenized_string)

class TestLargeDataset(Dataset):
    def __init__(self, data):
        self.data = data 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def test_collate_fn(batch):
    sequences = batch
    #I believe we can transform words into embeddings here
    embeddings=[]
    for seq in sequences:
        print(seq)
        x=[]
        for token in seq:
            x.append(emb_model.wv[token])
        embeddings.append(x)
    sequences=embeddings
    sequences = [torch.tensor(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return padded_sequences, lengths


test_as_tokenized_string
dataset = TestLargeDataset(test_as_tokenized_string)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=test_collate_fn, shuffle=False, num_workers=0)

for padded_sequences, lengths in dataloader:
    padded_sequences=padded_sequences.to(device)
    lengths=lengths.to(device)
    outputs = model(padded_sequences, lengths)
    entity_to_num = {"I_NUMBER": 0, "I_SIZE": 1, "I_TOPPING": 2, "I_STYLE": 3, "I_DRINKTYPE": 4, "I_CONTAINERTYPE": 5, "I_VOLUME": 6, "I_QUANTITY": 7, "B_NUMBER": 8, "B_SIZE": 9, "B_TOPPING": 10, "B_STYLE": 11, "B_DRINKTYPE": 12, "B_CONTAINERTYPE": 13, "B_VOLUME": 14, "B_QUANTITY": 15, "I_NOT_TOPPING": 16, "B_NOT_TOPPING": 17, "NONE": 18}
    for i, out in enumerate(outputs[0]):
        num = torch.argmax(out).int().item()
        for key, value in entity_to_num.items():
            if value == num:
                print(key)
                break
    print("-------------------------------")
