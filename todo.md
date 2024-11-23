- [ ] Data Preprocessing: Data cleaning, Tokenization
- [ ] Parse the raw dataset into a trainable structure (Create a map between the raw input and its corresponding label)
- [ ] Map example:
```json
map: {
"can":"none",
"i" : "none",
..
"large": "SIZE"
}
```
- [ ] Extract features like word embeddings and contextual embeddings
- [ ] Try different models like RNN, LSTM, BiLSTM, GRU 
- [ ] Try different optimizers like Adam, SGD
- [ ] Handle cases like "bbq pulled pork"