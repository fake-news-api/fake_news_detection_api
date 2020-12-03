# Libraries

import pandas as pd
import torch
import numpy as np
np.random.seed(2020)

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

# warnings
import warnings
warnings.filterwarnings('ignore')


# time
from time import time

model_folder = './BERT/Model/'
data_folder = './BERT/Data/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

names = ['text', 'authortext', 'titletext', 'authortitletext']
names = ['text']
models = {}
for name in names:
  print("Model:", name)
  model = BERT().to(device)
  load_checkpoint(model_folder + '/' + name + '_model.pt', model)
  models[name] = model
  print("")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('label', label_field), ('text', text_field)]


def predict(text, title = "", author = ""):
  try:
    item = ""
    data_var = ""
    
    if author:
      item += author + ". "
      data_var += "author"
    
    if title:
      item += title + ". "
      data_var += "title"

    if text:
      item += text
      data_var += "text"
    # print(item)
    model = models[data_var]

    pd.DataFrame([[item]], columns=['text']).to_csv(data_folder + "data.csv")

    # print(pd.read_csv(data_folder + "data.csv"))

    data = TabularDataset.splits(path=data_folder, train='data.csv', format='CSV', fields=fields, skip_header=True)
    # print(data)
    data_iter = Iterator(data[0], batch_size=1, device=device, train=False, shuffle=False, sort=False)

    model.eval()

    y_pred = []
    with torch.no_grad():
      for (labels, text), _ in data_iter:
        labels = labels.type(torch.LongTensor)           
        labels = labels.to(device)

        current_var = text
        current_var = current_var.type(torch.LongTensor)  
        current_var = current_var.to(device)
        output = model(current_var, labels)

        _, output = output
        y_pred.extend(torch.argmax(output, 1).tolist())
    return y_pred[0]
  except:
    return "error"
  # return output
