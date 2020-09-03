from transformers import BertTokenizer
import torch.nn as nn
import torch
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

df = pd.read_excel("pdQA_backup.xlsx") #pandas read excel
Q = df["question"].str.replace("Q_","")
A = df["answer"].str.replace("A_","")  

input_id_list = []
for q in Q:
    tokenize_splitQ = tokenizer.tokenize(q) # 切割q
    input_id = tokenizer.convert_tokens_to_ids(tokenize_splitQ) # conver to id
    while len(input_id) <= 128:
        input_id.append(0)
    input_id_list.append(input_id)

input_id_list = torch.tensor(input_id_list)
print(input_id_list)

queries = ['透析是甚麼']

for i in range(input_id_list.size(0)):
    query_tokenize = tokenizer.encode(queries)
    while len(query_tokenize) <= 128:
        query_tokenize.append(0)
    
    query_tokenize = torch.tensor(query_tokenize)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    output = cos(query_tokenize,input_id_list)