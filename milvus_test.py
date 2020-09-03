from milvus import Milvus, IndexType, MetricType, Status
from transformers import BertTokenizer
import pandas as pd
import torch
import numpy
# define tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# pandas read excel
df = pd.read_excel("pdQA_backup.xlsx") 
Q = df["question"].str.replace("Q_","")
A = df["answer"].str.replace("A_","")  

# 建立連線
milvus = Milvus(host='localhost', port='19530')

# 創建table
param = {'collection_name':'Bert_Q', 'dimension':129, 'index_file_size':1024, 'metric_type':MetricType.L2}
# print(milvus.create_collection(param))

# convert text to id_list
input_id_list = []
for q in Q:
    tokenize_splitQ = tokenizer.tokenize(q) # 切割q
    input_id = tokenizer.convert_tokens_to_ids(tokenize_splitQ) # conver to id
    while len(input_id) <= 128:
        input_id.append(0)
    input_id_list.append(input_id)

vectors = [i for i in input_id_list]
# print(numpy.array(vectors).shape)

# 插入向量
vector_ids = [id for id in range(130)]
# status, ids = milvus.insert(collection_name='Bert_Q',records=input_id_list,ids=vector_ids)
# print(status, ids)

# 設定索引類型
flat_param = {'nlist': 130}
milvus.create_index('Bert_Q', IndexType.FLAT, flat_param)

# 刪除table
# milvus.drop_collection(collection_name='Bert_Q')

# 測試
search_param = {'nprobe': 50}
q = '血液透析可能併發症'
tokenize_q = tokenizer.tokenize(q)
q_records = tokenizer.convert_tokens_to_ids(tokenize_q)
while len(q_records) <= 128:
    q_records.append(0)
q_list = []
q_list.append(q_records)
# print(q_list)
# print(numpy.array(q_list).shape)
print("Question : ",q_records)

status , result = milvus.search(collection_name='Bert_Q', query_records=q_list, top_k=1, params=search_param)
print(status , result)

# 透過ID找向量
print("Vector : ", milvus.get_entity_by_id(collection_name='Bert_Q',ids=[47]))
print()


milvus.disconnect