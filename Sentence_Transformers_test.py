from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import pickle

embedder = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

df = pd.read_excel("pdQA_backup.xlsx") 
Q = df["question"].str.replace("Q_","")
A = df["answer"].str.replace("A_","")

# Question_list
q_list = []
for q in Q :
    q_list.append(q)

# Answer_list
a_list = []
for a in A :
    a_list.append(a)

# ansewer_list(according id to find answer)
a_list = enumerate(a_list)

# Question_embedding 
corpus_embeddings = embedder.encode(q_list, convert_to_tensor=True)

# save list to pickle
data_features = {'Question_list':q_list,
                'Question_embedding':corpus_embeddings,
                 'Answer_list':a_list,
                }
output = open('data_features.pkl', 'wb')
pickle.dump(data_features,output)

# ===============================================================================#

# question
# queries = ['何謂全自動腹膜透析']

# test_predict
# top_k=1
# for query in queries:
#     query_embedding = embedder.encode(query, convert_to_tensor=True)
#     cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
#     cos_scores = cos_scores.cpu()

#     top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

#     print("\n\n======================\n\n")
#     print("Query:", query)
#     # print("\nTop 1 most similar sentences in corpus:")
#     for idx in top_results[0:top_k]:
#         print(idx)
#         print(q_list[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
#         for index , answer in a_list:
#             if idx==index:
#                 print("Answer : ", answer)
