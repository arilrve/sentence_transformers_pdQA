from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle

#setting embedding
embedder = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# pickle read pkl_file
pkl_file = pkl_file = open('data_features.pkl', 'rb')
data_features = pickle.load(pkl_file)

question_list = data_features['Question_list']
question_embedding = data_features['Question_embedding']
answer_list = data_features['Answer_list']



# test_predict
while(True):
    print("如果要離開請輸入 exit ")
    print("==========================================")
    queries = input("Enter your question : ")
    if queries != "exit":
        top_k=3

        query_embedding = embedder.encode(queries, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, question_embedding)[0]
        cos_scores = cos_scores.cpu()

        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", queries)
        for idx in top_results[0:top_k]:
                # print(idx)
                print("你可能提問的問題 : ",question_list[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
                for index , answer in answer_list:
                    if idx==index:
                        print("Answer : ", answer)
                        print("=============================")
    else :
        print("========結束程式=========")
        break