import torch
import csv
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearnex import patch_sklearn
from sklearn.metrics.pairwise import cosine_similarity



class Config:
    def __init__(self):
        self.name = 'Sentence_bert_cos'
        self.bert_path = 'data/sentence-transformers_all-mpnet-base-v2'
        self.data_path = 'data/' 
        self.train_path = os.path.join(self.data_path, 'train.csv')
        self.theta = 0.3

def data_process(data_path):
    data = list()
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            data.append(row[0])
    return data


def gen_konwledge(model, data, theta):
    data_base = list()
    data_text = list()
    
    data_base.append(model.encode(data[0]))
    data_text.append(data[0])

    par = tqdm(data, leave=True)
    for t in par:
        t_code = model.encode(t)
        sen_similarity = cosine_similarity([t_code], data_base)
        if sen_similarity.max() < theta:
            data_base.append(t_code)
            data_text.append(t)
    return data_base, data_text


if __name__ == "__main__":
    all_config = Config()
    patch_sklearn()

    train_data = data_process(all_config.train_path)

    model = SentenceTransformer(all_config.bert_path)

    knowledge_array, knowledge_text = gen_konwledge(model, train_data, all_config.theta)

    knowledge_text = pd.DataFrame(data=knowledge_text)
    knowledge_text.to_csv("database.csv", sep="\t", index=False, encoding='utf-8')
    knowledge_array = np.array(knowledge_array)
    np.save("database.npy", knowledge_array)
    
    print(knowledge_text[:5])

    


