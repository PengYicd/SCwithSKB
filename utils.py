import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
import csv
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearnex import patch_sklearn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
        self.smooth = SmoothingFunction()
    
    def compute_blue_score(self, real, predicted):
        sent1 = remove_tags(real).split()
        sent2 = remove_tags(predicted).split()
        score = (sentence_bleu([sent1], sent2, 
                        weights=(self.w1, self.w2, self.w3, self.w4), smoothing_function=self.smooth.method1))
        return score


"""sentence transformer"""        
class SimilarityScore():
    def __init__(self):
        # This is a sentence-transformers model: 
        # It maps sentences & paragraphs to a 768 dimensional dense vector space 
        # and can be used for tasks like clustering or semantic search.
        data_dir = os.path.join(os.getcwd(), 'data/sentence-transformers_all-mpnet-base-v2')
        self.model = SentenceTransformer(data_dir)
        patch_sklearn()

    def compute_similarity_score(self, sen_input, sen_output):
        """"
        sen_input : str
        sen_output : str
        """
        sentence_embeddings = self.model.encode([remove_tags(sen_input), remove_tags(sen_output)])
        sen_similarity = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])
        return sen_similarity 


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):
    src_mask = (src == padding_idx).type(torch.FloatTensor) #[batch, seq_len]
    trg_mask = (trg == padding_idx).type(torch.FloatTensor) #[batch, seq_len]
    return src_mask.to(device), trg_mask.to(device)


def loss_function(x, trg, padding_idx, criterion):
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    loss *= mask
    return loss.mean()


class check_id():
    # used to find the most similar knowledge from the knowledge base 
    def __init__(self) -> None:
        self.database = np.load("database.npy")
        self.datatext = []
        with open('database.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                self.datatext.append(row)
        model_dir = os.path.join(os.getcwd(), 'data/sentence-transformers_all-mpnet-base-v2')
        self.model = SentenceTransformer(model_dir)
        patch_sklearn()

    def find_similar(self, sent):
        sent = self.model.encode(sent)
        sen_similarity = cosine_similarity(sent, self.database)
        index = sen_similarity.argsort().flatten()[-1] 
        sim_sent = self.datatext[index]
        return sim_sent[0]


def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


