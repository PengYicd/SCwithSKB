import os
from sys import flags
import torch
import numpy as np
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt 
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SentenceData, collate_data
from model import SC_SKB, Channels
from utils import check_id, adjust_learning_rate, create_masks, loss_function
import random


class Config:
    def __init__(self):
        self.name = 'Sentence_bert_cos'
        self.bert_path = 'data/sentence-transformers_all-mpnet-base-v2'
        self.vocab_file_len = 30527
        self.data_path = 'data/' 
        self.train_path = os.path.join(self.data_path, 'train.csv')
        self.valid_path = os.path.join(self.data_path, 'valid.csv')
        self.test_path = os.path.join(self.data_path, 'test.csv')
        self.save_path = 'checkpoint'
        self.use_sotore = False
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.epochs = 100
        self.batch_size = 256
        self.learning_rate = 0.1
        self.weight_decay = 0
        self.max_length = 20
        self.min_length = 5

        self.channel = "AWGN" # 'AWGN' 'Rayleigh' 'Rician'


def data_process(data_path, bert_path, flag, mode):
    # For the first training, you need to set 'flag' to 'False', the code will find the knowledge corresponding to the input sentence and save it.
    # To speed up the training, we recommend setting 'flag' to 'True' afterward to directly employ npy cached files.
    data = []
    if flag == True:
        if mode == "train":
            data_src = np.load("database11.npy")
            data_tgt = np.load("database12.npy")
            print("Load train dataset")
        elif mode == "valid" :
            data_src = np.load("database21.npy")
            data_tgt = np.load("database22.npy")
            print("Load valid dataset")


    elif flag == False:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                data.append(row)

        data_src = []
        data_tgt = []
        tokenzier = transformers.AutoTokenizer.from_pretrained(bert_path)
        find_id = check_id()

        with tqdm(total=len(data), desc='find_similar_sentence', leave=True, ncols=100, unit='it', unit_scale=True) as pbar:
            for row in data:
                sim_sent = find_id.find_similar(row)
                src = tokenzier(row, padding='max_length', max_length=20, truncation=True)
                tgt = tokenzier(sim_sent, padding='max_length', max_length=20, truncation=True)
                data_src.append(src.data['input_ids'])
                data_tgt.append(tgt.data['input_ids'])
                pbar.update()

        if mode == "train":
            data_array = np.array(data_src)
            np.save("database11.npy", data_array)
            data_array = np.array(data_tgt)
            np.save("database12.npy", data_array)

        elif mode =="valid":
            data_array = np.array(data_src)
            np.save("database21.npy", data_array)
            data_array = np.array(data_tgt)
            np.save("database22.npy", data_array)

    return data_src, data_tgt


if __name__ == '__main__':
    all_config = Config()

    train_src, train_tgt = data_process(all_config.train_path, all_config.bert_path, all_config.use_sotore, "train")   
    pro_data = SentenceData(train_src,  train_tgt)
    train_dataloader = DataLoader(pro_data, batch_size=all_config.batch_size, collate_fn=collate_data, drop_last=True, shuffle=True)

    vaild_src, vaild_tgt = data_process(all_config.valid_path, all_config.bert_path, all_config.use_sotore, "valid")
    pro_data = SentenceData(vaild_src,  vaild_tgt)
    vaild_dataloader = DataLoader(pro_data, batch_size=all_config.batch_size, collate_fn=collate_data, drop_last=True)
    print(all_config.channel)

    SC_model = SC_SKB(all_config.vocab_file_len)
    SC_model = SC_model.to(all_config.device)

    criterion_semantic = torch.nn.CrossEntropyLoss(reduction = 'none')

    optimizer = torch.optim.Adadelta(SC_model.parameters(), lr=all_config.learning_rate, rho=0.9, eps=1e-6, weight_decay=all_config.weight_decay)
    
    channels = Channels()
    loss_list = list()
    vaild_loss_list = list()

    for epoch in range(all_config.epochs):
        snr = random.randint(0,10)
        SC_model.train()

        run_loss = 0
        i = 0 
        train_pbar = tqdm(train_dataloader)
        for src_o, tgt_o in train_pbar:
            src_o = src_o.to(all_config.device)
            tgt_o = tgt_o.to(all_config.device)
            
            src_mask, tgt_mask = create_masks(src_o, tgt_o, padding_idx=1)
            src_embedding = SC_model.embedding_s(src_o)
            tgt_embedding = SC_model.embedding_k1(tgt_o)
            src_encode = SC_model.semantic_extraction_s(src_embedding, mask=None, src_key_padding_mask=src_mask)
            tgt_encode = SC_model.semantic_extraction_k1(tgt_embedding, mask=None, src_key_padding_mask=tgt_mask)

            sent = SC_model.decode_model_t(src_encode, tgt_encode)

            Tx_sig = SC_model.channel_encode(sent)
            Tx_sig = Tx_sig.reshape((all_config.batch_size, -1))

            if all_config.channel == 'AWGN':
                Rx_sig = channels.AWGN(Tx_sig, snr)
            elif all_config.channel == 'Rayleigh':
                Rx_sig = channels.Rayleigh(Tx_sig, snr)
            elif all_config.channel == 'Rician':
                Rx_sig = channels.Rician(Tx_sig, snr)
            else:
                raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

            Rx_sig = Rx_sig.reshape(all_config.batch_size,  -1, 12)
            recive = SC_model.channel_decode(Rx_sig)
            
            tgt_embedding_r = SC_model.embedding_k2(tgt_o)
            tgt_encode_r = SC_model.semantic_extraction_k2(tgt_embedding_r, mask=None, src_key_padding_mask=tgt_mask)

            decode = SC_model.decode_model_r(recive, tgt_encode_r)
            decode = SC_model.semantic_recover(decode, mask=None, src_key_padding_mask=src_mask)
            decode = SC_model.dense(decode)
            decode_len = decode.size(-1)

            loss = loss_function(decode.contiguous().view(-1, decode_len), \
                src_o.long().contiguous().view(-1), padding_idx=1, criterion=criterion_semantic)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pbar.set_description('Epoch: {}; SNR: {}; Type: Train; Loss: {:.5f}'.format(epoch + 1, snr, loss.item()))
            run_loss += loss.item()
            i += 1
       
        torch.save(SC_model, os.path.join(all_config.save_path, all_config.channel, 'SC_Epoch{}.pth'.format(epoch+1)))

        loss_list.append(run_loss/i) 
        print('Epoch: {};  Type: Train; Loss: {:.5f}'.format(epoch + 1, run_loss/i))
        adjust_learning_rate(optimizer, epoch, all_config.learning_rate)


        # validation
        if epoch % 10 == 9:
            snr = 0
            run_loss = 0
            SC_model.eval()

            vaild_pbar = tqdm(vaild_dataloader)
            j = 0
            for src_o, tgt_o in vaild_pbar:
                src_o = src_o.to(all_config.device)
                tgt_o = tgt_o.to(all_config.device)

                src_mask, tgt_mask = create_masks(src_o, tgt_o, padding_idx=1)
                src_embedding = SC_model.embedding_s(src_o)
                tgt_embedding = SC_model.embedding_k1(tgt_o)
                src_encode = SC_model.semantic_extraction_s(src_embedding, mask=None, src_key_padding_mask=src_mask)
                tgt_encode = SC_model.semantic_extraction_k1(tgt_embedding, mask=None, src_key_padding_mask=tgt_mask)

                sent = SC_model.decode_model_t(src_encode, tgt_encode)

                Tx_sig = SC_model.channel_encode(sent)
                Tx_sig = Tx_sig.reshape((all_config.batch_size, -1))

                if all_config.channel == 'AWGN':
                    Rx_sig = channels.AWGN(Tx_sig, snr)
                elif all_config.channel == 'Rayleigh':
                    Rx_sig = channels.Rayleigh(Tx_sig, snr)
                elif all_config.channel == 'Rician':
                    Rx_sig = channels.Rician(Tx_sig, snr)
                else:
                    raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

                Rx_sig = Rx_sig.reshape(all_config.batch_size,  -1, 12)
                recive = SC_model.channel_decode(Rx_sig)
                
                tgt_embedding_r = SC_model.embedding_k2(tgt_o)
                tgt_encode_r = SC_model.semantic_extraction_k2(tgt_embedding_r, mask=None, src_key_padding_mask=tgt_mask)

                decode = SC_model.decode_model_r(recive, tgt_encode_r)
                decode = SC_model.semantic_recover(decode, mask=None, src_key_padding_mask=src_mask)
                decode = SC_model.dense(decode)
                decode_len = decode.size(-1)

                vaild_loss = loss_function(decode.contiguous().view(-1, decode_len), \
                    src_o.long().contiguous().view(-1), padding_idx=1, criterion=criterion_semantic)

                vaild_pbar.set_description('Epoch: {}; SNR: {}  Type: Vaild; Loss: {:.5f}'.format(epoch + 1, snr, vaild_loss.item()))
                run_loss += vaild_loss.item()
                j += 1
            vaild_loss_list.append(run_loss/j)
            print('Epoch: {};  Type: Vaild; Loss: {:.5f}'.format(epoch + 1, run_loss/j))
    
    loss_list_np = np.array(loss_list)
    vaild_loss_list_np = np.array(vaild_loss_list)
    np.save("train_loss.npy", loss_list_np)
    np.save("vaild_loss.npy", vaild_loss_list_np)

    plt.figure(1)
    plt.plot(loss_list)
    plt.title("train")
    plt.xlabel("epoch")
    plt.ylabel("CE Loss")
    plt.savefig("fig1.png")
    # plt.show()

    plt.figure(2)
    plt.plot(vaild_loss_list)
    plt.title("vaild")
    plt.xlabel("epoch")
    plt.ylabel("CE Loss")
    plt.savefig("fig2.png")
    # plt.show()

    pass
 