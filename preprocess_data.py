import os
from tqdm import tqdm
from w3lib.html import remove_tags
import re
import pandas as pd
import random


class Config:
    def __init__(self):
        self.name = 'europarl_data_process'
        self.data_path = os.path.join(os.getcwd(), 'data/europarl')
        self.save_path = os.path.join(os.getcwd(), 'data/')
        self.train_path = os.path.join(self.save_path, "train.csv")
        self.vaild_path = os.path.join(self.save_path, "vaild.csv")
        self.test_path = os.path.join(self.save_path, "test.csv")


def cutted_data(cleaned, MIN_LENGTH=5, MAX_LENGTH=20):
    cutted_lines = list()
    for line in cleaned:
        if "i.e." in line or "e.g." in line or "(" in line or ")" in line or "-" in line or ":" in line or "[" in line or "]" in line or "'" in line or "-" in line:
            line = ""
        if "\""in line:
            line = ""
        if line == "":
            continue
        line = re.sub(r'[^0-9A-Za-z!?,.\s]+', '', line)
        line = re.sub(r"\?\s", r"?\n", line)
        line = re.sub(r"\!\s", r"!\n", line)
        line = re.sub(r"\.\s", r".\n", line)
        lines = re.split(r"\n", line)
        for p in lines:
            q = p
            length = len(q.split())
            if length > MIN_LENGTH and length < MAX_LENGTH:
                cutted_lines.append(p.strip())
    return cutted_lines


def text_file_process(text_path):
    fop = open(text_path, 'r', encoding='utf8')
    raw_data = fop.read()
    sentences = raw_data.strip().split('\n')
    raw_data_input = [remove_tags(data) for data in sentences]
    raw_data_input = cutted_data(raw_data_input)
    fop.close()
    return raw_data_input


if __name__ == '__main__':

    all_config = Config()

    sentences = list()
    print('Preprocess Raw Text')
    pbar = tqdm(os.listdir(all_config.data_path))
    for fn in pbar:
        if not fn.endswith('.txt'): continue
        process_sentences = text_file_process(os.path.join(all_config.data_path, fn))
        sentences += process_sentences

    length = len(sentences)
    print(length)
    random.shuffle(sentences)
    knowledge_text = pd.DataFrame(data=sentences[:int(length*0.8)])
    knowledge_text.to_csv(all_config.train_path, sep="\t", index=False, encoding='utf-8')

    knowledge_text = pd.DataFrame(data=sentences[int(length*0.8):int(length*0.9)])
    knowledge_text.to_csv(all_config.vaild_path, sep="\t", index=False, encoding='utf-8')

    knowledge_text = pd.DataFrame(data=sentences[int(length*0.9):])
    knowledge_text.to_csv(all_config.test_path, sep="\t", index=False, encoding='utf-8')
    