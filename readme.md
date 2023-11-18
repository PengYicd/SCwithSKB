# Deep Learning-Empowered Semantic Communication Systems With a Shared Knowledge Base  

**Peng Yi, Yang Cao, Xin Kang, and Ying-Chang Liang**

This is the implementation of the paper named [Deep Learning-Empowered Semantic Communication Systems with a Shared Knowledge Base | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/10318078). 

------

### BibTex

```latex
@ARTICLE{10318078,
  author={Peng Yi and
          Yang Cao and
          Xin Kang and
          Y.-C. Liang},
  journal={{IEEE} Trans. Wirel. Commun.}, 
  title={Deep Learning-Empowered Semantic Communication Systems with a Shared Knowledge Base}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TWC.2023.3330744}}

```

------

### Dataset

For more information about the dataset, pleases refer to the paper [Europarl: A Parallel Corpus for Statistical Machine Translation - ACL Anthology](https://aclanthology.org/2005.mtsummit-papers.11/). The dataset can be downloaded from [Europarl Parallel Corpus (statmt.org)](https://www.statmt.org/europarl/)

For the first step, you need to download the dataset, preprocess the data and store the data in ‘./data/train.csv’, ‘./data/valid.csv’ and ‘./data/test.csv’.

------

### Knowledge Base Generation 

Please use “KnowledgeBase_gen.py” to generate the knowledge base. It will output “database.csv” and “database.npy”. The “npy ”file is used to facilitate the training process.

------

### Training 

Please use “train.py” to train the model.
