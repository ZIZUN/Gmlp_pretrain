# Gmlp_pretrain
 Pretraining code to train Gmlp Language model 

codertimo/BERT-pytorch 코드베이스로 작성하였습니다.

## Gmlp pretrain

### 1. Conda 환경설정 및 requirements를 설치해주세요.

```
conda create -n gmlp -y python=3.7 && conda activate gmlp
pip install -r requirements.txt
```

### 2. Pretrain

Gmlp
```console
bash example.sh train amlp [bsz]
```
Gmlp + Tiny attention model
```console
bash example.sh train amlp [bsz]
```

DDP를 이용하여 train 가능합니다.

```console
bash example.sh train amlp [bsz] ddp [n_gpu]
```
### 2. Text Classification Finetuning (NSMC)

Gmlp
```console
bash example.sh finetune gmlp [bsz]
```
Gmlp + Tiny attention model
```console
bash example.sh finetune amlp [bsz]
```


  * Roberta의 fullsentence MLM으로 학습할 수 있도록 설정하였습니다.
  * 토크나이저는 koelectra-v3을 사용했습니다.
  * 학습은 data/train 디렉토리에 있는 json파일(인덱싱 된 문장들)들로 진행됩니다. 
  * 모두의 말뭉치 뉴스 데이터를 data/news 디렉토리에 넣고  (1)processing_news.py 를 진행하시면 처리된 json파일을 얻을 수 있습니다.

## Pretrain Loss graph (900,000 step)

 ![graph](./loss_graph.PNG)

  * batchszie 56으로 진행. bsz를 더 높이면 더 안정적으로 학습하여 더 좋은 성능 기대 됩니다.(accumulation step도 사용가능할듯 시간많이들겠지만;)

## 성능

### 감성분석(NSMC Dataset)
|                     | Accuracy (%) |
| ----------------- | ------------ |
| LSTM            | 79.79    |
| BERT(형태소-태그) | 86.57      |
| BERT(Multilingual)  | 87.43        |
| gmlp + tiny_att       | **87.70**        |
| RoBERTa       | 89.88        |

### 개체명인식(Naver NER Dataset)
|                                                                  | Slot F1 (%) |
| ---------------------------------------------------------------- | ----------- |
| CNN-BiLSTM-CRF                                                | 74.57       |
| DistilKoBERT                                                     | 84.13       |
| Bert-Multilingual                                                | 84.20       |
| gmlp + tiny_att(ours)                                          | **85.82**       |
| KoBERT                                                           | 86.11       |
| RoBERTa(ours)                                                   | 87.58       |



## 참조

  * codertimo/[BERT-pytorch][1]
  * lucidrains/[g-mlp-pytorch][2]
  * labmlai/[annotated_deep_learning_paper_implementations][3]
  * monologg/[KoELECTRA][4]
  * monologg/[KoBERT-NER][5]

[1]:https://github.com/codertimo/BERT-pytorch
[2]:https://github.com/lucidrains/g-mlp-pytorch
[3]:https://github.com/labmlai/annotated_deep_learning_paper_implementations
[4]:https://github.com/monologg/KoELECTRA
[5]:https://github.com/monologg/KoBERT-NER