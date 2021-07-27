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


+ Roberta의 fullsentence MLM으로 학습할 수 있도록 설정하였습니다.
+ 토크나이저는 koelectra-v3을 사용했습니다.
+ 학습은 data/train 디렉토리에 있는 json파일(인덱싱 된 문장들)들로 진행됩니다. 
+ 모두의 말뭉치 뉴스 데이터를 data/news 디렉토리에 넣고  (1)processing_news.py 를 진행하시면 처리된 json파일을 얻을 수 있습니다.


## 참조

  * codertimo/[BERT-pytorch][1]
  * lucidrains/[g-mlp-pytorch][2]
  * labmlai/[annotated_deep_learning_paper_implementations][3]
  * monologg/[KoELECTRA][4]

[1]:https://github.com/codertimo/BERT-pytorch
[2]:https://github.com/lucidrains/g-mlp-pytorch
[3]:https://github.com/labmlai/annotated_deep_learning_paper_implementations
[4]:https://github.com/monologg/KoELECTRA