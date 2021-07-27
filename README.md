# Gmlp_pretrain
 Pretraining code to train Gmlp Language model


## Gmlp pretrain

### 1. 환경설정 및 requirements를 설치해주세요.

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

