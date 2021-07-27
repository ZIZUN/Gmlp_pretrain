import argparse

from torch.utils.data import DataLoader
from gmlp.model import BERT, gMLP
from gmlp.trainer import nsmc_Trainer
from gmlp.dataset import LoadDataset_nsmc

import torch

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True, type=str, help="model for training (gmlp or bert)")

parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset")
parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
parser.add_argument("-v", "--vocab_size", required=True, type=int, help="built vocab model path with bert-vocab")
parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

parser.add_argument("--ddp", type=bool, default=False, help="for distrbuted data parrerel")
parser.add_argument("--local_rank", type=int, help="for distrbuted data parrerel")
parser.add_argument("--attn_dim", type=int, help="for gmlp attn_dim")


parser.add_argument("-hs", "--hidden", type=int, default=512, help="hidden size of transformer model")
parser.add_argument("-l", "--layers", type=int, default=48, help="number of layers")
parser.add_argument("-a", "--attn_heads", type=int, default=12, help="number of attention heads")
parser.add_argument( "--pretrain_seq_len", required=True, type=int, default=512, help="maximum sequence len when pretraining")
parser.add_argument("--input_seq_len", required=True, type=int, default=512, help="maximum sequence input len")
parser.add_argument("--causal", required=True, type=bool, default=False, help="when finetuning")

parser.add_argument("-b", "--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

args = parser.parse_args()

if args.ddp:
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)


print("Loading Train Dataset", args.train_dataset)
train_dataset = LoadDataset_nsmc(args.train_dataset, seq_len=args.input_seq_len, vocab_size=args.vocab_size)

print("Loading Test Dataset", args.test_dataset)
test_dataset = LoadDataset_nsmc(args.test_dataset, seq_len=args.input_seq_len, vocab_size=args.vocab_size) \
    if args.test_dataset is not None else None

if args.ddp:
    print("Creating Dataloader")
    train_sampler = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
else:
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

if args.ddp:
    print("Creating Dataloader")
    test_sampler = DistributedSampler(test_dataset)
    test_data_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None
else:
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

if args.model == 'gmlp' or args.model == 'amlp':
    print("Building gMLP model")
    model = gMLP(args.vocab_size, hidden=args.hidden, n_layers=args.layers,
                 seq_len=args.pretrain_seq_len, attn_dim = args.attn_dim, causal= args.causal)
elif args.model == 'bert':
    print("Building BERT model")
    model = BERT(vocab_size=args.vocab_size, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

print("Creating Trainer")
trainer = nsmc_Trainer(model, vocab_size=args.vocab_size, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                      lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                      with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                  distributed = args.ddp, local_rank = args.local_rank)

print("Training Start")
for epoch in range(args.epochs):
    if args.ddp:
        train_sampler.set_epoch(epoch)
        trainer.train(epoch)

        if args.local_rank == 0:
            trainer.save(epoch, args.output_path)
    else:
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

    if test_data_loader is not None:
        trainer.test(epoch)
