import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..model import BERT, Encoder
from .optim_schedule import ScheduledOptim

import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F

class nsmc_Trainer:
    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=1000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, distributed = False, local_rank = 0):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert

        self.local_rank = local_rank
        self.distributed = distributed


        self.avgloss = 0

        self.now_iteration = 0

        print("Total Parameters:", sum([p.nelement() for p in self.bert.parameters()]))

        # DDP
        if distributed:
            self.model = Encoder(self.bert, vocab_size)
            self.model.cuda()
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            self.device = torch.device(
                f"cuda:{local_rank}"  # if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
            )
        # nn.DataParallel if CUDA can detect more than 1 GPU
        elif with_cuda and torch.cuda.device_count() > 1:
            self.model = Encoder(self.bert, vocab_size).to(self.device)
            print("Using %d GPUS for your model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        else: # if gpu = 1 or not gpu
            self.model = Encoder(bert, vocab_size).to(self.device)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.CrossEntropyLoss()

        self.log_freq = log_freq

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)



    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        n_correct = 0
        data_len = 0

        for i, data in data_iter:
            if train:
                self.now_iteration += 1

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            encoder_output = self.model.forward(data["bert_input"])

            loss = self.criterion(encoder_output, data["bert_label"])

            predict = F.softmax(encoder_output, dim=1).argmax(dim=1)

            correct = (predict == data['bert_label'].long()).sum().item()

            acc = correct / len(data["bert_label"]) * 100

            if not train:
                n_correct += correct
                data_len += len(data["bert_label"])

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()

                #torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)        if you want gradient clipping

                self.optim_schedule.step_and_update_lr()

                avg_loss += loss.item()
                self.avgloss += loss.item()

            if self.distributed == False:
                post_fix = {
                    "epoch": epoch,
                    "iter": self.now_iteration,  # i,
                    "avg_loss": self.avgloss / self.now_iteration,
                    "loss": loss.item(),
                    "train_acc": acc
                }

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

            elif self.distributed == True and self.local_rank == 0 and train:

                post_fix = {
                    "epoch": epoch,
                    "iter": self.now_iteration,
                    "avg_loss": self.avgloss / (self.now_iteration),
                    "loss": loss.item(),
                    "train_acc" : acc
                }

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

        if not train:
            eval_acc = n_correct / data_len * 100
            print("eval acuracy = " + str(eval_acc))
            print(n_correct)
            print(data_len)

        if train:
            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
    def save(self, epoch, file_path="output/fintuned.model"):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch + '.fintune'
        torch.save(self.bert.cpu().state_dict(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
