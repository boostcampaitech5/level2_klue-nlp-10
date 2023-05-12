import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
import transformers

# 기존 load_data.py에서 load_data_marker.py로 변경 (marker를 추가하기 위함)
# marker가 추가된 데이터를 default로 사용하기로 합시다. 
from load_data_marker import *

import wandb
import argparse
import json

# Train, validation splitting을 위해 추가
from sklearn.model_selection import train_test_split

# pytorch_lightning을 사용하기 위해 추가
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import get_linear_schedule_with_warmup


# To disable warning in the FastTokenizer..
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
      """KLUE-RE AUPRC (with no_relation)"""
      labels = np.eye(30)[labels]

      score = np.zeros((30,))
      for c in range(30):
          targets_c = labels.take([c], axis=1).ravel()
          preds_c = probs.take([c], axis=1).ravel()
          precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
          score[c] = sklearn.metrics.auc(recall, precision)
      return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }

def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
    for v in label:
      num_label.append(dict_label_to_num[v])
    
    return num_label


# Inference에서만 호출되는 함수입니다. (학습시에는 호출되지 않습니다.)
def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, train_path, test_path, predict_path, shuffle=True):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.train_path = train_path
        self.test_path = test_path
        self.predict_path = predict_path
        
        self.train_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['PER', 'LOC', 'POH', 'DAT', 'NOH', 'ORG'])
        
    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = load_data(self.train_path)
            train_label = np.array(label_to_num(train_data['label'].values))
            
            # train, validation split
            # test size를 바꾸거나, random_state를 바꾸면 validation이 바뀝니다.
            train_idx, val_idx = train_test_split(np.arange(len(train_data)), test_size=0.1, random_state=42, stratify=train_label)
            train_data, val_data = train_data.iloc[train_idx], train_data.iloc[val_idx]
            train_label, val_label = train_label[train_idx], train_label[val_idx]
            
            tokenized_train = tokenized_dataset(train_data, self.tokenizer)
            tokenized_val = tokenized_dataset(val_data, self.tokenizer)
            
            self.train_dataset = RE_Dataset(tokenized_train, train_label)
            self.val_dataset = RE_Dataset(tokenized_val, val_label)
            
        else:
            _, tokenized_test, test_label = load_test_dataset(self.predict_path, self.tokenizer)
            # test_data = load_data(self.test_path)
            # test_label = np.array(test_data['label'].values)
            
            self.test_dataset = RE_Dataset(tokenized_test, test_label)
            
            self.predict_dataset = RE_Dataset(tokenized_test, test_label)
    
    # drop_last는 전부 False로 하여, 버려지는 데이터가 없도록 했습니다.
    # test를 위한 데이터셋이 따로 없으므로, 실제로 test_data_loader는 사용되지 않습니다. (위의 setup 참고)
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4, drop_last=False)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4, drop_last=False)
    
    def test_data_loader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, drop_last=False)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=4, drop_last=False)
    
class Model(pl.LightningModule):
    def __init__(self, model_name, lr, vocab_size, use_LSTM=False, loss=torch.nn.CrossEntropyLoss, warmup_steps=500):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.lr = lr
        self.use_LSTM = use_LSTM
        self.loss_func = loss
        self.validation_step_logits = []
        self.validation_step_labels = []
        
        self.warmup_steps = warmup_steps
        
        # 모델 정의
        # LSTM 사용 시
        if self.use_LSTM:
            self.plm = transformers.AutoModel.from_pretrained(model_name)
            self.plm.resize_token_embeddings(vocab_size)
            input_size = self.plm.config.hidden_size
            self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=2, batch_first=True, bidirectional=True)
            self.classifier = torch.nn.Linear(input_size*2, 30)
        # LSTM 미사용 시 : Usual huggingface SequenceClassification
        else:
            # Transformer 호출
            self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=30)
            self.plm.resize_token_embeddings(vocab_size)
        
    def forward(self, x):
        # x : dict. keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if self.use_LSTM:
            # plm output -> LSTM -> classifier
            # plm output : (sequence_output, pooled_output)
            #   Shape = ((batch_size, sequence_length, hidden_size), (batch_size, hidden_size))
            #   sequence_output : 각 input token의 hidden state
            #   pooled_output : [CLS] token의 hidden state
            # LSTM output : (output, (hidden, cell))
            #   We only use hidden.
            #   Shape of hidden = (num_layers * num_directions, batch_size, hidden_size)
            # classifier output : (batch_size, 30)
            x = self.plm(input_ids=x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])[0]  # All hidden states
            out, (hidden, cell) = self.lstm(x)
            x = torch.cat([hidden[-1], hidden[-2]], dim=-1) # Concating last hidden state of forward and backward at last layer
            x = self.classifier(x)
            return x
        else:
            # plm output -> classifier
            # plm output에서 'logits' 만 사용. 
            #  Shape = (batch_size, num_labels)
            x = self.plm(input_ids=x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])['logits']
            return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log("val_loss", loss)
        
        # F1 score, AUPRC score 계산을 위해 logits, labels 저장
        self.validation_step_logits.append(logits)
        self.validation_step_labels.append(y)
        
        return loss
    
    def on_validation_epoch_end(self):
        logits = torch.cat(self.validation_step_logits)
        preds = torch.argmax(logits, axis=-1).cpu().detach().numpy()
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().detach().numpy()
        
        y = torch.cat(self.validation_step_labels).cpu().detach().numpy()
        f1_score = klue_re_micro_f1(preds, y)
        auprc_score = klue_re_auprc(probs, y)
        acc = accuracy_score(y, preds)
        self.log("val_f1", f1_score)
        self.log("val_auprc", auprc_score)
        self.log("val_acc", acc)
        
        self.validation_step_logits.clear()
        self.validation_step_labels.clear()
    
    # def on_validation_epoch_end(self, outputs):
    #     loss, logits, y = outputs[0]
    #     avg_loss = torch.stack(loss).mean()
    #     self.log("val_loss", avg_loss)
        
    #     preds = np.argmax(logits, axis=-1).cpu().detach().numpy()
    #     probs = torch.nn.functional.softmax(logits, dim=-1).cpu().detach().numpy()
    #     probs = torch.cat(probs)
        
    #     y = torch.cat(y).cpu().detach().numpy()
    #     f1_score = klue_re_micro_f1(preds, y)
    #     auprc_score = klue_re_auprc(logits, y)
    #     acc = accuracy_score(y, preds)
    #     self.log("val_f1", f1_score)
    #     self.log("val_auprc", auprc_score)
    #     self.log("val_acc", acc)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        preds = torch.argmax(logits, axis=-1).cpu().detach().numpy()
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().detach().numpy()
        
        # 추후에 output 결과를 보고 수정할 예정
        return preds, probs
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, 
        #                                                 pct_start = 1.0 - 500.0 / self.trainer.estimated_stepping_batches,
        #                                                 total_steps = self.trainer.estimated_stepping_batches, anneal_strategy='linear')
        return [optimizer], [scheduler]
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=float, default=500)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--use_LSTM', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    
    parser.add_argument('--data_path', type=str, default='../dataset/')
    
    parser.add_argument('--wandb_username', default='username', type=str)
    parser.add_argument('--wandb_project_name', default='project_name', type=str)
    parser.add_argument('--wandb_entity', default='entity', type=str)
    parser.add_argument('--config', default=False, type=str, help='config file path')
    parser.add_argument('--wandb_key', default='key')
    
    # config.json 파일에서 args를 불러오기 때문에, 위의 옵션들은 config.json에서 조정하는 것을 권장함.
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            parser.set_defaults(**json.load(f))
        args = parser.parse_args()
    
    train_path = args.data_path + 'train/train.csv'
    test_path = args.data_path + 'test/test_data.csv'
    predict_path = args.data_path + 'test/test_data.csv'
    
    # Callbacks in pytorch lightning trainer
    cp_callback = ModelCheckpoint(monitor='val_loss',    # val_loss를 모니터링하여 저장함.
                                    verbose=False,            # 중간 출력문을 출력할지 여부. False 시, 없음.
                                    save_last=True,           # 직전 ckpt는 last.ckpt 로 따로 저장됨
                                    save_top_k=5,             # k개의 최고 성능 체크 포인트를 저장.
                                    save_weights_only=True,   # Weight만 저장할지, 학습 관련 정보도 저장할지 여부.
                                    mode='min',                # 'min' : monitor metric이 증가하면 저장.
                                    dirpath='./checkpoints',    # ckpt file을 저장할 경로
                                    filename=f'{args.model_name.replace("/","-")}-' + '-{step}-{val_loss:.3f}', # ckpt file name
                                    )

    early_stop_callback = EarlyStopping(monitor='val_loss', 
                                        patience=7,         # x 번 이상 validation 성능이 안좋아지면 early stop
                                        mode='min'          # 'max' : monitor metric은 최대화되어야 함.
                                        )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callback_list = [cp_callback, early_stop_callback, lr_monitor]
    
    # Wandb logger
    wandb.login(key=args.wandb_key)
    wandb_logger = WandbLogger(
        log_model=False, 
        name=f'{args.model_name.replace("/","-")}_{args.learning_rate:.2e}_{args.batch_size}_{args.max_epoch}_{args.use_LSTM}', # wandb run name
        project=args.wandb_project_name,    # wandb project name
        entity=args.wandb_entity    # wandb entity name (owner name of project)
    )
    
    # dataloader, vocab_size to define model 
    dataloader = Dataloader(args.model_name, args.batch_size, train_path, test_path, predict_path, shuffle=args.shuffle)
    vocab_size = len(dataloader.tokenizer)    
    
    # Loss function은 여기서 정의할 것. Custom이 필요할 경우, 따로 정의해서 여기서 불러오면 됨.
    loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # model and trainer
    model = Model(args.model_name, args.learning_rate, vocab_size, use_LSTM=args.use_LSTM,loss=loss, warmup_steps=args.warmup_steps)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, callbacks=callback_list, 
                         log_every_n_steps=1, logger=wandb_logger, val_check_interval=0.25)
    
    # Training
    trainer.fit(model=model, datamodule=dataloader)
    # Test는 수행하지 않습니다. (Test를 위한 데이터셋이 없기 때문)