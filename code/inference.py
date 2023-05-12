import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
import transformers
from load_data_marker import *
import wandb
import argparse
from utils import *
import json
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import get_linear_schedule_with_warmup

from train import Model, Dataloader

def num_to_label(label):
    """
        숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
    
    return origin_label

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
    
    parser.add_argument('--data_path', type=str, default='../dataset/')
    
    parser.add_argument('--wandb_username', default='username', type=str)
    parser.add_argument('--wandb_project_name', default='project_name', type=str)
    parser.add_argument('--wandb_entity', default='entity', type=str)
    parser.add_argument('--config', default=False, type=str, help='config file path')
    parser.add_argument('--wandb_key', default='key')
    parser.add_argument('--checkpoint_file', default='./checkpoints/last.ckpt')
    
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            parser.set_defaults(**json.load(f))
        args = parser.parse_args()
    
    train_path = args.data_path + 'train/train.csv'
    test_path = args.data_path + 'test/test_data.csv'
    predict_path = args.data_path + 'test/test_data.csv'
    
    dataloader = Dataloader(args.model_name, args.batch_size, train_path, test_path, predict_path, shuffle=args.shuffle)
    vocab_size = len(dataloader.tokenizer)    
    loss = torch.nn.CrossEntropyLoss()
    
    checkpoint_file = args.checkpoint_file
    
    model = Model.load_from_checkpoint(args.checkpoint_file)
    
    trainer = pl.Trainer(accelerator='gpu')
    
    predictions = trainer.predict(model=model, datamodule=dataloader)
    
    pred_list = []
    prob_list = []
    
    for pred, prob in predictions:
        pred_list.append(pred)
        prob_list.append(prob)
        
    preds = np.concatenate(pred_list).tolist()
    probs = np.concatenate(prob_list, axis=0).tolist()
    
    preds = num_to_label(preds)
    
    temp = pd.read_csv(predict_path)
    temp = temp['id']
    output = pd.DataFrame({'id':temp,'pred_label':preds, 'probs':probs})
    output.to_csv('./prediction/submission.csv', index=False)
    print('---- Finish! ----')
    
    
    
    
    