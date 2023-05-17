import pickle as pickle
import os
import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    label = torch.tensor(self.labels[idx])
    # shorter code for the above line:
    return item, label

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  
  subject_entity = []
  object_entity = []
  sub_type = []
  obj_type = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    # 1. Dict로 변환
    i = eval(i)
    j = eval(j)
    subject_entity.append(i['word'])
    object_entity.append(j['word'])
    sub_type.append(i['type'])
    obj_type.append(j['type'])
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity, 
                              'sub_type':sub_type, 'obj_type':obj_type, 'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  sent_list = []
  for e01, e02, sent in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    # wrap sub word with @, obj word with #.
    sent = sent.replace(e01, f'@ {e01} @')
    sent = sent.replace(e02, f'# {e02} #')
    sent_list.append(sent)
  tokenized_sentences = tokenizer(
      sent_list,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences
