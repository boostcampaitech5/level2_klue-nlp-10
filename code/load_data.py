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
        item['labels'] = torch.tensor(self.labels[idx])
        return item

  def __len__(self):
        return len(self.labels)

def load_data(path):

    data = pd.read_csv(path)

    sub_entity, sub_type = [], []
    obj_entity, obj_type = [], []
    sub_idx, obj_idx = [], []
    sentence = []

    for i, [x, y, z] in enumerate(zip(data['subject_entity'], data['object_entity'], data['sentence'])):
        sub_typ = x[1:-1].split(':')[-1].split('\'')[-2]
        obj_typ = y[1:-1].split(':')[-1].split('\'')[-2]

        for idx_i in range(len(x)):
            if x[idx_i: idx_i + 9] == 'start_idx':
                sub_start = int(x[idx_i+12:].split(',')[0].strip())
            if x[idx_i: idx_i+7] == 'end_idx':
                sub_end = int(x[idx_i+10:].split(',')[0].strip())

            if y[idx_i: idx_i + 9] == 'start_idx':
                obj_start = int(y[idx_i+12:].split(',')[0].strip())
            if y[idx_i: idx_i+7] == 'end_idx':
                obj_end = int(y[idx_i+10:].split(',')[0].strip())

        sub_i = [sub_start, sub_end]
        obj_i = [obj_start, obj_end]

        sub_entity.append(z[sub_i[0]: sub_i[1]+1])
        obj_entity.append(z[obj_i[0]: obj_i[1]+1])
        sub_type.append(sub_typ)
        sub_idx.append(sub_i)
        obj_type.append(obj_typ)
        obj_idx.append(obj_i)

        if sub_i[0] < obj_i[0]:
            z = z[:sub_i[0]] + '@*'+sub_typ+'*' + \
                z[sub_i[0]: sub_i[1]+1] + '@' + z[sub_i[1]+1:]
            z = z[:obj_i[0]+7] + '#^' + obj_typ + '^' + \
                z[obj_i[0]+7: obj_i[1]+8] + '#' + z[obj_i[1]+8:]
        else:
            z = z[:obj_i[0]] + '#^' + obj_typ + '^' + \
                z[obj_i[0]: obj_i[1]+1] + '#' + z[obj_i[1]+1:]
            z = z[:sub_i[0]+7] + '@*'+sub_typ+'*' + \
                z[sub_i[0]+7: sub_i[1]+8] + '@' + z[sub_i[1]+8:]

        sentence.append(z)

    df = pd.DataFrame({'id': data['id'], 'sentence': sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
                      'subject_type': sub_type, 'object_type': obj_type, 'label': data['label'],
                       'subject_idx': sub_idx, 'object_idx': obj_idx})

    return df


""" 이제 데이터 프레임에 있는 sentence를 tokenizer를 통해서 쪼개보자 !"""


def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    #tokens = ['PER', 'LOC', 'POH', 'DAT', 'NOH', 'ORG']
    #tokenizer.add_tokens(tokens)

    concat_entity = []
    for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'], dataset['subject_type'], dataset['object_type']):
        temp = '@*' + t01 + '*' + e01 + '@' + '와' + \
            '#^' + t02 + '^' + e02 + '#' + '의 관계'
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=True
    )

    return tokenized_sentences