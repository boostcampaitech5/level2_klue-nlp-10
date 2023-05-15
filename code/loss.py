import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import numpy as np
from load_data import *

class FocalLossTrainer(Trainer):         
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Initialize FocalLoss
        focal_loss_fct = FocalLoss(gamma=2.0)

        # Compute FocalLoss
        loss = focal_loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

class LDAMLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Define cls_num_list
        df = pd.read_csv("../dataset/train/train.csv")
        val_cnt = df["label"].value_counts()
        val_cnt_dict = val_cnt.to_dict()
        #print(val_cnt_dict)

        with open('dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pickle.load(f)

        new_dict = {}
        for key in val_cnt_dict.keys():
            new_dict[ dict_label_to_num[key] ] = val_cnt_dict[key]
        #print(new_dict)

        cls_num_list = []
        for key in range(30):
            cls_num_list.append(new_dict[key])
        #print(cls_num_list)

        # Initialize LDAMLoss
        ldaml_loss_fct = LDAMLoss(cls_num_list, max_m=0.5, s=30)

        # Compute LDAMLoss
        loss = ldaml_loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss



def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)




class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

