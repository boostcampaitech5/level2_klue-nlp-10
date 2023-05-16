import torch
import pickle
import pandas as pd
import numpy as np


class FocalLossWithLabelSmoothing(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.0):
        super(FocalLossWithLabelSmoothing,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        
    def forward(self, outputs, targets):
        if isinstance(self.alpha, torch.Tensor):
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none', 
                                                        label_smoothing=self.smoothing, weight=self.alpha.to(outputs.device))
        else:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none', 
                                                    label_smoothing=self.smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt)**self.gamma) * ce_loss
        focal_loss = focal_loss.mean()
        return focal_loss     

class LDAMLoss(torch.nn.Module):
    def __init__(self, max_m=0.5, weight=None, s=30, train_path='../dataset/train/train.csv'): #,cls_num_list=cls_num_list)
        super(LDAMLoss, self).__init__()

        df = pd.read_csv(train_path)
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

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :].to(target.device), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return torch.nn.functional.cross_entropy(self.s * output, target, weight=self.weight)