#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yaml
import numpy as np


# In[2]:


def eval_probs(pd: pd.DataFrame):
    pd["ev_probs"] = pd.apply(lambda x: np.array(eval(x["probs"])), axis=1)
    return pd


# In[95]:


def merge_probs(dfs: list):
    res = dfs[0]["ev_probs"].copy()
    for i in range(1, len(dfs)):
        res += dfs[i]["ev_probs"]
    res /= len(dfs)
    return res


# In[134]:


def update_pred_label(pd: pd.DataFrame, config: dict):
    pd["pred_label"] = pd["probs"].apply(lambda x : config["label_list"][x.argmax()])
    return pd


# In[147]:


def save_csv(pd: pd.DataFrame, config: dict):
    pd.to_csv(config["output_path"], index=False)


# In[149]:


def main():
    # load config
    with open('./merge_setting.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load data
    dfs = []
    for csv_path in config["csv_paths"]:
        dfs.append(pd.read_csv(csv_path))
    
    res = dfs[0].copy()
    
    for df in dfs:
        eval_probs(df)
    
    res.probs = merge_probs(dfs)

    update_pred_label(res, config)

    save_csv(res, config)

    print("merged inference results are saved at", config["output_path"])


# In[150]:


if __name__ == "__main__":
    main()

