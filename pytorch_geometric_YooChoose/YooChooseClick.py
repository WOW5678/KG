# -*- coding: utf-8 -*-
"""
 @Time    : 2019/6/10 0010 下午 8:30
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:
"""
import numpy as np
import pandas as pd
import pickle
import csv
import os
import torch
print(torch.__version__) #1.1.0
from torch_geometric.data import Data

np.random.seed(42)
#nrows：read top 100行数据，速度较快，有助于测试
df=pd.read_csv('data/input/yoochoose-data/yoochoose-clicks.dat',nrows=100)
df.columns=['session_id','timestamp','item_id','category']
#print(df.head(5))

buy_df=pd.read_csv('data/input/yoochoose-data/yoochoose-buys.dat',header=None,nrows=100)
buy_df.columns=['session_id','timestamp','item_id','price','quantity']
print(buy_df.head(5))
print(buy_df.nunique())
print(df.nunique())
#
#
print('========================')
#增加一列 valid_session，值为True或者False
df['valid_session']=df.session_id.map(df.groupby('session_id')['item_id'].size()>2)
print('df[valid_session]:',df.head(5))
df=df.loc[df.valid_session].drop('valid_session',axis=1)
print(df.head(5))
#
# Randomly sample a couple of them
sampled_session_id=np.random.choice(df.session_id.unique(),10,replace=False)
df=df.loc[df.session_id.isin(sampled_session_id)] #df.loc执行的是切片操作 切出取样的数据
print(df.nunique())

print(df.isna().sum())

# Average length of session
#计算每个session的平均长度（即每个session中item数目的平均长度）
df.groupby('session_id')['item_id'].size().mean()

from sklearn.preprocessing import LabelEncoder
item_encoder=LabelEncoder() #LabelEncoder将string 编码成id，注意并不是one-hot表示
df['item_id']=item_encoder.fit_transform(df.item_id)
print(df.head(5))
df['label'] = df.session_id.isin(buy_df.session_id)
print(df.head(5))
print(df.drop_duplicates('session_id')['label'].mean())


# Model部分
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self,root=None,transform=None,pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root,transform,pre_transform)
        self.data,self.slices=torch.load(self.processed_paths[0])

        #raw_file_nemes 返回一个list of raw,unprocessed file names, 一般返回空列表就行，在process()函数中指明文件名即可
        #Python内置的@property装饰器就是负责把一个方法变成属性调用的
        @property
        def raw_file_names(self):
            return []

        #Similar to the last function, it also returns a list containing the file names of all the processed data.
        #After process() is called, Usually, the returned list should only have one element, storing the only processed data file name.
        @property
        def processed_file_names(self):
            return ['data/input/yoochoose-data/yoochoose_click_binary_1M_sess.dataset']

        #This function should download the data you are working on to the directory as specified in self.raw_dir.
        #If you don’t need to download data, simply drop in pass
        def download(self):
            pass

        #This is the most important method of Dataset. You need to gather your data into a list of Data objects.
        #Then, call self.collate() to compute the slices that will be used by the DataLoader object.
        def process(self):
            data_list=[]

            #process by session_id
            grouped=df.groupby('session_id')
            for session_id,group in tqdm(grouped):
                sess_item_id=LabelEncoder().fit_transform(group.item_id)
                group = group.reset_index(drop=True)
                group['sess_item_id'] = sess_item_id
                node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                    'sess_item_id').item_id.drop_duplicates().values

                # unsqueeze(arg):表示在第arg维增加一个维度值为1的维度
                node_features = torch.LongTensor(node_features).unsqueeze(1)
                target_nodes = group.sess_item_id.values[1:]
                source_nodes = group.sess_item_id.values[:-1]

                edge_index = torch.tensor([source_nodes,target_nodes], dtype=torch.long)
                x = node_features

                y = torch.FloatTensor([group.label.values[0]])

                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

print(os.getcwd())
dataset=YooChooseBinaryDataset(root='../data')
dataset=dataset.shuffle()
train_dataset=dataset[:80]
val_dataset=dataset[80:90]
test_dataset=dataset[90:]
len(train_dataset),len(val_dataset),len(test_dataset)
