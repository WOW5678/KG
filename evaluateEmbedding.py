# -*- coding: utf-8 -*-
"""
 @Time    : 2018/8/8 0008 下午 5:24
 @Author  : Shanshan Wang
 @Version : Python3.5
 function : 该类实现对训练好的embedding的效果进行评估
 分为两个指标:(1)filtered mean rank(2)filtered Hits@10
 评估过程如下：
 1.对训练集中每个正确的三元组a随机替换头或者尾（所有的实体都替换一个遍，假设n个实体），则得到n个三元组（包含这个三元组a）
 2.计算n个三元组的能量值
 3.对能量值进行升序排列，并计算正确的三元组的位置K
 4.若前K-1个三元组中有m个正确的三元组，则将k设置为k-m
 5.对所有的正确位置求平均，得到filter mean Rank指标
 6.统计位置小于10的正确三元组所占比例，即为hits@10指标
"""
import  tensorflow as tf
import numpy as np
import numpy.linalg as LA

class EvaluateEmbedding(object):
    def __init__(self,tuples,entity2id,relation2id,word2vector):
        self.tuples=tuples
        self.entity2id=entity2id
        self.id2entity={val:key for key,val in entity2id.items()}
        self.relation2id=relation2id
        self.id2relation={val:key for key,val in relation2id.items()}
        self.word2vector=word2vector
    '''
    获取mean Rank值
    此处我们都是替换的头实体
    '''
    def get_mean_rank(self):
        indexList=[]
        for tuple in self.tuples:
            # 替换头实体
            tuple2energy=self.get_n_tuples(tuple)
            # 对能量值按照从低到高进行排序
            ordered_tuple2energy=sorted(tuple2energy.items(),key=lambda x:x[1],reverse=False)
            print(ordered_tuple2energy)
            # 将元组替换成tuple2energy(注意:现在变成了列表)中的key 并获得真实三元组的索引
            tuple_str=str(tuple[0])+'-'+str(tuple[1])+'-'+str(tuple[2])
            for key,val in ordered_tuple2energy:
                if key==tuple_str:
                    k=ordered_tuple2energy.index((key,val))
                    #若前K-1个元组中有m个真实的元组，则k=k-m
                    k=self.reviseK(k,ordered_tuple2energy)
                    indexList.append(k)
        # 对所有k值求平均
        meanRank=sum(indexList)/len(indexList)
        return indexList,meanRank

    def reviseK(self,k,ordered_tuple2energy):
        tuple_strs=[]
        for tuple in self.tuples:
            tuple_strs.append(str(tuple[0])+'-'+str(tuple[1])+'-'+str(tuple[2]))
        # 记录前K-1个元组中真实元组的个数
        m=0
        for key,val in ordered_tuple2energy[:k]:
            if key in tuple_strs:
                m+=1
        return k-m

    def get_n_tuples(self,tuple):
        tuple2energy={}
        for entity in self.entity2id.values():
            tmp=[entity,tuple[1],tuple[2]]
            tuple_str=str(tmp[0])+'-'+str(tmp[1])+'-'+str(tmp[2])
            #计算生成的元组的能量值
            energy=self.caculate_energy(tmp)
            if tuple_str not in tuple2energy:
                tuple2energy[tuple_str]=energy
        return tuple2energy

    def caculate_energy(self,tuple):
        # 计算每个元组的能量值
        # 首先找见每个id对应的vector
        head_entity=np.array(self.word2vector.get(self.id2entity.get(tuple[0])))
        relation=np.array(self.word2vector.get(self.id2relation.get(tuple[1])))
        tail_entity=np.array(self.word2vector.get(self.id2entity.get(tuple[2])))
        return LA.norm(head_entity+relation-tail_entity,axis=0)
    def get_hit10(self,indexList):
        # 统计索引排在前10位的真实元组所占的比例
        top10=[item for item in indexList if item<=9]
        return len(top10)/len(indexList)

def load_data(file):
    entity2vector={}
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            lineList=line.replace('\n','').strip().split('\t')
            entity=lineList[0]
            vector=[float(item) for item in lineList[1:]]
            if entity not in entity2vector:
                entity2vector[entity]=vector
    return entity2vector

def load_word_id(file):
    entity2vector={}
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            lineList=line.replace('\n','').strip().split('\t')
            entity=lineList[0]
            vector=float(lineList[1])
            if entity not in entity2vector:
                entity2vector[entity]=vector
    return entity2vector

def load_train_ids(file):
    tuples_ids=[]
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            lineList=line.replace('\n','').strip().split('\t')
            tuples_ids.append([float(item) for item in lineList])
    return tuples_ids

if __name__ == '__main__':
    word2vector=load_data('../data/vector.txt')
    entity2id=load_word_id('../data/entity2id.txt')
    relation2id=load_word_id('../data/relation2id.txt')
    tuple_ids=load_train_ids('../data/tuple_ids.txt')
    evaluateEmbedding=EvaluateEmbedding(tuple_ids,entity2id,relation2id,word2vector)
    indexList,meanRank=evaluateEmbedding.get_mean_rank()
    print('mean Rank of embedding is:',meanRank)
    hit10=evaluateEmbedding.get_hit10(indexList)
    print('Hit@10 of embedding is:',hit10)