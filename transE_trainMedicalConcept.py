# -*- coding: utf-8 -*-
"""
 @Time    : 2018/8/7 0007 下午 2:28
 @Author  : Shanshan Wang
 @Version : Python3.5
 使用transE方法训练医疗数据，得到每个医疗概念的向量表示
"""
import random
import numpy as np
import tensorflow as tf
import csv

class TransE(object):
    def __init__(self,entity2id,relation2id,tuples,margin=1,dim=50,batch_size=4,epochs=1000):
        self.entity2id=entity2id
        self.relation2id=relation2id
        self.margin=margin
        self.dim=dim
        self.tuples=tuples
        self.batch_size=batch_size
        self.epochs=epochs

    def trainDataGenerate(self):
        # 从训练数据集中进行取样
        samples=random.sample(self.tuples,self.batch_size)
        Tbatch=[]
        # 针对每个sample 生成它的负例元组
        for sample in samples:
            tuplewithCorruptedTuple=[sample,self.corruptedTupleGeneration(sample)]
            Tbatch.append(tuplewithCorruptedTuple)
        return Tbatch
    def corruptedTupleGeneration(self,sample):
        # 根据随机数随机选择替换头实体还是尾实体
        i = np.random.uniform(-1, 1)
        if i>0:
            while True:
                headEntity=random.choice(list(self.entity2id.values()))
                if headEntity!=sample[0] and headEntity!=sample[2]:
                    break
            corruptedTuple=[headEntity,sample[1],sample[2]]
        else:
            while True:
                tailEntity=random.choice(list(self.entity2id.values()))
                if tailEntity!=sample[2] and tailEntity!=sample[0]:
                    break
            corruptedTuple=[sample[0],sample[1],tailEntity]
        return corruptedTuple

    def model_train(self,Tbatch,train_batch_size=300):
        pos_triple_batch=tf.placeholder(tf.int32,[None,3])
        neg_triple_batch=tf.placeholder(tf.int32,[None,3])

        # embedding table
        print('entity2id:',self.entity2id)
        print('relation2id:',self.relation2id)
        embedding_table=tf.Variable(tf.random_uniform((len(self.entity2id)+len(relation2id),self.dim),minval=-6/np.sqrt(self.dim),maxval=6/np.sqrt(self.dim),dtype=tf.float32))
        pos_embedding_tuples=tf.nn.embedding_lookup(embedding_table,pos_triple_batch)


        pos_embedding_tuples_norm=self.vector_norm(pos_embedding_tuples)
        neg_embedding_tuples=tf.nn.embedding_lookup(embedding_table,neg_triple_batch)
        neg_embedding_tuples_norm=self.vector_norm(neg_embedding_tuples)

        #求不同类型元组的势能函数
        pos_loss=self.l1_energy(pos_embedding_tuples_norm)
        neg_loss=self.l1_energy(neg_embedding_tuples_norm)
        loss=tf.reduce_sum(tf.nn.relu(self.margin+pos_loss-neg_loss))

        optimizer=tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

        # run the graph
        sess=tf.Session()
        sess.run(tf.initialize_all_variables())
        Tbatch=np.array(Tbatch)
        for step in range(self.epochs):
            loss_epoch=0
            # 取样，先洗牌
            shuffle_idx=np.random.permutation(np.arange(len(Tbatch)))
            tuples=Tbatch[shuffle_idx]
            num_batches=len(Tbatch)//train_batch_size
            for i in range(num_batches):
                min_ix=i*self.batch_size
                max_ix=min((i+1)*self.batch_size,len(tuples))
                train_batch=tuples[min_ix:max_ix]
                loss_val,_=sess.run([loss,optimizer],feed_dict={pos_triple_batch:train_batch[:,0],neg_triple_batch:train_batch[:,1]})
                loss_epoch+=loss_val
            print('step: %d, loss:%f' % (step, loss_epoch))
        print('train process is finished.')
        embedding_table = sess.run(embedding_table,
                                   feed_dict={pos_triple_batch: Tbatch[:, 0], neg_triple_batch: Tbatch[:, 1]})
        pos_embeddings=sess.run(pos_embedding_tuples_norm,feed_dict={pos_triple_batch: Tbatch[:, 0], neg_triple_batch: Tbatch[:,1]})
        print('pos_embeddings:',pos_embeddings)
        return embedding_table,pos_embeddings,Tbatch[:,0]

    def l1_energy(self,batch):
        return tf.reduce_sum(tf.abs(batch[:, 0, :] + batch[:, 1, :] - batch[:, 2, :]), 1)
    def vector_norm(self,neg_embedding_tuples):
        return tf.nn.l2_normalize(neg_embedding_tuples,dim=2)

    def writeVector(self,filename,pos_embeddings, tuples):
        id2entity={id:val for val,id in self.entity2id.items()}
        id2relation={id:val for val,id in self.relation2id.items()}
        with open(filename, 'w',encoding='utf-8') as f:
            word2vector=dict()
            for i in range(len(tuples)):
                # 注意tuples[i]是个列表
                entity1=id2entity.get(tuples[i][0])
                if entity1 not in word2vector:
                    word2vector[entity1]=pos_embeddings[i][0]
                relation=id2relation.get(tuples[i][1])
                if relation not in word2vector:
                    word2vector[relation]=pos_embeddings[i][1]
                entity2=id2entity.get(tuples[i][2])
                if entity2 not in word2vector:
                    word2vector[entity2]=pos_embeddings[i][2]

            print(word2vector)
            # 将每个单词以及对应的向量写入文件中
            for word,vector in word2vector.items():
                f.write(word+'\t')
                for item in vector:
                    f.write(str(item)+'\t')
                f.write('\n')

def processData(file):
    entities=[]
    relations=[]
    tuples=[]
    with open(file,encoding='utf-8') as f:
        reader=csv.reader(f)
        for line in reader:
            entities.append(line[0])
            entities.append(line[2])
            relations.append(line[1])
            tuples.append(line[:-1])

    # 对实体以及关系分别进行id化
    entity2id=func_name2id(entities)
    relation2id=func_name2id(relations)
    tuple_ids=[]
    for row in tuples:
        tuple_ids.append([entity2id.get(row[0]),relation2id.get(row[1]),entity2id.get(row[2])])
    write_text('../data/entity2id.txt', entity2id)
    write_text('../data/relation2id.txt', relation2id)
    write_text('../data/tuple_ids.txt', tuple_ids)
    return entity2id,relation2id,tuple_ids

def write_text(file,name2id):
    with open(file,'w',encoding='utf-8') as f:
        if type(name2id)==dict:
            for key,val in name2id.items():
                f.write(key+'\t')
                f.write(str(val)+'\n')
        else:
            for line in name2id:
                for item in line:
                    f.write(str(item)+'\t')
                f.write('\n')
def func_name2id(list_):
    set_=set(list_)
    name2id={name:id for id,name in enumerate(set_)}
    return name2id

if __name__ == '__main__':
    entity2id, relation2id,tuples=processData('../data/relationship.csv')
    print(len(tuples))
    transE=TransE(entity2id,relation2id,tuples,batch_size=len(tuples),epochs=2000)
    Tbatch=transE.trainDataGenerate()
    embedding_table, pos_embeddings, tuples=transE.model_train(Tbatch)
    transE.writeVector('../data/vector.txt',pos_embeddings, tuples)






