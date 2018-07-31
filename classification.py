# -*- coding: utf-8 -*-
"""
 @Time    : 2018/7/30 0030 下午 10:10
 @Author  : Shanshan Wang
 @Version : Python3.5
 加载已经处理好的数据集，使用神经网络对数据进行分类
"""
import pickle
import keras
from keras.models import  Sequential
from keras.layers import Dense
import numpy as np


seed=7
np.random.seed(seed)


class NN(object):
    def __init__(self,structureList,input_dim):
        self.model = Sequential()
        assert len(structureList)==3
        self.model.add(Dense(structureList[0],input_dim=input_dim,init='uniform',activation='relu'))
        self.model.add(Dense(structureList[1],init='uniform',activation='relu'))
        self.model.add(Dense(structureList[2],init='uniform',activation='softmax'))

    def train(self,X,Y):
        #编译模型
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        #训练模型
        self.model.fit(X,Y,nb_epoch=1000,batch_size=1)

        #评估模型
        scores=self.model.evaluate(X,Y)
        print('%s:%.2f%%'%(self.model.metrics_names[1],scores[1]*100))

    # 保存模型
    def save_mode(self):
        pass
    def result_predict(self,x):
        result=self.model.predict(np.array(x))
        print('result:',result)

if __name__ == '__main__':
    #加载数据集
    with open('dataset.pkl','rb') as f:
        num_positive_features, positive_y, num_negative_features, negative_y=pickle.load(f)
    allFeatures=num_negative_features+num_positive_features
    allY=positive_y+negative_y
    print(len(allFeatures))
    print(allFeatures)
    print(len(allY))
    with open('feature_index.pkl','rb') as f:
        feature2id,id2feature=pickle.load(f)

    cnn=NN([10,10,2],len(feature2id))
    cnn.train(np.array(allFeatures),np.array(allY))



    predictPath=[['liveinCity','cityLocatedinCountry','cityLocatedinCountry-1'],
                 ['borninCity'],
                 ['classmates','borninCity']]
    #将路径转换为特征值
    num_features=[]
    for row in predictPath:
        num_feature = np.zeros(len(id2feature))
        for i in range(len(row)):
            if feature2id.get(row[i])!=None:
                num_feature[i] =1
        num_features.append(num_feature)

    cnn.result_predict(num_features)
