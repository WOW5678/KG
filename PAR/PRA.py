# -*- coding: utf-8 -*-
"""
 @Time    : 2018/7/30 0030 下午 4:38
 @Author  : Shanshan Wang
 @Version : Python3.5

实现一个简单的PRA 算法
使用的特征值是二值特征（路径出现与否，出现为1，否则为0）
目标关系：borninCity,即为borninCity这种关系创建模型
"""
import pandas as pd
import itertools
import random
import numpy as np
import pickle


class Edge(object):
    def __init__(self,label,start,end):
        self.label=label
        self.start=start
        self.end=end
        self.str=self.__str__()
    def __str__(self):
        return self.label

class Node(object):
    def __init__(self,name):
        self.name=name
        self.neighbors=set()
        #a dict of the form {node: [edge1, edge2, ...]}
        #保存着它的每个邻居结点相连的边
        self.neighbors2edges={}
        # a dict of the form {node: [edge1.str, edge2.str, ...]}
        #这个字典的创建是为了便于输出的美观
        self.neighbors2edgesstr = {}
        # 每种类型的边的个数
        # store the fan out for each edge label, a dict of the form {edge_label: fan_out (i
        self.edge_fan_out = {}
    def __str__(self):
        return 'Node({})'.format(self.name)

    def add_edge(self,edge):
        self.neighbors.add(edge.end)
        self.neighbors2edges[edge.end]=self.neighbors2edges.get(edge.end,[])+[edge]
        #self.neighbors2edgesstr[edge.end]=self.neighbors2edgesstr.get(edge.end,[])+[edge.str]
        #self.edge_fan_out[edge.label]=self.edge_fan_out.get(edge.label,0)+1


class Graph(object):
    def __init__(self):
        self.nodes={} # nodes are indexed by name，即{name:Node,....}

    def get_node(self,name,create=False):
        # 如果create指示为True 并且该节点不存在 则创建
        if create and not name in self.nodes:
            self.nodes[name] = Node(name)

        return self.nodes.get(name)

    def partical_build_from_df(self,df):
        '''
        从DF数据中创建子图
        :return:
        '''
        # 注意 针对每个事实 创建的都是双向图
        for idx, row in df.iterrows():
            head = self.get_node(row['head'], create=True)
            tail = self.get_node(row['tail'], create=True)
            relation = row['relation']
            head.add_edge(Edge(relation,head,tail))


class PRA(object):
    def __init__(self,graph):
        self.graph=graph

    ''''
       从指定节点出发 宽度遍历 获取所有可达的节点的节点路径
       如指定Tom节点，outputs中保存着到达每个节点经过的节点列表（包含起始节点和结束节点）
    '''

    def bfs_node_seqs(self, start_node, max_depth):
        outputs = {}
        queue = [(start_node, [start_node], 0)]
        while queue:
            (vertex, path, level) = queue.pop(0)
            for node in vertex.neighbors-set(path):
                outputs[node] = outputs.get(node, []) + [path + [node]]
                if level + 1 < max_depth:
                    queue.append((node, path + [node], level + 1))
        # outputs:{终节点1：[[开始节点,..终节点1],[],[]]，终结点2：[开始节点，...,终结点2]}
        return outputs

    ''''
    由节点序列转换成边的序列
    '''
    def get_edge_seqs(self,node_seqs,invert=False):
        edge_seq=set()
        print('len(node_seqs):',len(node_seqs))
        for node_seq in node_seqs:
            debug_print(node_seq)
            possiable_edge_seqs=[]
            for i in range(1,len(node_seq)):
                possiable_edge_seqs.append(node_seq[i-1].neighbors2edges.get(node_seq[i]))
            #print('possiable_edge_seqs:',possiable_edge_seqs)
            #paths.update(itertools.product(*possible_paths))
            edge_seq.update(itertools.product(*possiable_edge_seqs))
        #     #print(edge_seq)
        # print('+++++++++++++++++++++++++++')
        # for row in edge_seq:
        #     debug_print(row)

        return edge_seq

    ''''
    指定关系，得到具有该关系的实体对，作为正例样本
    '''
    def get_relation_pairs(self,relation):
        positiveSamples=[]
        featureSet=[]
        for node in self.graph.nodes.values():
            outputs=self.bfs_node_seqs(node,max_depth=20)
            for end_node,paths in outputs.items():
                pathSet=self.get_edge_seqs(paths)
                for path in pathSet:
                    #如果当前关系在第一个 并且边长度为1,说明找到了这样的实体对
                    if path[0].label==relation and len(path)==1:
                        positiveSamples.append([node,end_node])
                        featureSet.append(path)
        return positiveSamples,featureSet

    ''''
    通过替换正例中一个实体，得到每个正例的负样例
    positiveSamples:[[],[],[]...]
    '''
    def get_negative_samples(self,postiveSamples_):
        negative_samples=[]
        negative_featureSet=[]
        for sample in postiveSamples_:
            # 替换掉尾实体，替换的规则是随机替换（注意：这里最好使用同一类的实体进行替换）
            sample[1]=self.random_replace(sample[0],sample[1])
            negative_samples.append(sample)
            outputs=self.bfs_node_seqs(sample[0],max_depth=20)
            print('=============================')
            debug_print(sample)
            nodePaths=outputs.get(sample[1])
            sample_paths=self.get_edge_seqs(nodePaths)
            negative_featureSet+=sample_paths

        return negative_samples,negative_featureSet

    def random_replace(self,entity_1,entity_2):
        # 从邻居结点中随机选择一个节点
        while True:
            newEntity=random.choice(list(self.graph.nodes.values()))
            #print('newEntity:',newEntity)
            if entity_1!=newEntity and newEntity!=entity_2:
                return newEntity

def debug_print(ne_list):
    l=[]
    for n in ne_list:
        l.append(n.__str__())
    print(l)
if __name__ == '__main__':
    # 数据集
    data = [['Tom', 'classmates', 'Bob'],
            ['Bob','classmates-1','Tom'],
            ['Tom', 'borninCity', 'Pairs'],
            ['Pairs', 'borninCity-1','Tom'],
            ['Tom', 'liveinCity', 'Lyon'],
            ['Lyon', 'liveinCity-1','Tom'],
            ['Tom', 'nationality', 'FR'],
            ['FR', 'nationality-1','Tom'],
            ['Bob', 'borninCity', 'Pairs'],
            ['Pairs', 'borninCity-1','Bob' ],
            ['Pairs', 'cityLocatedinCountry', 'FR'],
            ['FR', 'cityLocatedinCountry-1','Pairs'],
            ['Lyon', 'cityLocatedinCountry', 'FR'],
            [ 'FR', 'cityLocatedinCountry-1','Lyon']]
    data_zig = pd.DataFrame(data, columns=['head', 'relation', 'tail'])

    #创建图
    graph=Graph()
    graph.partical_build_from_df(data_zig)
    start_node=graph.get_node('Tom')
    print(start_node.neighbors2edges)
    for key,val in start_node.neighbors2edges.items():
        print('key:',key)
        debug_print(val)
    pra=PRA(graph)

    # 确定目标关系，并寻找到该目标关系对应的正样例和负样例
    # 并且为每种样例生成特征集合
    positiveSamples,positive_featureSet=pra.get_relation_pairs('borninCity')
    negative_samples,negative_featureSet=pra.get_negative_samples(positiveSamples)
    print('=======================')
    allFeatures=[]
    for row in negative_featureSet:
        allFeatures.append([item.label for item in row])
    for row in positive_featureSet:
        allFeatures.append([item.label for item in row])
    print(allFeatures)

    #构建特征值，路径出现则为1，否则为0，类似于one-hot编码
    featureBags=[]
    for row in allFeatures:
        for item in row:
            if item not in featureBags:
                featureBags.append(item)
    print(featureBags)
    feature2id={feature:id for id,feature in enumerate(featureBags)}
    id2feature={id:feature for feature,id in feature2id.items()}
    print('feature2id:',feature2id)

    with open('feature_index.pkl','wb') as f:
        pickle.dump([feature2id,id2feature],f)


    #基于词典向量化每个特征样本
    num_positive_features=[]
    num_negative_features=[]
    for row in positive_featureSet:
        num_feature=np.zeros(len(id2feature))
        for i in range(len(row)):
            num_feature[i]=feature2id[row[i].label]
        num_positive_features.append(num_feature)
    for row in negative_featureSet:
        num_feature = np.zeros(len(id2feature))
        for i in range(len(row)):
            if feature2id.get(row[i].label)!=None:
                num_feature[i] =1
        num_negative_features.append(num_feature)
    print(len(positive_featureSet))
    print(len(negative_featureSet))
    positive_y=[[1,0]]*len(positive_featureSet)
    negative_y=[[0,1]]*len(negative_featureSet)
    with open('dataset.pkl','wb') as f:
        pickle.dump([num_positive_features,positive_y,num_negative_features,negative_y],f)










