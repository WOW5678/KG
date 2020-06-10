# -*- coding:utf-8 -*-
'''
Create time: 2020/6/10 11:13
@Author: 大丫头
'''

# 创建节点类
class Vertex:
    def __init__(self,key):
        self.id=key
        self.connectedTo={}

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr]=weight

    def __str__(self):
        return str(self.id)+'connectedTo:'+str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]

# 定义图类
class Graph:
    def __init__(self):
        self.vertList={}
        self.numVertices=0

    def addVertex(self,key):
        self.numVertices=self.numVertices+1
        newVertex=Vertex(key)
        self.vertList[key]=newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self,f,t,cost=0):
        if f not in self.vertList:
            nv=self.addVertex(f)
        if t not in self.vertList:
            nv=self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t],cost)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

# 使用真实的数据组建图
import pandas as pd
nodes=pd.read_csv('ASOIAF_nodes.csv')
nodes=nodes['Id']
num_nodes=nodes.shape[0]
print(nodes.head())


edges=pd.read_csv('ASOIAF_edges.csv',usecols=[0,1])
print(edges.head())


g=Graph()

for i in range(num_nodes):
    g.addVertex(nodes[i])

for i in range(len(edges)):
    edge=edges.iloc[i]
    g.addEdge(edge['Source'],edge['Target'])
    g.addEdge(edge['Target'],edge['Source'])

for v in g:
    for w in v.getConnections():
        print("%s---%s"%(v.getId(),w.getId()))

# Traverse the Nodes with Depth First Search

def traverseAll(g):
    visited=[]
    connected=0
    for v in g:
        if v not in visited:
            visited_v=dfs(v)
            visited+=visited_v
            connected+=1
            print('\n\n Node {} has {} connected components'.format(v.getId(),len(visited_v)))
            print('\n\n')

    print("Total number of connected components is {}".format(connected))
    print(len(visited))

def dfs(s):

    stack=[s]
    visited=[]
    while stack:
        v=stack.pop()
        if v not in visited:
            visited.append(v)
            for w in v.getConnections():
                if w not in visited:
                    stack.append(w)
    return visited


traverseAll(g)

# Breadth First Search to Find the Distance Between Characters

def bfs(g,s,t):
    visited={}
    d={}
    for v in g:
        visited[v]=False
        d[v]=float('inf')

    d[s]=0
    queue=[s]
    visited[s]=True

    while queue:
        v=queue.pop(0)

        for w in v.getConnections():
            if visited[w]==False:
                visited[w]=True
                queue.append(w)
                d[w]=d[v]+1
    print('The distance from {} to {} is {}'.format(s.getId(),t.getId(),d[t]))

#举个例子
s = g.getVertex('Addam-Marbrand')
t = g.getVertex('Emmond')
bfs(g,s,t)


## Visualization with Graphviz

from graphviz import Digraph

dot=Digraph(comment='VIP Graph')

for i in range(num_nodes):
    dot.node(nodes[i])

for i in range(len(edges)):
    edge=edges.iloc[i]
    dot.edge(edge['Source'],edge['Target'])
print(dot.source)

dot.render('VIP-graph.gv',view=True)
