# -*- coding:utf-8 -*-
"""
@Time: 2019/08/14 14:18
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
from urllib.parse import quote
import urllib
import urllib.request
import json
import numpy as np

#第一个例子
#输入实体指称项名称，返回对应实体（entity）的列表，json格式
input_entity=quote('葡萄')
input_url='http://shuyantech.com/api/cndbpedia/ment2ent?q='
url=input_url+input_entity
response=urllib.request.urlopen(url)
print(response.read().decode('utf-8'))

#第二个例子
from urllib.parse import quote
import urllib
import json
import numpy as np

#输入实体名，返回实体全部的三元组知识
input_entity_name = quote('糖尿病')
input_url='http://shuyantech.com/api/cndbpedia/avpair?q='
url=input_url+input_entity_name
response=urllib.request.urlopen(url)
print(response.read().decode('utf-8'))

#第三个例子
from urllib.parse import quote
import urllib
import json
import numpy as np
#给定实体名和属性名，返回属性值
input_entity_name=quote('沈阳航空航天大学')
input_attr=quote('外文名称')
input_url='http://shuyantech.com/api/cndbpedia/value?q='
url=input_url+input_entity_name+'&attr='+input_attr
response=urllib.request.urlopen(url)
print(response.read().decode('utf-8'))