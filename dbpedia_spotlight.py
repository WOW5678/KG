# -*- coding:utf-8 -*-
"""
@Time: 2019/09/18 20:34
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 使用API 完成命名实体连接（NEL）
还可以使用封装好的python包pyspotlight(https://pypi.org/project/pyspotlight/)
完成NEL的任务

"""
import requests
from IPython.core.display import display, HTML

class APIError(Exception):
    def __init__(self,status):
        self.status=status

    def __str__(self):
        return 'APIError:status={}'.format(self.status)

# Parameters
# 'text' - text to be annotated
# 'confidence' -   confidence score for linking

params = {"text": "My name is Sundar. I am currently doing Master's in Artificial Intelligence at NUS. I love Natural Language Processing.",
          "confidence": 0.35}

# Base URL for spotlight API
base_url="http://api.dbpedia-spotlight.org/en/annotate"
headers={'accept': 'text/html'}

# GET Request

res = requests.get(base_url, params=params, headers=headers)
if res.status_code != 200:
    # Something went wrong
    raise APIError(res.status_code)
# Display the result as HTML in Jupyter Notebook
display(HTML(res.text))
print(res.text)