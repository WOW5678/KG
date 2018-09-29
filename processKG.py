# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/29 0029 下午 7:11
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import zipfile
import os
from urllib.parse import unquote
import csv

def extract_hudongbaike(zipFileName,write_file,mode):
    '''
    从hudongbaike数据中提取到所有试题和部分有效的属性
    :return:
    '''
    if not os.path.exists(hudong_dir):
        print('目录不存在，请仔细核对~')
        return
    z=zipfile.ZipFile(zipFileName,'r')
    for filename in z.namelist():
        print(filename)
        content_string=z.read(filename)
        ttl_to_triples(content_string,write_file,mode)

def ttl_to_triples(content_string,write_file,mode):
    '''
    解析字符串，从中提取出三元组来
    '''
    writerF=open(write_file,'w',newline='',encoding='utf-8')
    writer=csv.writer(writerF)
    # 使用\n对字符串进行分割
    entities_List=str(content_string,encoding='utf-8').split('\n\n')
    for line in entities_List:
        tuple=[]
        lineList=line.split('\n')
        for i in range(len(lineList)):
            line_=unquote(lineList[i],'utf-8')
            if line_.strip().startswith('<http://zhishi.me/hudongbaike/resource'):
                entity=line_.strip().replace('>','').split('/')[-1]
            if line_.strip().startswith(mode):
                property=line_.strip().replace('>','').split('/')[-1]
                value=lineList[i+1].strip().replace('@zh','').replace(';','').replace('.','').replace('"','')
                tuple.append([entity,property,value])
        print(tuple)
        writer.writerows(tuple)
    writerF.close()

def extract_hudongbaike2(zipFileName,write_file):
    '''
    从hudongbaike数据中提取到所有试题和部分有效的属性
    :return:
    '''
    if not os.path.exists(hudong_dir):
        print('目录不存在，请仔细核对~')
        return
    z = zipfile.ZipFile(zipFileName, 'r')
    for filename in z.namelist():
        print(filename)
        content_string = z.read(filename)
        #ttl_to_triples(content_string, write_file, mode)

        '''
        解析字符串，从中提取出三元组来
         '''
        writerF = open(write_file, 'w', newline='', encoding='utf-8')
        writer = csv.writer(writerF)
        # 使用\n对字符串进行分割
        entities_List = str(content_string, encoding='utf-8').split('\n\n')
        for line in entities_List:
            tuple = []
            lineList = line.split('\n')
            lineList=[row for row in lineList if len(row.strip())>0]
            assert len(lineList)==3
            lineList=[unquote(line, 'utf-8') for line in lineList]
            entity = lineList[0].strip().replace('>', '').split('/')[-1]
            property = lineList[1].strip().replace('>', '').split('/')[-1]
            value = lineList[2].strip().replace('@zh', '').replace(';', '').replace('.', '').replace('<', '').replace('>','')
            tuple.append([entity, property, value])
            print(tuple)
            writer.writerows(tuple)
        writerF.close()

def extract_hudongbaike3(zipFileName,write_file):
    '''
    从hudongbaike数据中提取到所有试题和部分有效的属性
    :return:
    '''
    if not os.path.exists(hudong_dir):
        print('目录不存在，请仔细核对~')
        return
    z = zipfile.ZipFile(zipFileName, 'r')
    for filename in z.namelist():
        print(filename)
        content_string = z.read(filename)
        #ttl_to_triples(content_string, write_file, mode)

        '''
        解析字符串，从中提取出三元组来
         '''
        writerF = open(write_file, 'w', newline='', encoding='utf-8')
        writer = csv.writer(writerF)
        # 使用\n对字符串进行分割
        tuple_List = str(content_string, encoding='utf-8').split('\n')
        tuple_List = [row for row in tuple_List if len(row.strip()) > 0]
        tuple_List= [unquote(line, 'utf-8') for line in tuple_List]
        for line in tuple_List:
            tuple=[]
            line_list=line.replace('\r','').split(' ')
            print(line_list)
            entity = line_list[0].strip().replace('>', '').split('/')[-1]
            property = line_list[1].strip().replace('>', '').split('/')[-1]
            value = line_list[2].strip().replace('@zh', '').replace(';', '').replace('.', '').replace('<', '').replace('>','')
            tuple.append([entity, property, value])
            print(tuple)
            writer.writerows(tuple)
        writerF.close()

        # for line in entities_List:
        #     tuple = []
        #     lineList = line.split('\n')
        #
        #     assert len(lineList)==3
        #
        #     entity = lineList[0].strip().replace('>', '').split('/')[-1]
        #     property = lineList[1].strip().replace('>', '').split('/')[-1]
        #     value = lineList[2].strip().replace('@zh', '').replace(';', '').replace('.', '').replace('<', '').replace('>','')
        #     tuple.append([entity, property, value])
        #     print(tuple)
        #     writer.writerows(tuple)


if __name__ == '__main__':
    # zhishi.me源数据的存储路径
    zhishime_data_dir = u'F:\数据总结\KG\zhishime-ttl\zhishime1'
    # 存放处理结果的路径
    process_dir='E:\codePractices\KG\zhishi_me\\resultData'
    hudong_dir = os.path.join(zhishime_data_dir, 'hudongbaike')

    #propertyZipPath=os.path.join(hudong_dir, '3.0_hudongbaike_infobox_properties_zh.zip')
    #extract_hudongbaike(propertyZipPath,'resultData/entity_property_vlues.csv',mode='<http://zhishi.me/hudongbaike/property')
    #labelZipPath=os.path.join(hudong_dir,'3.0_hudongbaike_labels_zh.zip')
    #extract_hudongbaike(labelZipPath,'resultData/entity_label.csv',mode='<http://www.w3.org/2000/01/rdf-schema')

    #redirectZipPath=os.path.join(hudong_dir,'3.0_hudongbaike_redirects_zh.zip')
    #extract_hudongbaike2(redirectZipPath,'resultData/entity_redirect_page.csv')

    # disambiguationZipPath=os.path.join(hudong_dir,'3.0_hudongbaike_disambiguations_zh.zip')
    # extract_hudongbaike2(disambiguationZipPath,'resultData/entity_distributions.csv')

    # instanceZipPath=os.path.join(hudong_dir,'hudongbaike_instance_types_zh.zip')
    # extract_hudongbaike3(instanceZipPath,'resultData/instance_type.csv')