#coding=utf-8
from xml.dom.minidom import parse  
import xml.dom.minidom

def KBPparser():
	infile = open('tac_kbp_2014(1).txt','w',encoding='utf-8')
	#cur.execute("create table KBP2014(id int auto_increment primary key, docid varchar(50),mention varchar(100),candidate varchar(100))")
	DOMTree = xml.dom.minidom.parse("dataset/tac_2014_kbp_english_entity_linking_training_AMR_queries.xml")
	Tree = DOMTree.documentElement
	data = Tree.getElementsByTagName('query')
	for item in data:
		#queryid = item.getAttribute('id')
		name = item.getElementsByTagName('name')[0].childNodes[0].data.encode('utf-8','ignore')
		print('name:',name)
		name = '_'.join(str(name,encoding='utf-8').split())
		wikititle = item.getElementsByTagName('wikititle')[0].childNodes[0].data.encode('utf-8','ignore')
		print('wikititle:',wikititle)
		docid = item.getElementsByTagName('docid')[0].childNodes[0].data.encode('utf-8','ignore')
		print('docid:',docid)
		print (name,wikititle,docid)
		#name = name.replace("'", "''")
		#wikititle = wikititle.replace("'", "''")
		strtemp = str(docid,encoding='utf-8') + '\t' + name + '\t' + str(wikititle,encoding='utf-8') + '\n'
		print(strtemp)
		infile.write(strtemp)
if __name__ == '__main__':
	KBPparser()
