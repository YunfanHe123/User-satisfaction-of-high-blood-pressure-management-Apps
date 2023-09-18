import pandas as pd
import json
import urllib.request
import urllib.parse
import urllib.request

# 加载数据集，review列为要进行情感分析的数据列
#data=pd.read_csv('E:\qinggan.csv',encoding='utf-8')
#review_list=data.review
# 获取AccessToken，AccessToken是后续调用百度AI的API端口起到密钥的作用
url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"
values = {
 'host':'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials',
 'client_id':'VMojOedoG6N5TtB7PdZtxYBG',  #从百度AI网页注册后创建应用获得的 API KEY
 'client_secret' : 'vw9szPpet7O5zTXOjvrdyCSLh8Uu4zhC' #从百度AI网页注册后创建应用获得的 Secret Key
}
# host中需要将client_id和client_secret的值相应修改
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=VMojOedoG6N5TtB7PdZtxYBG&client_secret=vw9szPpet7O5zTXOjvrdyCSLh8Uu4zhC'
request = urllib.request.Request(host)
request.add_header('Content-Type', 'application/json; charset=UTF-8')
response = urllib.request.urlopen(request)
content = response.read()
if (content):
    print(content)  # 获得了access token
