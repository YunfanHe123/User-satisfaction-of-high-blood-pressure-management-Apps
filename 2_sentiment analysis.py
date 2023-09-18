import pandas as pd
import urllib3
import json
import time
import numpy

# 读取数据
data=pd.read_csv('E:\qinggan.csv',encoding='ansi')
review_list=data.review
# 设置最后输出的标签（0 1 2）及正向预测概率
labels=[]
label_prediction =[]

# 读入 access token ，该数值在第一个文件中生成
access_token='24.f0da1c09cc03127565e4731bf93accaf.2592000.1695613966.282335-37845346'
http=urllib3.PoolManager()
url='https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token='+access_token

# 调用服务进行分析
for i in range(len(review_list)):
    if (i + 1) % 2 == 0:
        time.sleep(1)  # 每分析两条数据就暂停1秒，因为百度AI免费版的QRS上限是2，如果付费可以达到20，这里相应的修改
    params={'text':review_list[i]}
    encoded_data = json.dumps(params).encode('utf-8')
    request = http.request('POST',
                           url,
                           body=encoded_data,
                           headers={'Content-Type': 'application/json'})
    result = str(request.data, 'utf-8')
    a = json.loads(result)
    output = a['items'][0]
    labels.append(output['sentiment'])
    label_prediction.append(output['positive_prob'])
    print("第 " + str(i + 1) + "条评论已完成分析")
    
# 分析结果输出，label只有3中 0 1 2
data['label']=labels   # 0 负向 1 中性 2 正向
data['positive_prob']=label_prediction # 正向预测的概率
data.to_csv('D:\qingganjieguo.csv',encoding="utf-8")