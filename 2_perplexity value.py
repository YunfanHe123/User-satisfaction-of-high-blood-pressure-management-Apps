# 第二步 构建LDA模型 计算困惑度 通过困惑度确定主题数
import pandas as pd
import re
import jieba
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import math

# 加载数据集
data_all = pd.read_csv("D:\zhongwen.csv")

# 去除文本中的表情符号（只保留中英文和数字）
def clear_character(sentence):
    pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line = re.sub(pattern,'',sentence)
    new_sentence = ''.join(line.split())
    return new_sentence
train_text = [clear_character(data_all) for data_all in data_all['review']]
data_all['review_clear'] = train_text

# 添加词典
jieba.load_userdict("D:\jiebaDic.txt")

# 分词
train_seg_text = [jieba.lcut(s) for s in train_text]
data_all['review_seg'] = train_seg_text

# 加载停用词
stop_words_path = "D:\stop word.txt" #自己导入停用词表，可以根据结果不断更新

def get_stop_words():
    return set([item.strip() for item in open(stop_words_path,'r').readlines()])

stopwords = get_stop_words()

# 去除文本中的停用词
def drop_stopwords(line):
    line_clear = []
    for word in line:
        if word in stopwords:
            continue
        line_clear.append(word)
    return line_clear

train_st_text = [drop_stopwords(s) for s in train_seg_text]
data_all['review_st'] = train_st_text

# 过滤词长，只保留中文
def is_fine_word(words, min_length=2):
    line_clear = []
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    for word in words:
        if len(word) >= min_length and re.search(rule, word):
            line_clear.append(word)
    return line_clear
train_fine_text = [is_fine_word(s,min_length=2) for s in train_st_text]
data_all['review_fine'] = train_fine_text

#构建bigram 和 trigram 将一些高频词合并成一个单词
bigram = gensim.models.Phrases(train_fine_text,min_count=5,threshold=5)  # threshold是阈值，阈值越高，短语越少
trigram = gensim.models.Phrases(bigram[train_fine_text],threshold=5)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigram(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigram(texts):
    return [trigram_mod[doc] for doc in texts]

data_words_bigrams = make_bigram(train_fine_text)
data_words_trigrams = make_trigram(train_fine_text)

#构建词典 语料向量化表示
id2word = corpora.Dictionary(train_fine_text)     #create dictionary
texts = train_fine_text                          #create corpus
corpus = [id2word.doc2bow(text) for text in texts]   #term document frequency

# 构建LDA模型 获得困惑度 利用循环找到最佳K值（主题数）
# 构建函数
def computer_perplexity_values(testset,dictionary,size_dictionary,limit,start,step):
    perplexity_values = []
    # 构建LDA模型，将选取好的最佳主题数输入参数中num_topics
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=dictionary,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    eta=0.1,
                                                    alpha='asymmetric',
                                                    per_word_topics=True)#此处的参数只有两个需要调整，一个是eta,就是beta,文献里有
                                                                         #直接确定为0.1的，也可以让算法自己先学习一下，那就填‘auto’
                                                                         # 看一下效果，之后调整。alpha同理
        # 计算困惑度
        prep = 0.0
        prob_doc_sum = 0.0
        topic_word_list = []
        for topic_id in range(num_topics):
            topic_word = lda_model.show_topic(topic_id, size_dictionary)
            dic = {}
            for word, probability in topic_word:
                dic[word] = probability
            topic_word_list.append(dic)
        doc_topics_list = []
        for doc in testset:
            doc_topics_list.append(lda_model.get_document_topics(doc, minimum_probability=0))
        testset_word_num = 0
        for i in range(len(testset)):
            prob_doc = 0.0
            doc = testset[i]
            doc_word_num = 0
            for word_id, num in doc:
                prob_word = 0.0
                doc_word_num += num
                word = dictionary[word_id]
                for topic_id in range(num_topics):
                    prob_topic = doc_topics_list[i][topic_id][1]
                    prob_topic_word = topic_word_list[topic_id][word]
                    prob_word += prob_topic * prob_topic_word
                prob_doc += math.log(prob_word)
            prob_doc_sum += prob_doc
            testset_word_num += doc_word_num
        prep = math.exp(-prob_doc_sum / testset_word_num)
        perplexity_values.append(prep)
    return perplexity_values

#主题数的起止和步长根据自己数据特征预估自行修改，就是下面的‘30，2，1’
perplexity_values = computer_perplexity_values(corpus,id2word,len(id2word.keys()),30,2,1)

# 将困惑度与主题数绘制出来 主题数的起止及步长可以自行修改
limit=30;start=2;step=1;
x= range(start,limit,step)
plt.plot(x,perplexity_values)
plt.xlabel("num topics")
plt.ylabel("perplexity value")
plt.legend(("perplexity value"),loc='best')
plt.show()

# 显示各个主题的困惑度 选择困惑度最低时的主题数
for m,pv in zip(x,perplexity_values):
    print("num topics = ",m,"has perplexity value of",round(pv,4))







