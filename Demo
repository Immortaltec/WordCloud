import jieba
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
 
# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
 
 
# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('data/baidu_stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
 
 
inputs = open('data/bingdundun.txt', 'r', encoding='utf-8')
outputs = open('data/outbingdundun.txt', 'w',encoding='utf-8')
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg + '\n')
outputs.close()
inputs.close()
mask = np.array(Image.open("data/image01.jpg"))#模板图片
inputs=open('data/outbingdundun.txt','r',encoding='utf-8')
mytext=inputs.read()
wordcloud =WordCloud(mask=mask,width=3000,height=3000,background_color='white',margin=1,
                    max_words=300,min_font_size=10,max_font_size=None,repeat=False,
                    font_path='font/FZKaTong-M19S.ttf').generate(mytext)#生成云图
wordcloud.to_file('wordcloudbing.jpg')
inputs.close()

plt.figure(dpi=150)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
