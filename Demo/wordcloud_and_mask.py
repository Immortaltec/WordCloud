import jieba
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from scipy.ndimage import gaussian_gradient_magnitude

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
outputs = open('data/outbingdunduntwo.txt', 'w',encoding='utf-8')

#对文本文件预处理,分词
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg + '\n')
outputs.close()
inputs.close()

# Mask image图片的处理
mask_color = np.array(Image.open('data/parrot-by-jose-mari-gimenez2.jpg'))
mask_color = mask_color[::3, ::3]
mask_image = mask_color.copy()
mask_image[mask_image.sum(axis=2) == 0] = 255#把黑色转变成白色

# Edge detection
edges = np.mean([gaussian_gradient_magnitude(mask_color[:, :, i]/255., 2) for i in range(3)], axis=0)
mask_image[edges > .08] = 255
im=Image.fromarray(mask_image)
im.save('parrot-change.jpg')
#处理完毕后黑色背景变成了白色,就可以正常识别轮廓了

# make WordCloud绘制词云
inputs =open('data/outbingdundun.txt','r',encoding='utf-8')
mytext=inputs.read()
wordcloud =WordCloud(mask=mask_image,margin=1,relative_scaling=0,
                    max_words=10000,min_font_size=10,max_font_size=None,repeat=False,collocations=False,
                     #colormap='inferno',#颜色风格
                     mode='RGB',font_path='font/FZKaTong-M19S.ttf').generate(mytext)#生成云图
wordcloud.generate(mytext)
wordcloud.to_file('wordcloud-parrot.jpg')
inputs.close()

#如果我们希望生成的图片的颜色也和原始图像接近。因此，需要加上重新
# Create coloring from image 重新着色,
image_colors = ImageColorGenerator(mask_color)
wordcloud.recolor(color_func=image_colors)

#保存至图片
wordcloud.to_file('wordcloud-parrot2.jpg')
inputs.close()

# Plot
plt.figure(dpi=200)#通过这里可以放缩
plt.imshow(wordcloud)#plt显示图片
plt.axis('off')#不显示坐标轴
plt.show()#显示图片
#图片有空隙说明词集选择不当
