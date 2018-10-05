import os
import re   #正则表达式
import time
import babel
start=time.clock()#记录程序开始时间
with open('yuanzhu.stl','r') as infile:
    content=infile.read()#将所有stl数据读到content中
reg='vertex (.*?)\n'#正则
vertexs=re.findall(reg,content)#保存所有在'vertexs'和回车之间的数据
print("未去重前的点个数：",len(vertexs))
s=sorted(set(vertexs),key=lambda x:(x))#排序，去除重复点
print("去除重复点后点个数：",len(s))
dic={}
n=0
for each in s:#给字典中每个点添加键值
    dic[each]=n
    n=n+1
outfile=open('yuanzhu.off','w')#打开保存文件
outfile.write('OFF'+'\n'+str(len(s))+' '+str(len(vertexs)/3)+' '+str(0)+'\n')#off文件开头
for each in s:#保存所有的顶点
    outfile.write(each+'\n')
k=0
for f in vertexs:   #保存所有的面，每个面是三个点的键值
    if k%3==0:
        outfile.write(str(3))
    outfile.write(' '+str(dic[f]))
    if k%3==2:
        outfile.write('\n')
    k=k+1
outfile.close()
end=time.clock()
print("程序运行时间：%.03f"%(end-start))
os.system("pause")



