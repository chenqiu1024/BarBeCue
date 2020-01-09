import os
from PIL import Image
import numpy as np


def load_data(filePath):
    '''
    图片转数组
    '''
    data = np.empty((42000,1,28,28),dtype='float32')    
    label = np.empty((42000,),dtype='uint8')
    #PIL 的 open() 函数用于创建 PIL 图像对象
    img = Image.open(filePath)
    #Convert the input to an array
    arr = np.asarray(img,dtype='float32')
    return arr

##图片数组转数字
def picToNum(arr,maxNum):
    numList = []
    old = -1
    total = 0
    fristNum = 0
    for num in arr:
        num = 0 if num<30 else num
        num = 255 if num>200 else num
        if old==-1:
            total = total + 1 
        elif num==old :
            total = total + 1 
        elif num!=old and total>0:
            numList.append(total)
            total = 0
        old = num
    endNum = numList[-1]
    endNum = endNum - fristNum ##计算去尾巴
    if endNum<=0: ## 如果小于等于0去掉
        del(numList[-1])
    else :
        numList[-1] = endNum
    
    del(numList[0]) ##去头
    
    ##连续数字根据阈值转换为[1,2]
    newNumList = []
    for i in numList:
        if i<maxNum:
            newNumList.append(1)
        else :
            newNumList.append(2)
    #print(newNumList)
    return newNumList
    
## 加载目录，遍历图片
files = os.listdir("/Users/qiudong/Projects/BarCodesScanner/Resources/cards/")
for file in files:
    arr = load_data("/Users/qiudong/Projects/BarCodesScanner/Resources/cards/"+file)
    nums = picToNum(arr[0],14) ##只去一行即可
    print({file:nums,'位数':len(nums)})  ##打印结果  
    
