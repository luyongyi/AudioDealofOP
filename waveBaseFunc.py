
from scipy import signal,fft
import numpy as np
def Log(A):
    print(A)
def spectrum(data,fftSize,rate=48000,overlap=0.5):
    '''
    与audition对应频率分析图表
    ：param data：数据段
    ：param fftSize：分析fft大小，一般为512/1024/2048/4096/...等
    ：param rate：采样率，用于横坐标对齐
    ：param overlap：重叠率,一般为50%，可以自行选择
    '''
    dataSplit=[]
    window=signal.windows.hanning(fftSize)
    splitLinespace=range(0,len(data),int(fftSize*overlap)) #会漏掉最后一部分数据，但是不多，问题不大
    for i in splitLinespace:
        if(i+fftSize)>len(data):
            break
        dataSplit.append(data[i:i+fftSize])
    fftSum=np.zeros(fftSize//2)
    for i in dataSplit:                                     #分区
        i=i*window                                          #加窗
        fftData=abs(fft.fft(i))                             #fft
        fftSum+=np.array(fftData[:fftSize//2])              #求和
    fftAvg=fftSum/len(dataSplit)                            #平均
    
    fftSpectrum=20*np.log10(abs(fftAvg)/(fftSize//4))       #幅度换算与校准
    fftFreq=np.linspace(0,rate//2,fftSize//2,endpoint=False)#横坐标计算
    return fftFreq,fftSpectrum

def delay(data,trigger):
    '''
    定位data内，trigger波形的位置
    ：param data：数据段
    ：param trigger：trigger波形
    '''
    if(len(trigger)>=len(data)):
        return False
    matching=[]
    for i in range(len(data)-len(trigger)):
        matching.append(np.dot(data[i:i+len(trigger)],trigger))
    
    location=matching.index(max(matching))

    if(type(location)==type([])):
        Log("multi Trigger Location")
        return location[0]
    return location

