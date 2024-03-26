# AudioDealofOP
除常规业务项目外，其余技术方面比较深入的工作内容精炼总结，主要是自动化、硬件场景搭建、音频杂音识别相关内容。有必要提醒一下，由于html和Github支持的Latex语法问题，所以不单独修改语法，需要查阅请自行clone到本地

# 基础
## 标准化

### 数字音频标准化:波形大小标准化、dBFS和与Audition对齐
1.soundfile库会把能读取的wav格式文件，波形数值范围自动标准化为-1—1范围，取名为$sample_{read}$，其中$sample_{read}\not=0$   
2.部分格式soundfile库无法读取，比如32bit的wav文件或者其他格式，读取可能为对应PCM值，需要自行换算为$sample/int[depth].max$，标准化为-1—1的范围,$sample_{read}$，其中$sample_{read}\not=0$    
3.针对波形和FFT对应的dBFS幅度计算方法：  
原理：dBFS计算公式-> $20*log_{10}(sample\div S_{max})$ 
  
所以针对波形换算为dBFS幅度，使用直接映射计算：dBFS= $20 * log_{10}{sample}_{read}$ ，其中$sample_{read}\not=0$，若遇到sample点为0，自行映射到$-\infty$或其他低于该最小幅度值以外，比如16bit PCM文件，最小dBFS为$20*log_{10}1/ int16_{MAX}=-90.3dBFS$ ,聪明的读者会问：我得知的16bit PCM文件动态范围是96dBFS,为什么在你这却少了-6dB,因为那个计算方法是计算-32768~32767整个动态范围，而我说的公式只计算1~32767范围，而整个动态范围计算简洁明了：$20*log_{10}(uint16_{max})=96dB$
上述能延申出几个知识点，比如：6dB相差一倍而非3dB，在此不做延申

所以针对FFT换算dBFS幅度，也可以直接按上述方式换算为dBFS,如果是自行使用的话，同样计算方式下dBFS对比值是有意义的，当然看绝对值没有意义，所以需要和Audition的频率分析工具内幅度对齐，那样其他小组或部门分析或整合的阈值可以直接套用到自身系统上  
如果直接对齐Audition，直接上代码，可以自行对比，若有错误此处不做阐述，请自行校验和修改
```python
from scipy import signal,fft
import numpy as np
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
    splitLinespace=range(0,len(data),int(fftSize*overlap))  #会漏掉最后一部分数据，但是不多，问题不大
    for i in splitLinespace:                                #
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
    return fftFreq,fftSpectrum                              #返回横坐标和纵坐标
```

### 设备标准化(Audio Precition平替)：声卡&采集标准化/线材连接标准化/音源标准化
最好使用AP开发，如果没有AP，那就使用下面方案
需要拆解耳机识别原理/耳机CODEC MAX输入输出电压，熟悉分压原理


#### 声卡&采集标准化 
以下参考按照测试场景需求搭建，没有标准答案
声卡参考：瑞声新谱1284等
麦克风参考：Grass 46AM
线材参考：2k电阻 3.5mm - BNC 1.5V电源
  

有线耳机拓扑参考：共地;手机输出->声卡input;声卡output->2k电阻->手机麦克风
外放拓扑参考：人工耳，麦克风矩阵
蓝牙耳机拓扑参考：人工头

#### 线材连接标准化
线材参考：2k电阻 3.5mm - BNC 1.5V电源 
手机模拟耳机场景与声卡连接使用2k电阻，，声卡output->2k电阻->手机麦克风
数字耳机使用声卡output->1.5V电源->数字转接器麦克风 ，确保数字转接器麦克风端电压>声卡output

#### 音源标准化
时域频域都要有自己思考，各个场景都要定制。无法对准，时域对齐场景使用自相关函数对齐


### UI操作/ID调用与声卡录制标准
外放场景：前置准备->设备录制->手机播放->手机暂停->停止录制->复位->算法计算->报告输出  
录制场景：前置准备->手机录制->设备播放->设备停止->手机停止录制->复位->算法计算->报告输出  
通话场景：前置准备->建立通话-> 设备播放->设备录制->设备停止->复位->算法计算->报告输出  

# 进阶
## 设备录制与手机播放时间差校准
如上述，部分场景设备录制需要先于手机，部分场景手机录制先于设备播放，此场景有至少2s差异，如果遇到uiautomator断连情况，可能延长到20s甚至50s，因此需要做时间差校准

### 自相关函数（时域）（精度最高，误差25/48000 s内）
在测试音源中嵌入trigger音源，后使用trigger音源逐采样点计算自相关值，最大即为trigger位置，可以使用二次校验排除误差情况
trigger定制需要巧妙，时间要在降噪算法生效前尽可能长，尽可能寻找不被手机算法捕捉直接播放的特制音源
算法可见wavBaseFunc.delay函数  

```python  
from nump
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
```
可以使用cuda优化长音源计算时间    
```python
import cuda
@cuda.jit
def delay(data,trigger,result):
    id=cuda.threadIdx.x+cuda.BlockIdx*cuda.blockDim.x
    gS=cuda.gridDim.x*cuda.blockDim.x
    for i in range(id,len(data)-len(tri),gS):
        temp=0
        for j in range(len(tri)):
            temp+=data[i+j]*tri[j]
        result[i]=temp          #或者不用cuda，使用signal.correlate也可以有这样的效果，运算速度也比python自身硬算快
```
由于你使用自相关，互相关函数一般是直接取最大值，所以对应的trigger音源应该有以下注意事项：  
1.trigger的整体RMS值应该尽可能大    
2.trigger整体时长应该在1s左右，尽可能小  
3.trigger的波形特征不能为单频音量，波形特征应该明显    



### 频谱阈值（频域）（精度较差）
就是使用某单频或复频音做校准，但是时间误差稍大，1024的FFT SIZE根据自身场景定制阈值即可，阈值小可能误判，阈值大了误差会增大   
整体思路就是判定好某个阈值，单个频点或者多个频点符合阈值筛选范围的，再对这个时间范围和对应频点幅度做峰值检测，就能够识别出最匹配时间，但是前提是FFT SIZE要跟音源波形时间有一定对应关系，或者在峰值检测处，对峰值相等的波峰做居中选择，在此不做赘述，可以自行做实验。




 