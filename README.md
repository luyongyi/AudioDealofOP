# AudioDealofOP
除常规业务项目外，其余技术方面比较深入的工作内容精炼总结

# 基础音源处理函数
## 单位换算与标准化

### 波形大小标准化、dBFS和与Audition对齐
1.soundfile库会把能读取的wav格式文件，波形数值范围自动标准化为-1—1范围，取名为$sample_{read}$，其中$sample_{read}\not=0$   
2.部分格式soundfile库无法读取，比如32bit的wav文件或者其他格式，读取可能为对应PCM值，需要自行换算为$sample/int[depth].max$，标准化为-1—1的范围,$sample_{read}$，其中$sample_{read}\not=0$    
3.针对波形和FFT对应的dBFS幅度计算方法：  
原理：dBFS计算公式-> $20*log_{10}(sample\div S_{max})$ 
  
所以针对波形换算为dBFS幅度，使用直接映射计算：dBFS=$20*log_{10}sample_{read}$，其中$sample_{read}\not=0$，若遇到sample点为0，自行映射到$-\infty$或其他低于该最小幅度值以外，比如16bit PCM文件，最小dBFS为$20*log_{10}1/ int16_{MAX}=-90.3dBFS$ ,聪明的读者会问：我得知的16bit PCM文件动态范围是96dBFS,为什么在你这却少了-6dB,因为那个计算方法是计算-32768~32767整个动态范围，而我说的公式只计算1~32767范围，而整个动态范围计算简洁明了：$20*log_{10}(uint16_{max})=96dB  
上述能延申出几个知识点，比如：6dB相差一倍而非3dB，在此不做延申

所以针对FFT换算dBFS幅度，也可以直接按上述方式换算为dBFS,如果是自行使用的话，同样计算方式下dBFS对比值是有意义的，当然看绝对值没有意义，所以需要和Audition的频率分析工具内幅度对齐，那样其他小组或部门分析或整合的阈值可以直接套用到自身系统上  
如果直接对齐Audition，直接上代码，可以自行对比，若有错误此处不做阐述，请自行校验和修改
```python
from scipy import signal,fft

def spectrum(data,fft_size,rate=48000,overlap=0.5)
    '''
    与audition对应频率分析图表
    ：param data：数据段
    ：param fft_size：分析fft大小，一般为512/1024/2048/4096/...等
    ：param rate：采样率，用于横坐标对齐
    ：param overlap：重叠率,一般为50%，可以自行选择
    '''
    dataSplit=[]
    window=signal.windows.hanning(fft_size)
    
```



 