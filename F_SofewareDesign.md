# 前端软件框架设计或部分注意事项
目前是散写阶段，后续统一整理  
## UI设计注意事项
不单独设计UI，将UI需求并入整个测试系统一起开发，标准化同时能够节省人耗  
公司内必定会有一些测试总平台，比如OP就有O测等系统，UI合并能让本业务测试开发工作量减少，能让人员更专注于开发核心功能上
## 控制手机模块
APPIUM和UIautomator2均可，按用例适配也可以，但是控制逻辑不建议直接嵌入python代码内，因为最后交付于业务端不一定是懂Python的人员维护，而且手机UI实际更新频繁，需要大量人力维护。因此需要寻找到一个能交互给外包普通人维护的方式，可以使用以下架构或者技巧。  
1.使用单独的XML文件或者其他规范文本文件进行UI点击或者打开逻辑的控制。  （此优化确保能够让外包普通人参与到控制手机模块的手机控制顺序维护，要尽可能简单易懂）  
2.格式化读取手机控制逻辑XML文档，能够根据文档内容进行手机控制  
3.良好的LOG记录和出错处理，有完善的纠错机制  
4.与手机建立的每个连接应该属于独立的单例模式，方便多线程控制和不同模块调用获取，以防不必要的连接建立  

控制手机模块的关键点不是大量的手机控制代码，而是完善的文档和XML模板交付，能让业务和业务内外包人员维护起来，这样才是一个成功的手机控制模块交付  

## 音频输入输出模块
输入输出一般要做四五种类型，场景如下：
### 1.Record：正常测试设备播放场景    
技巧：使用sounddevice里面自带的record功能即可，缓冲时间足够一般都能满足要求  
### 2.Play：正常测试设备录制场景  
技巧：使用sounddevice里面自带的play功能即可，直接播放音频文件  
### 3.Play&Record：一般是通话和VOIP场景，DUT和AUX两者同时进行分别进行录制播放操作   
技巧：合并1，2两个功能即可，按照自己场景设置播放优先还是录制优先
### 4.Stream 流读取：用于实时处理场景，比如说和人物互动需要实时抓取和分析，或者是某些超长时间录制场景  
技巧：同使用sounddevices的stream record，回调函数里面把数据post出来或者按照自身业务二次开发即可，不好掌控，建议尽量使用1/2/3方式开发，稳定高效。
### 5.数据转换
技巧：pydub或者自己使用ffmpeg库，如果只是一些简单增益或者裁剪添加，可以自行稍微开发一些库函数，比较简单  

## 数据处理端
数据处理有两个方式  
1.在前端直接使用库函数进行处理（不推荐）  
2.前端只存储参数和文件，把文件和参数POST到后端进行统一计算（推荐）  
若使用1的逻辑，自行在前端软件里嵌入处理音频内容即可，若使用2，则着重看重后端设计和开发,详情关注[后端计算软件设计逻辑](B_SofewareDesign.md)。
## 报告/结果处理端
由于OP公司是着重excel交互的，所以在报告处理端使用excel相关处理比较多，如果贵司亦是如此，那推荐使用excel模板+xlwings库进行结果填入  
若要求较高，可以自行适配自身云测平台上数据库类型，在此不做太多讲述，按照自己的业务理解和技术理解去适配即可

