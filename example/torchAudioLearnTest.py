import torchaudio
from matplotlib import pyplot
import torch
import torch.nn.functional as F
import time
start_time = time.time()  # 获取当前时间戳
# 执行待计时的代码块
print(torchaudio.__version__)
devices=torch.device("mps")
cpuDevice=torch.device("cpu")
# info=torchaudio.info("/Users/luyongyi/yongyi的文档/OPPO工作经验/AudioDealofOP/torch4audio/test1.mp3")
# print(info)
data,rate=torchaudio.load("des.wav")
tri,triRate=torchaudio.load("tri.wav")
triDeives=tri[0].view(1,1,-1).to(devices)
dataDeives=data[0].view(1,1,-1).to(devices)
output = F.conv1d(dataDeives, triDeives ,padding="valid",stride=1)
#print(output.device)
print(torch.argmax(output).cpu().item()/rate)
end_time = time.time()  # 获取当前时间戳
elapsed_time = end_time - start_time  # 计算时间差
print("代码执行时间为：{:.6f}秒".format(elapsed_time))