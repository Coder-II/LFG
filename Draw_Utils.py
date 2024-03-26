import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimSun"]
plt.rcParams["axes.unicode_minus"]=False
import numpy as np

def process(arr):
    maxLen=max([len(x) for x in arr])
    arr2=[]
    for d in arr:
        e = maxLen - len(d)
        l = min(d)
        for i in range(e):
            d.append(l)
        arr2.append(d)
    ## or
    # arrLen = [len(x) for x in arr]
    # avgLen = int(sum(arrLen) / len(arrLen))
    # arr2 = []
    # for d in arr:
    #     if len(d) < avgLen:
    #         e = avgLen - len(d)
    #         l = min(d)
    #         for i in range(e):
    #             d.append(l)
    #     elif len(d) > avgLen:
    #         d = d[:avgLen]
    #     arr2.append(d)
    return arr2

def getData(_path,mode,SVD):
    with open(_path,"r+") as f:
        content=f.read()
    lines=content.split("MBase" if SVD else "Early stopped")[0 if mode=="UBase" else 1].split("\n")
    data=[]
    for line in lines:
        idx=line.find("validation RMSE: " if SVD else " MinRMSE:")
        if idx != -1:
            val=float(line[idx + 17:]) if SVD else float(line[idx-7:idx])
            data.append(val)
    _min=min(data)
    _idx=data.index(_min)
    return data[:(_idx+1)]