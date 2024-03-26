from Draw_Utils import *
dirs=["pkl/sparse/100k/SVD_","pkl/sparse/100k/biasSVD_","pt/sparse/100k/"]
colors=['blue','green','red']
styles=["-.","--","-"]
for mode in ["UBase", "MBase"]:
    plt.figure(figsize=(6, 7))
    plt.xlabel('训练回合', fontsize=16)
    plt.ylabel('验证集RMSE', fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    for i,_dir in enumerate(dirs):
        alldata=[]
        for vIdx in range(5):
            _path="./{}FOLD-{}.log".format(_dir,vIdx+1)
            alldata.append(getData(_path,mode,False if _dir.startswith("pt") else True))
        alldata=process(alldata)
        _data=np.mean(alldata,axis=0)
        plt.plot(_data, label='5折平均',color=colors[i],linestyle=styles[i])
        plt.plot(len(_data)-1,_data[-1],'s',markersize=4,color='dark'+colors[i])
    plt.tight_layout()
    plt.show()
    input()