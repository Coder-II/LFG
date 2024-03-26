from Draw_Utils import *

dirs=["STD","Batchnorm","Dropout","Bias"]

for _dir in dirs[1:]:
     for mode in ["UBase", "MBase"]:
        plt.figure(figsize=(6, 7))
        alldata0 = []
        for vIdx in range(5):
            _path="./Ablation/{}/FOLD-{}.log".format("SparseSTD" if _dir == "RandomMask" else dirs[0], vIdx + 1)
            alldata0.append(getData(_path,mode,SVD=False))
        alldata0=process(alldata0)
        _data0 = np.mean(alldata0, axis=0)
        alldata=[]
        for vIdx in range(5):
            _path = "./Ablation/{}/FOLD-{}.log".format(_dir,vIdx+1)
            alldata.append(getData(_path,mode,SVD=False))
        alldata=process(alldata)
        _data=np.mean(alldata,axis=0)
        plt.plot(_data0, label='原模型5折平均',color="red",linestyle="--")
        plt.plot(_data, label='消融模型5折平均',color="blue")
        if len(_data) != len(_data0):
            plt.plot(len(_data0)-1,_data0[-1],'s',markersize=4,color='darkred')
            plt.plot(len(_data)-1,_data[-1],'s',markersize=4,color='darkgreen')
        name="Ablation_{}_{}_MovieLens_DLFG.png".format(_dir,"User-base" if mode == "UBase" else "Movie-base")
        plt.xlabel('训练回合',fontsize=16)
        plt.ylabel('验证集RMSE',fontsize=16)
        plt.legend(fontsize=16)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.tight_layout()
        plt.savefig("./Ablation/{}".format(name),dpi=300)
        plt.show()
