from Draw_Utils import *

dirs=["pkl/{}{}/SVD_","pkl/{}{}/biasSVD_","pt/{}{}/"]
labels=["SVD","biasSVD","DLFG"]
colors=['blue','green','red']
styles=["--","-.","-"]
mask=False
sparse=True

for dataVersion in ["100k", "1m"]:
  for mode in ["UBase", "MBase"]:
    plt.figure(figsize=(6, 7))
    plt.xlabel('训练回合', fontsize=16)
    plt.ylabel('验证集RMSE', fontsize=16)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    for i,_dir in enumerate(dirs):
     alldata=[]
     for vIdx in range(5):
         _SVD = True
         _apdx = "-mask"
         if _dir.startswith("pt"):
          _SVD = False
          _apdx = "_mask"
         if not mask: _apdx=""
         _path = (_dir+"FOLD-{}{}.log").format("sparse/" if sparse else "",dataVersion, vIdx+1,_apdx)
         alldata.append(getData(_path,mode,SVD=_SVD))
     alldata=process(alldata)
     _data=np.mean(alldata,axis=0)
     plt.plot( _data, label=labels[i],color=colors[i],linestyle=styles[i])
    plt.legend(fontsize=16)
    # plt.title(name)
    plt.tight_layout()
    plt.savefig("{}{}_MovieLens-{}.png".format("Sparse/" if sparse else "","User-base" if mode == "UBase" else "Movie-base",dataVersion),dpi=300)
    plt.show()
