from copy import deepcopy
from Config import *
import pickle
import torch
from torch import nn
import logging
logger=logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
from Evaluate import Gen
from Utils import shuffle
from Network import DLFG as generator
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument("--vIdx", type=int, required=True)
parser.add_argument("--mode", type=str,default="")
parser.add_argument("--mask", action="store_true")
parser.add_argument("--sparse", action="store_true")
args = parser.parse_args()
mask = args.mask
vIdx = args.vIdx

log_file="./pt/{}{}/FOLD-{}{}.log".format("sparse/" if args.sparse else "",dataVersion,args.vIdx+1,"_mask" if mask else "")
fh=logging.FileHandler(filename=log_file, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(fh)
logger.info("\n\
dataset=movielen-{}\n\
mask={}\n\
validIdx={}\n\
factorNumber={}\n\
useExtUserInfo={}\n\
useExtMovieInfo={}\n\
lr={}\n".format(dataVersion,mask,vIdx,factorNumber,useExtUserInfo,useExtMovieInfo,lr))

MSE = nn.MSELoss(reduction='none')

with open(usersInfoPkl, "rb") as f:
    usersInfo = pickle.load(f)
with open(moviesInfoPkl, "rb") as f:
    moviesInfo = pickle.load(f)

# User-base LFG
MBase_SVD_Path="./pkl/{}{}/{}_vIdx-{}_SVD.pkl".format("sparse/" if args.sparse else "",dataVersion,"maskMovie" if mask else "MBase",vIdx)
if os.path.exists(MBase_SVD_Path):
    with open(MBase_SVD_Path,"rb") as f:
          _, _, U,_, _  = pickle.load(f)
    FactorizedUserFactors = torch.tensor(U,dtype=torch.float).to(device)
else:   FactorizedUserFactors=U=None

UBase_SVD_Path="./pkl/{}{}/{}_vIdx-{}_SVD.pkl".format("sparse/" if args.sparse else "",dataVersion,"maskUser" if mask else "UBase",vIdx)
if os.path.exists(UBase_SVD_Path):
    with open(UBase_SVD_Path,"rb") as f:
        _, _, M,_, _ = pickle.load(f)
    FactorizedMovieFactors = torch.tensor(M,dtype=torch.float).to(device)
else:   FactorizedMovieFactors=M=None

G = generator(userInfoShape, userNumber, movieInfoShape, movieNumber, factorNumber, FactorizedUserFactors,FactorizedMovieFactors).to(device)
G_best = None

if args.mode != "":
    modes = [args.mode]
else:
    modes = ["UBase", "MBase"]
for mode in modes:
    optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    if mode == "UBase":
        dataPkl = UBaseDataPkl
        maskPkl = userMaskPkl
        normalized_shape = movieNumber
    elif mode == "MBase":
        dataPkl = MBaseDataPkl
        maskPkl = movieMaskPkl
        normalized_shape = userNumber

    with open(dataPkl, "rb") as f:
        data = pickle.load(f)
    with open(maskPkl, "rb") as f:
        group = pickle.load(f)

    normalize = nn.LayerNorm(normalized_shape=normalized_shape, elementwise_affine=False).to(device)

    minRMSE=999
    early_stop=0

    train_data = {}
    valid_data = {}

    _data = {}
    _train_data = {}
    _valid_data = {}
    for g in range(5):
        for uom, histories in data[g].items():
            if g == vIdx:
                if not uom in _valid_data.keys(): _valid_data[uom] = []
                _valid_data[uom] += histories
            else:
                if not uom in _train_data.keys(): _train_data[uom] = []
                _train_data[uom] += histories

    if mask:
        for g, uoms in group.items():
            if g != 0:
                for uom in uoms:
                    train_data[uom] = _train_data[uom]
                    valid_data[uom] = _valid_data[uom]
    else:
        for g, uoms in group.items():
                for uom in uoms:
                    train_data[uom] = _train_data[uom]
                    valid_data[uom] = _valid_data[uom]

    if args.sparse:
        train_data,valid_data = valid_data,train_data
    uoms = list(train_data.keys())
    uoms.sort()
    inputLen = {"UBase":movieNumber,"MBase":userNumber}[mode]
    start = time.time()
    for ep in range(epoch):
        G.train()
        train_batches = (len(uoms) + train_batch - 1) // train_batch
        for b in range(train_batches):
            _uoms = uoms[b*train_batch:(b+1)*train_batch]
            histories = [train_data[uom] for uom in _uoms]

            data=[]
            for history in histories:
                data.append(shuffle(history))

            extInfo=[]
            if mode == "UBase" and useExtUserInfo:
                extInfo = [usersInfo[uom]["extUserInfo"] for uom in _uoms]
            elif mode == "MBase" and useExtMovieInfo:
                extInfo = [moviesInfo[uom]["extMovieInfo"] for uom in _uoms]
            extInfo = torch.tensor(extInfo, dtype=torch.float).to(device)

            train_vecs = torch.tensor([[0.0] * inputLen for i in _uoms]).to(device)
            train_vecs_gt0 = torch.tensor([[0.0] * inputLen for i in _uoms]).to(device)

            for j,v0 in enumerate(data):
                for id, v in v0:
                    train_vecs_gt0[j][id] = 1.0
                    train_vecs[j][id] = v

            G.train()
            fake_R= G(extInfo,train_vecs,mode=mode)
            mse=MSE(fake_R,train_vecs)
            mse = mse[train_vecs>0]
            loss = torch.mean(mse)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (ep+1) % valid_interval == 0 and (ep+1) > valid_start:
            _,E=Gen(G, device,vIdx=vIdx,mode=mode,mask=mask,valid=True,sparse=args.sparse)
            if minRMSE > E:
                early_stop=0
                minRMSE = E
                G_best = deepcopy(G.state_dict())
            logger.info('Mode:{} Fold:{} Epoch: {}/{} cost:{:.4f} loss:{:.5f} RMSE:{:.5f} MinRMSE:{}'.format(mode,vIdx+1,ep+1, epoch,time.time()-start,loss.item(),float(E),'Unknown' if minRMSE > 900 else '{:.5f}'.format(minRMSE)))
        early_stop+=1
        if early_stop > patience:
            logger.info("Early stopped...")
            break
    G.load_state_dict(G_best)

for mode in modes:
    timeCost,E=Gen(G,device,vIdx=vIdx,mode=mode,mask=mask,valid=True,sparse=args.sparse)
    logger.info('Mode:{} valid RMSE:{}'.format(mode,float(E)))
    timeCost,E=Gen(G,device,vIdx=vIdx,mode=mode,mask=mask,sparse=args.sparse)
    logger.info('Mode:{} test RMSE:{}\n'.format(mode,float(E)))

torch.save(G_best, "./pt/{}{}/vIdx-{}_DLFG{}_best.pt".format("sparse/" if args.sparse else "",dataVersion,vIdx,"_mask" if mask else ""))