import sys
sys.path.append("..")
from Config import *
import pickle
from Network import DLFG as generator
import torch
from torch.nn import MSELoss
import time


def Gen(G, device, vIdx, mode, mask=False, new=False, valid=False,sparse=False):
    assert not (new and mask)
    timeCost = 0

    if mode == "UBase":
        dataPkl = UBaseDataPkl
        maskPkl = userMaskPkl
    elif mode == "MBase":
        dataPkl = MBaseDataPkl
        maskPkl = movieMaskPkl

    with open(dataPkl, "rb") as f:
        data = pickle.load(f)
    with open(maskPkl, "rb") as f:
        group = pickle.load(f)

    with open(usersInfoPkl, "rb") as f:
        usersInfo = pickle.load(f)
    with open(moviesInfoPkl, "rb") as f:
        moviesInfo = pickle.load(f)

    train_data = {}
    valid_data = {}
    test_data = {}
    _data = {}
    _train_data = {}
    _valid_data = {}
    _test_data = {}

    for g in range(5):
        for uom, histories in data[g].items():
            if g == vIdx:
                if not uom in _valid_data.keys(): _valid_data[uom] = []
                _valid_data[uom] += histories
            else:
                if not uom in _train_data.keys(): _train_data[uom] = []
                _train_data[uom] += histories

    for uom, histories in data['test'].items():
        if not uom in _test_data.keys(): _test_data[uom] = []
        _test_data[uom] += histories

    if new:
        for uom in group[0]:
            train_data[uom] = _train_data[uom]
            valid_data[uom] = _valid_data[uom]
            test_data[uom] = _test_data[uom]
    elif mask:
        for g, uoms in group.items():
            if g != 0:
                for uom in uoms:
                    train_data[uom] = _train_data[uom]
                    valid_data[uom] = _valid_data[uom]
                    test_data[uom] = _test_data[uom]
    else:
        train_data = _train_data
        valid_data = _valid_data
        test_data = _test_data

    if sparse:
        train_data,valid_data = valid_data,train_data

    if valid:
        test_data = valid_data

    loss_fun = MSELoss(reduction="mean")
    uoms = list(train_data.keys())
    uoms.sort()
    inputLen = {"UBase": movieNumber, "MBase": userNumber}[mode]
    test_batches = (len(uoms) + test_batch - 1) // test_batch
    RMSE = []
    for b in range(test_batches):
        _uoms = uoms[b * test_batch:(b + 1) * test_batch]
        extInfo = []
        if mode == "UBase" and useExtUserInfo:
            extInfo = [usersInfo[uom]["extUserInfo"] for uom in _uoms]
        elif mode == "MBase" and useExtMovieInfo:
            extInfo = [moviesInfo[uom]["extMovieInfo"] for uom in _uoms]
        extInfo = torch.tensor(extInfo, dtype=torch.float).to(device)

        histories = [train_data[uom] for uom in _uoms]
        _test_data = [test_data[uom] for uom in _uoms]
        test_vecs = [[0] * inputLen for i in _uoms]
        test_vecs_gt0 = [[0] * inputLen for i in _uoms]
        train_vecs = [[0] * inputLen for i in _uoms]

        for j, v0 in enumerate(_test_data):
            for id, v in v0:
                test_vecs_gt0[j][id] = 1
                test_vecs[j][id] = v
        for j, v0 in enumerate(histories):
            for id, v in v0:
                train_vecs[j][id] = v
        train_vecs = torch.tensor(train_vecs, dtype=torch.float).to(device)
        test_vecs = torch.tensor(test_vecs, dtype=torch.float).to(device)

        start = time.time()
        # calc result
        G.eval()
        fake_ratings = G(extInfo, train_vecs, mode=mode)

        # calc rmse
        gt0_idx = [test_vecs > 0]
        test_vecs = test_vecs[gt0_idx]
        fake_ratings = fake_ratings[gt0_idx]
        fake_ratings[fake_ratings < 1] = 1
        fake_ratings[fake_ratings > 5] = 5
        MSEloss = loss_fun(test_vecs, fake_ratings)
        RMSE.append(torch.sqrt(MSEloss))

    timeCost += time.time() - start
    return timeCost, sum(RMSE) / len(RMSE)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--vIdx", type=int, required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    mask = args.mask
    new = args.new
    mode = args.mode

    appendix = "" if not (mask or new) else "_mask"
    if args.model == "":
        args.model = "./pt/{}{}/vIdx-{}_DLFG{}_best.pt".format("sparse/" if args.sparse else "",dataVersion,args.vIdx,appendix)
    # print(args.model)
    if mode == "":
        modes = ["UBase", "MBase"]
    else:
        modes = [mode]
    for mode in modes:
        G = generator(userInfoShape, userNumber, movieInfoShape, movieNumber, factorNumber, None,None).to(
            device)
        G.load_state_dict(torch.load(args.model),strict=False)
        timeCost, E = Gen(G, args.device, args.vIdx, mode=mode, mask=mask, new=new, valid=args.valid,sparse=args.sparse)
        if simplify_output:
            print("{:.6f}".format(float(E)))
        else:
            print('Fold:{} Cost:{:.4f} Mode:{} RMSE:{}'.format(args.vIdx + 1, timeCost, mode, float(E)))