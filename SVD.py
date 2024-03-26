import pickle
import time
import numpy as np
from Config import *
from Utils import getRatingMatrix, truncated_SVD

class SVD(object):
    def __init__(self, userNum, itemNum, f, prefix, logger, addBias=False, epoch=35, lr=0.01, beta=0.1,
                 save_model=True):
        super(SVD, self).__init__()
        self.epoch = epoch
        self.userNum = userNum
        self.itemNum = itemNum
        self.lr = lr
        self.beta = beta
        self.f = f
        self.save_model = save_model
        self.bu = np.zeros(self.userNum)
        self.bi = np.zeros(self.itemNum)
        self.prefix = prefix
        self.addBias = addBias
        self.logger=logger

    def load(self, meanV, U, M, bu, bi):
        assert meanV is not None and U is not None and M is not None
        self.meanV = meanV
        self.U = U
        self.M = M
        self.bu = bu if bu is not None else self.bu
        self.bi = bi if bi is not None else self.bi

    def fit(self, train, val=None):
        minRmse = 999
        start = time.time()
        for i in range(self.epoch):
            sumRmse = 0.0
            for sample in train:
                uid = int(sample[0])
                iid = int(sample[1])
                vij = float(sample[2])
                _mean = self.meanV[uid] if type(self.meanV) == list or self.meanV.size != 1 else self.meanV
                p = _mean + self.bu[uid] + self.bi[iid] + np.sum(self.U[uid] * self.M[iid])
                error = vij - p
                sumRmse += error ** 2

                self.U[uid] += self.lr * (error * self.M[iid] - self.beta * self.U[uid])
                self.M[iid] += self.lr * (error * self.U[uid] - self.beta * self.M[iid])

                if self.addBias:
                    self.bu[uid] += self.lr * (error - self.beta * self.bu[uid])
                    self.bi[iid] += self.lr * (error - self.beta * self.bi[iid])
            trainRmse = np.sqrt(sumRmse / train.shape[0])

            if val.any():
                _, valRmse = self.evaluate(val)
                self.logger.info("Epoch %d cost time %.4f, train RMSE: %.4f, validation RMSE: %.4f" % (
                    i, time.time() - start, trainRmse, valRmse))
                if minRmse > valRmse:
                    minRmse = valRmse
                    if self.save_model:
                        model = (self.meanV, self.U, self.M, self.bu, self.bi)
                        pickle.dump(model,
                                    open("./pkl/{}{}/{}{}SVD.pkl".format("sparse/" if args.sparse else "", dataVersion,
                                                                         self.prefix, "bias" if self.addBias else ""),
                                         'wb'))
            else:
                self.logger.info("Epoch %d cost time %.4f, train RMSE: %.4f" % (i, time.time() - start, trainRmse))
                if i == 24:
                    model = (self.meanV, self.U, self.M, self.bu, self.bi)
                    pickle.dump(model,
                                open("./pkl/{}{}/{}{}SVD.pkl".format("sparse/" if args.sparse else "", dataVersion,
                                                                     self.prefix, "bias" if self.addBias else ""),
                                     'wb'))
        self.logger.info("min valid RMSE:{}".format(minRmse))

    def evaluate(self, val):
        import torch
        gt_matrix = torch.tensor([[0] * self.itemNum for i in range(self.userNum)], dtype=float).to(device)
        for uid, iid, rating in val:
            gt_matrix[uid][iid] = rating

        _Mean = torch.tensor(self.meanV, dtype=float).to(device)
        _U = torch.tensor(self.U, dtype=float).to(device)
        _M = torch.tensor(self.M, dtype=float).to(device)
        _bu = torch.tensor(self.bu, dtype=float).to(device)
        _bi = torch.tensor(self.bi, dtype=float).to(device)
        loss_fun = torch.nn.MSELoss()

        start = time.time()
        # calc result
        pred_matrix = _U @ (_M.T)
        pred_matrix += _Mean
        pred_matrix += _bu.unsqueeze(1).repeat(1, self.itemNum)
        pred_matrix += _bi

        # calc rmse
        gt0_idx = [gt_matrix > 0]
        test_vecs = gt_matrix[gt0_idx]
        fake_ratings = pred_matrix[gt0_idx]
        fake_ratings[fake_ratings < 1] = 1
        fake_ratings[fake_ratings > 5] = 5
        mse = loss_fun(test_vecs, fake_ratings)
        rmse = torch.sqrt(mse)

        return time.time() - start, rmse


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vIdx", type=int, required=True)
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--epoch", type=int, default=75)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    vIdx = args.vIdx
    mode = args.mode
    if args.new: args.mask = args.test = True

    logger = None
    if (not args.test) and (not args.valid):
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        log_file = "./pkl/{}{}/{}SVD_FOLD-{}{}.log".format("sparse/" if args.sparse else "", dataVersion,
                                                           "bias" if args.bias else "", vIdx + 1,
                                                           "-mask" if args.mask else "")
        fh = logging.FileHandler(filename=log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)

    if mode == "":
        modes = ["UBase", "MBase"]
    else:
        modes = [args.mode]
    if not args.mask:
        # original SVD UBase/MBase
        for mode in modes:
            prefix = "{}_vIdx-{}_".format(mode, vIdx)
            dataPkl = {"UBase": UBaseDataPkl, "MBase": MBaseDataPkl}[mode]
            with open(dataPkl, "rb") as f:
                data = pickle.load(f)

            train_data = {}
            valid_data = {}

            for g in range(5):
                if g == vIdx:
                    for uom, histories in data[g].items():
                        if not uom in valid_data.keys(): valid_data[uom] = []
                        valid_data[uom] += histories
                else:
                    for uom, histories in data[g].items():
                        if not uom in train_data.keys(): train_data[uom] = []
                        train_data[uom] += histories

            test_data = {}
            for uom, histories in data['test'].items():
                if not uom in test_data.keys(): test_data[uom] = []
                test_data[uom] += histories

            if args.sparse:
                train_data, valid_data = valid_data, train_data
            train = []
            val = []
            test = []
            for dataset in [(train_data, train), (valid_data, val), (test_data, test)]:
                for uom1, history in dataset[0].items():
                    for uom2, rating in history:
                        dataset[1].append([uom1, uom2, rating])
            train = np.array(train)
            val = np.array(val)
            test = np.array(test)
            umNum = {"UBase": [userNumber, movieNumber], "MBase": [movieNumber, userNumber]}
            svd = SVD(userNum=umNum[mode][0], itemNum=umNum[mode][1], f=factorNumber, prefix=prefix, logger=logger,
                      addBias=args.bias,
                      epoch=args.epoch, save_model=(not args.nosave))

            if args.test or args.valid:
                with open("./pkl/{}{}/{}{}SVD.pkl".format("sparse/" if args.sparse else "", dataVersion, prefix,
                                                          "bias" if args.bias else ""), "rb") as f:
                    meanV, U, M, bu, bi = pickle.load(f)
                svd.load(meanV, U, M, bu, bi)
                timeCost, rmse = svd.evaluate(test if args.test else val)
                if simplify_output:
                    print("{:.6f}".format(rmse))
                else:
                    print("{} RMSE: {:.6f}".format("test" if args.test else "valid", rmse))

            else:
                logger.info("\n" + mode)
                Matrix, Mean = getRatingMatrix(data=train_data, mode=mode)
                SU, SM = truncated_SVD(Matrix, factorNumber)
                svd.load(Mean, SU, SM, None, None)
                svd.fit(train, val=val)
                with open("./pkl/{}{}/{}{}SVD.pkl".format("sparse/" if args.sparse else "", dataVersion, prefix,
                                                          "bias" if args.bias else ""), "rb") as f:
                    meanV, U, M, bu, bi = pickle.load(f)
                svd.load(meanV, U, M, bu, bi)
                timeCost, rmse = svd.evaluate(test)
                logger.info("test RMSE: {:.6f}".format(rmse))
    else:
        # SVD with maksked UBase/MBase
        for mode in modes:
            prefix = {"UBase": "maskUser_", "MBase": "maskMovie_"}[mode]
            prefix += "vIdx-{}_".format(vIdx)
            dataPkl = {"UBase": UBaseDataPkl, "MBase": MBaseDataPkl}[mode]
            maskPkl = {"UBase": userMaskPkl, "MBase": movieMaskPkl}[mode]

            umNum = {"UBase": [userNumber, movieNumber], "MBase": [movieNumber, userNumber]}
            svd = SVD(userNum=umNum[mode][0], itemNum=umNum[mode][1], f=factorNumber, prefix=prefix, logger=logger,
                      addBias=args.bias,
                      epoch=args.epoch, save_model=(not args.nosave))

            train_data = {}
            valid_data = {}
            test_data = {}
            with open(dataPkl, "rb") as f:
                data = pickle.load(f)
            with open(maskPkl, "rb") as f:
                group = pickle.load(f)
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

            if args.new:
                for uom in group[0]:
                    test_data[uom] = _test_data[uom]
            else:
                for g, uoms in group.items():
                    if g != 0:
                        for uom in uoms:
                            train_data[uom] = _train_data[uom]
                            valid_data[uom] = _valid_data[uom]
                            test_data[uom] = _test_data[uom]

            if args.sparse:
                train_data, valid_data = valid_data, train_data
            train = []
            val = []
            test = []
            for dataset in [(train_data, train), (valid_data, val), (test_data, test)]:
                for uom1, history in dataset[0].items():
                    for uom2, rating in history:
                        dataset[1].append([uom1, uom2, rating])
            train = np.array(train)
            val = np.array(val)
            test = np.array(test)

            if args.test or args.valid:
                with open("./pkl/{}{}/{}{}SVD.pkl".format("sparse/" if args.sparse else "", dataVersion, prefix,
                                                          "bias" if args.bias else ""), "rb") as f:
                    meanV, U, M, bu, bi = pickle.load(f)
                svd.load(meanV, U, M, bu, bi)
                timeCost, rmse = svd.evaluate(test if args.test else val)
                if simplify_output:
                    print("{:.6f}".format(rmse))
                else:
                    print("{} RMSE: {:.6f}".format("test" if args.test else "valid", rmse))
            else:
                logger.info("\n" + mode)
                Matrix, Mean = getRatingMatrix(data=train_data, mode=mode)
                SU, SM = truncated_SVD(Matrix, factorNumber)
                svd.load(Mean, SU, SM, None, None)
                svd.fit(train, val=val)
                with open("./pkl/{}{}/{}{}SVD.pkl".format("sparse/" if args.sparse else "", dataVersion, prefix,
                                                          "bias" if args.bias else ""), "rb") as f:
                    meanV, U, M, bu, bi = pickle.load(f)
                svd.load(meanV, U, M, bu, bi)
                timeCost, rmse = svd.evaluate(test)
                logger.info("test RMSE: {:.6f}".format(rmse))
