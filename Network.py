from torch import nn
import torch

class DLFG(nn.Module):
    def __init__(self, userInfoShape, userNumber, movieInfoShape, movieNumber, factorNumber, FactorizedUserFactors,FactorizedMovieFactors):
        super(DLFG, self).__init__()
        self.movieBiasMatrix = nn.Parameter(torch.tensor([0.0] * movieNumber, dtype=torch.float), requires_grad=True)
        self.userBiasMatrix = nn.Parameter(torch.tensor([0.0] * userNumber, dtype=torch.float), requires_grad=True)
        if FactorizedUserFactors is None: FactorizedUserFactors = torch.tensor([[0.0] * factorNumber for i in range(userNumber)])
        if FactorizedMovieFactors is None: FactorizedMovieFactors = torch.tensor([[0.0] * factorNumber for i in range(movieNumber)])
        self.FactorizedUserFactors = nn.Parameter(FactorizedUserFactors,requires_grad=True)
        self.FactorizedMovieFactors = nn.Parameter(FactorizedMovieFactors, requires_grad=True)
        self.UGen = self.Gen(userInfoShape,movieNumber,factorNumber)
        self.MGen = self.Gen(movieInfoShape,userNumber,factorNumber)

    def Gen(self,extInfoShape,baseInfoShape,factorNumber):
        _Gen=nn.Sequential(
            nn.Linear(extInfoShape + baseInfoShape, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024, 0.05),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.8),
            nn.Linear(1024, factorNumber + 1)
            , nn.Tanh()
        )
        return _Gen

    def matrixRecontruct(self,GeneratedFactors,FactorizedFactors,meanV,bias):
        result=torch.sum(GeneratedFactors[:,:-1].unsqueeze(1)*FactorizedFactors,axis=2)
        result+=(meanV+GeneratedFactors[:,-1]).unsqueeze(1).repeat(1,len(FactorizedFactors))
        result += bias
        return result

    def forward(self, extInfo,ratings,mode,extract=False):
        history = torch.cat((extInfo,ratings),len(extInfo.shape)-1)
        if mode == "UBase":
            GeneratedFactors = self.UGen(history)
            FactorizedFactors = self.FactorizedMovieFactors
            bias = self.movieBiasMatrix
        elif mode == "MBase":
            GeneratedFactors = self.MGen(history)
            FactorizedFactors = self.FactorizedUserFactors
            bias = self.userBiasMatrix


        global_mean = torch.mean(ratings[ratings>0])
        # meanV=[3.5]*len(ratings)
        meanV = []
        for rating in ratings:
            rating=rating[rating>0]
            _mean=torch.mean(rating)
            if torch.isnan(_mean) or len(rating) < 10:_mean=global_mean
            meanV.append(_mean)
        meanV = torch.tensor(meanV,dtype=torch.float).to(ratings.device)

        if extract:
            return meanV, GeneratedFactors, self.matrixRecontruct(GeneratedFactors,FactorizedFactors, meanV,bias)
        else:
            return self.matrixRecontruct(GeneratedFactors,FactorizedFactors, meanV,bias)