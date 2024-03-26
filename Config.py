import os.path

epoch=120
patience=10
test_batch=1000000
valid_start=0
valid_interval = 1

train_batch=1000000
lr=0.001
factorNumber=50

useExtUserInfo=True
useExtMovieInfo=True

simplify_output=True

folders_1=["pt","pkl"]
folders_2=["100k","1m"]
for folder1 in folders_1:
    if not os.path.exists(folder1): os.mkdir(folder1)
    _path=os.path.join(folder1,"sparse")
    if not os.path.exists(_path): os.mkdir(_path)
    for folder2 in folders_2:
        _path=os.path.join(folder1,folder2)
        if not os.path.exists(_path): os.mkdir(_path)
        _path=os.path.join(folder1, "sparse", folder2)
        if not os.path.exists(_path): os.mkdir(_path)

dataVersion="100k"
userNumber={"100k":943,"1m":6040}[dataVersion]
movieNumber={"100k":1682,"1m":3883}[dataVersion]

numExtUserInfo=3
numExtMovieInfo=18
userInfoShape=numExtUserInfo if useExtUserInfo else 0
movieInfoShape=numExtMovieInfo if useExtMovieInfo else 0

movieCSV="./Dataset/{}-movies.csv".format(dataVersion)
ratingCSV="./Dataset/{}-ratings.csv".format(dataVersion)
userCSV="./Dataset/{}-users.csv".format(dataVersion)

usersInfoPkl="./Dataset/{}-usersInfo.pkl".format(dataVersion)
moviesInfoPkl="./Dataset/{}-moviesInfo.pkl".format(dataVersion)

UBaseDataPkl="./Dataset/{}-UBaseData.pkl".format(dataVersion)
MBaseDataPkl="./Dataset/{}-MBaseData.pkl".format(dataVersion)

userMaskPkl="./Dataset/{}-userMask.pkl".format(dataVersion)
movieMaskPkl="./Dataset/{}-movieMask.pkl".format(dataVersion)

device="cpu"
# import torch
# if torch.cuda.is_available():
#   device="cuda"