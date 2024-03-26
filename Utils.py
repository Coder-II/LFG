def shuffle(data):
    import random
    indexed = {i: v for i, v in enumerate(data)}
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = []
    for i in index:
        data.append(indexed[i])
    return data

def getRatingMatrix(data,mode="UBase"):
    import numpy as np
    from Config import userNumber,movieNumber
    if mode=="MBase":
        userNumber,movieNumber = movieNumber,userNumber
    ratingList = [[0] * movieNumber for i in range(userNumber)]
    for uom,history in data.items():
        for mou,rating in history:
            ratingList[uom][mou]=rating
    _flatten = np.array(ratingList).flatten()
    meanV = np.average(_flatten[_flatten.nonzero()])
    for i in range(userNumber):
        for j in range(movieNumber):
            if ratingList[i][j] != 0:
                ratingList[i][j] -= meanV
    return ratingList,meanV

# 0 usage, call it manually
def maskUM():
    import pickle
    from Config import usersInfoPkl,moviesInfoPkl,userMaskPkl,movieMaskPkl
    from Utils import shuffle
    def mask(Ids):
        Partition = {}
        Ids = shuffle(Ids)
        _len = len(Ids)
        for g in range(5):
            if g not in Partition.keys(): Partition[g] = {}
            spl_start = int(_len * 0.2 * g)
            spl_end = int(_len * 0.2 * (g + 1))
            if g == 4: spl_end = _len
            Partition[g] = Ids[spl_start:spl_end]
        return Partition
    with open(usersInfoPkl, "rb") as f:
        usersInfo = pickle.load(f)
    userIds = usersInfo.keys()
    userPartition = mask(userIds)
    with open(userMaskPkl, "wb") as f:
        pickle.dump(userPartition, f)
    with open(moviesInfoPkl, "rb") as f:
        moviesInfo = pickle.load(f)
    movieIds = moviesInfo.keys()
    moviePartition = mask(movieIds)
    with open(movieMaskPkl, "wb") as f:
        pickle.dump(moviePartition, f)

def truncated_SVD(A,k=50,print_matirx=False):
    import numpy as np
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    if print_matirx:
        SS= np.sqrt(np.diag(S))
        print((U @ SS) @ (SS @ Vt))
    U = U[:, :k]
    S = S[:k]
    Vt = Vt[:k, :]
    SS = np.sqrt(np.diag(S))
    if print_matirx:
        print((U @ SS) @ (SS @ Vt))
    return (U @ SS),(SS @ Vt).T