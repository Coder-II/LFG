import pickle
from Utils import shuffle
from Config import *

movieCSV="./Dataset/{}-movies.csv".format(dataVersion)
ratingCSV="./Dataset/{}-ratings.csv".format(dataVersion)
userCSV="./Dataset/{}-users.csv".format(dataVersion)

def preProcess():
    extUsersInfo={}
    with open(userCSV, "r+", encoding="utf-8") as f:
        userTable = f.readlines()
    for row in userTable:
        if row == "":continue
        userId,age,sex,job = row.split(",")
        job=job.replace("\n","")
        extUsersInfo[userId]=[float(x) for x in [age,sex,job]]

    extMoviesInfo={}
    with open(movieCSV,"r+",encoding="utf-8") as f:
        movieTable=f.readlines()[1:]  # get rid of field names
    for row in movieTable:
        if row == "": continue
        movieInfo = row.split(",")[:-1]  # get rid of titles\n
        extMoviesInfo[movieInfo[0]] = [float(x) for x in movieInfo[1:]]

    movieid2idx = {id:idx for idx,id in enumerate(extMoviesInfo.keys())}
    userid2idx = {id:idx for idx,id in enumerate(extUsersInfo.keys())}

    usersInfo = {userid2idx[userId]:{"extUserInfo":extUserInfo,"history":[]} for userId,extUserInfo in extUsersInfo.items()}
    moviesInfo = {movieid2idx[movieId]: {"extMovieInfo":extMovieInfo,"history":[]} for movieId,extMovieInfo in extMoviesInfo.items()}

    with open(ratingCSV,"r+",encoding="utf-8") as f:
        ratingTable = f.readlines()
        for row in ratingTable:
            if row == "": continue
            userId,movieId,rating=row.split(",")
            userIdx = userid2idx[userId]
            movieIdx = movieid2idx[movieId]
            rating= int(rating.replace("\n",""))
            usersInfo[userIdx]["history"].append([movieIdx, rating])
            moviesInfo[movieIdx]["history"].append([userIdx, rating])

    with open(usersInfoPkl,"wb") as f:
        pickle.dump(usersInfo,f)

    with open(moviesInfoPkl,"wb") as f:
        pickle.dump(moviesInfo,f)

def data_split(inputInfoPkl,outputDataPkl):
    data={"test":{}}
    with open(inputInfoPkl, "rb") as f:
        inputInfo = pickle.load(f)
    for idx, info in inputInfo.items():
        history = shuffle(info["history"])
        test_part = int(len(history)/5)
        data["test"][idx] = history[-test_part:]
        history=history[:-test_part]
        _len=len(history)
        for g in range(5):
            if g not in data.keys(): data[g] = {}
            spl_start = int(_len * 0.2 * g)
            spl_end = int(_len * 0.2 * (g + 1))
            if g == 4: spl_end = _len
            data[g][idx] = history[spl_start:spl_end]
    with open(outputDataPkl, "wb") as f:
        pickle.dump(data, f)

preProcess()
print("Preparing Cross Validation Dataset...")
data_split(usersInfoPkl,UBaseDataPkl)
data_split(moviesInfoPkl,MBaseDataPkl)
print("Done.")