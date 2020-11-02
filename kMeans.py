from numpy import *
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = []
        for i in curLine:
            fltLine.append(float(i))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # 创建 K个 质心
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            # compare point to k point distance, getting min distance
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # if min distance change，setting changed flag
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # setting the value of i index, the value is centroid position.
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            # find the points that belong to cent position
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# 二分K-均值聚类算法
def bikmeans(dataSet, k, distMeans=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # mean every column, axis = 0 is meaning every column, axis = 1 is meaning every row.
    # Because dataSet is matrix whose meaning is two-dimension, which need to become list.
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    # calculate the nearest distance every sample to center point
    for j in range(m):
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            # getting current cluster
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            print("current list ", ptsInCurrCluster)
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("The bestCentToSplit is : ", bestCentToSplit)
        print("The len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        # direct assignment bestCluster to best divided cluster
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    print("centList is ", centList)
    return centList, clusterAssment

# 球面上两点之间的距离
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi/180) * sin(vecB[0, 1]*pi/180)
    b = cos(vecA[0, 1] * pi/180) * cos(vecB[0, 1]*pi/180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a+b)*6371.0

import