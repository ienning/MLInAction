from numpy import *
# 通过阈值进行比较分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    print("retArray is ", shape(retArray))
    if threshIneq == "lt":
        # 这里做的是数组操作不是单个元素的操作
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray
# 找到最佳单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ["lt", "gt"]:
                threshVal = (rangeMin + float(j) * stepSize)
                # 弱分类器
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                #print("predictedVals:\n",predictedVals)
                #print("labelMat:\n", labelMat)
                #print(shape(predictedVals))
                #print(shape(labelMat))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                #print(split: dim %f, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshVal
                    bestStump["ineq"] = inequal
    return bestStump, minError, bestClassEst

# 基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump["alpha"] = alpha
        weakClassArr.append(bestStump)
        print("classEst", classEst.T)
        # classLabels 和 classEst是相同值相乘就是正值，否则就是负值。刚好前面乘以负一是符合公式的
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

# AdaBoost分类函数
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]["dim"], classifierArr[i]["thresh"], classifierArr[i]["ineq"])
        aggClassEst += classifierArr[i]["alpha"]*classEst
        print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split("\t"))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c="b")
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], "b--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("The Area Under the Curve is: ", ySum* xStep)
