from numpy import *
import copy

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    ssCnt = {}
    DD = copy.deepcopy(D)

    ##print("list is ", list(D))
    for tid in DD:
        #print("cycle times ", tid)
        Cks = copy.deepcopy(Ck)
        for can in Cks:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(list(D)))
    #print("numItems is ", numItems)
    #print("ssCnt is ", ssCnt)
    retList = []
    supportData = { }
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            # that can ensure before(index from 0 to k-2) value equal, after(that is last index) value not equal,
            # so can get different list
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    #print("retList is ", retList)
    return retList

def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    DD = copy.deepcopy(D)
    ##print("list D ", list(D))
    L1, supportData = scanD(DD, C1, minSupport)
    #print(L1)
    L = [L1]
    #print("Test result ", supportData)
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        #print("Ck is ", Ck, "\n")
        DD = copy.deepcopy(D)
        Lk, subK = scanD(DD, Ck, minSupport)
        #print("LK is ", Lk)
        supportData.update(subK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf = 0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet - conseq, "--->", conseq, "conf:", conf)
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m+1):
        hmp1 = aprioriGen(H, m+1)
        hmp1 = calcConf(freqSet, hmp1, supportData, br1, minConf)
        if (len(hmp1) > 1):
            rulesFromConseq(freqSet, hmp1, supportData, br1, minConf)

