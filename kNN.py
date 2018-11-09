# -*- coding:UTF-8 -*-
from numpy import *
import operator
from os import listdir

def creatDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]);
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 0 1st dimention
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile: repeat matrix
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # ==1 add as row
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # sort matrix as the index
    classCount = {}  # dict
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # simple sorted function,just sort the classCount dict.
    # key is the 2nd value of (B,2)
    return (sortedClassCount[0][0]);


def file2Matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(word2Int(listFromLine[-1])))
        index += 1
    print classLabelVector
    return returnMat, classLabelVector


def word2Int(string):
    level = 0
    if string == "largeDoses":
        level = 3
    elif string == "smallDoses":
        level = 2
    elif string == "didntLike":
        level = 1
    return level


def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    range = maxVal - minVal
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minVal, (m, 1))
    normalDataSet = normalDataSet / tile(range, (m, 1))
    return normalDataSet, range, minVal


def datingClassTest():
    testRatio = 0.1
    datingDataMat, datingLabels = file2Matrix("E:/Desktop/ML/datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    testNum = int(testRatio * m)
    errorCount = 0.0
    for i in range(testNum):
        result = classify0(normMat[i, :], normMat[testNum:m, :], datingLabels[testNum:m], 3)
        print "classifier--> %d || true label--> %d " % (result, datingLabels[i])

        if (result != datingLabels[i]):
            errorCount += 1
    print "total error rate is %f" % (float(errorCount / testNum))


def classsifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTime = float(raw_input("percentage of game time?"))
    flyMile = float(raw_input("fly miles in a year?"))
    iceCream = float(raw_input("ice per year?"))
    datingDataMat, datingLabels = file2Matrix("E:\Desktop\machinelearninginaction\Ch02/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    personData = array([percentTime, flyMile, iceCream])
    classifyResult = classify0(personData / ranges, normMat, datingLabels, 3)
    print "the answer is ", resultList[classifyResult - 1]


def img2Vector(fileName):
    returnVec = zeros((1, 1024))
    img = open(fileName)
    for i in range(32):
        lineData = img.readline()
        for j in range(32):
            returnVec[0, i * 32 + j] = lineData[j]
    return returnVec

def handwrittingClassTest():
    dir="E:\Desktop\machinelearninginaction\Ch02\digits/trainingDigits/"
    hwLabels=[]
    trainingListDir = listdir(dir)
    m = len(trainingListDir)
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileName = trainingListDir[i]
        fileStr = fileName.split(".")[0]
        fileClassLabel = int(fileName.split("_")[0])
        hwLabels.append(fileClassLabel)
        trainMat[i,:] = img2Vector(dir + fileName)
    testListDir = listdir("E:\Desktop\machinelearninginaction\Ch02\digits/testDigits/")
    errorCount=0.0
    testNum = len(testListDir)
    for i in range(testNum):
        testName = testListDir[i]
        fileStr = testName.split(".")[0]
        testSmapleClass = int(testName.split("_")[0])
        testVec = img2Vector("E:\Desktop\machinelearninginaction\Ch02\digits/testDigits/" + testName)
        testResult = classify0(testVec, trainMat, hwLabels, 3)
        print "classifier--> %d || true label--> %d " % (testResult, testSmapleClass)
        if testResult!=testSmapleClass: errorCount+=1

    print  "total number error is %d ---- and error Ratio is ---%f" % (errorCount,errorCount/testNum)