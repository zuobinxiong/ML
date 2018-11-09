# -*- coding: UTF-8 -*-
from numpy import *
import feedparser


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word %s is not in the VocabList" % (word)
    return returnVec


def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        # else:
        #     print "the word %s is not in the VocabList" % (word)
    return returnVec


def trainNB0(trainMartix, trainCategory):
    numTrainDoc = len(trainMartix)
    numWords = len(trainMartix[0])
    pAbusive = sum(trainCategory) / float(numTrainDoc)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDoc):
        if trainCategory[i] == 1:
            p1Num += trainMartix[i]
            p1Denom += sum(trainMartix[i])
        else:
            p0Num += trainMartix[i]
            p0Denom += sum(trainMartix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0Vect, p1Vect, pAbusive = trainNB0(array(trainMat), array(listClasses))
    testEntry = ["love", "my", "maltion"]
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive)
    testEntry = ["stupid", "garbage"]
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive)


def testParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = [];
    for i in range(1, 26):
        wordList = testParse(open('E:/Desktop/machinelearninginaction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)  # 添加新列表结尾
        fullText.extend(wordList)  # 并入列表
        classList.append(1)
        wordList = testParse(open('E:/Desktop/machinelearninginaction/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)  # 添加新列表结尾
        fullText.extend(wordList)  # 并入列表
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50);
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0Vect, p1Vect, pAbusive = trainNB0(array(trainMat), array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVec, p0Vect, p1Vect, pAbusive) != classList[docIndex]:
            errorCount += 1
    print 'errorRate is  ', float(errorCount) / len(testSet)


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
        sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:20]


def localWords(feed1, feed0):
    import feedparser
    docList = [];
    classList = [];
    fullText = [];
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = testParse(feed1['entries'][i]['summary_detail'])
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)
        wordList = testParse(feed0['entries'][i]['summary_detail'])
        docList.append(wordList)
        classList.append(0)
        fullText.extend(wordList)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen);
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2VecMN(vocabList, docList[docIndex]))
        trainClass.append(classList[randIndex])
    p0Vect, p1Vect, pAbusive = trainNB0(array(trainMat), array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVec = bagOfWord2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVec), p0Vect, p1Vect, pAbusive) != classList[docIndex]:
            errorCount += 1
    print 'errorRate is  ', float(errorCount) / len(testSet)
    return vocabList, p0Vect, p1Vect


def getTopWords(ny, sf):
    import operator
    vocabList, pSF, pNY = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(pSF)):
        if pSF[i] > -6.0: topSF.append(vocabList[i], pSF[i])
        if pNY[i] > -6.0: topNY.append(vocabList[i], pNY[i])
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SFSFSFSFSFFSFFSF"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "YNYNYNYNYNY"
    for item in sortedNY:
        print item[0]
