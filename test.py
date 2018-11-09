# coding=utf-8
import kNN
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import trees
import treePlotter
import bayes
import feedparser

# group,labels=ML.creatDataset()

# ML.classify0([0,0],group,labels,3)

reload(bayes)
# datingDataMat,datingLabels = ML.file2Matrix("E:/Desktop/ML/datingTestSet.txt")
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,0],15.0 * array(datingLabels),15.0 * array(datingLabels))
# plt.show()

# normMat,ranges,minVals = ML.autoNorm(datingDataMat)

# print normMat, ranges,minVals

# ML.datingClassTest()

# ML.classsifyPerson()

# vector = ML.img2Vector("E:\Desktop\machinelearninginaction\Ch02\digits/trainingDigits/0_0.txt")
# print vector

# kNN.handwrittingClassTest()
# ----------------------chapter 1-----------------------
listOPosts, labels = trees.createDataSet()
# print listOPosts
# shannon_ent = trees.calcShannonEnt(listOPosts)
# print  shannon_ent
# listOPosts[0][-1]="maybe"
# print trees.calcShannonEnt(listOPosts)
# print trees.splitDataSet(listOPosts, 1, 1)
# print trees.chooseBestFeatureToSplit(listOPosts)
# print trees.createTree(listOPosts, labels)
# # treePlotter.createPlot()
# reload(treePlotter)
# treePlotter.retriveTree(1)
# mytree = treePlotter.retriveTree(0)
# trees.storeTree(mytree, "myTree.txt")
# tree=trees.grabTree("myTree.txt")
# print treePlotter.getNumLeafs(mytree)
# print treePlotter.getTreeDepth(mytree)
# treePlotter.createPlot(mytree)
# trees.classify(mytree, labels, [1, 0])
# print trees.classify(tree, labels, [1, 0])

# fr = open("E:\Desktop\machinelearninginaction\Ch03/lenses.txt")
# lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# lenth = len(lenses)
# for i in range(lenth):
#     print lenses[i]
# lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
# tree = trees.createTree(lenses, lensesLabels)
# print tree
# treePlotter.createPlot(tree)
listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
# print myVocabList
# print listOPosts
vec = bayes.setOfWord2Vec(myVocabList, listOPosts[0])
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWord2Vec(myVocabList, postinDoc))
p0Vect, p1Vect, pAbusive = bayes.trainNB0(trainMat, listClasses)
print p0Vect
print p1Vect
print pAbusive
# bayes.testingNB()

# bayes.spamTest()
# ny = feedparser.parse("http://feeds.bbci.co.uk/zhongwen/simp/rss.xml")
# sf = feedparser.parse("http://www.voachinese.com/api/")
# vocabList, pSF, pNY = bayes.localWords(ny, sf)
# print ny['entries']
# print len(ny['entries'])
# print len(sf['entries'])
# bayes.getTopWords(ny, sf)
