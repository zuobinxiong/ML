import svmMLiA

dataMat, labelMat = svmMLiA.loadDataSet("E:\Desktop\machinelearninginaction\Ch06/testSet.txt")
print dataMat
print labelMat

b, alphas = svmMLiA.smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
print b
print alphas[alphas > 0]

for i in range(100):
    if alphas[i] > 0.0: print dataMat[i], labelMat[i]
