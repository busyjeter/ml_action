# import numpy as np 
from numpy import *
import operator
from os import listdir

def createDataSet():
	group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group, labels

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0] #dataSet行数
	diffMat=tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat=diffMat**2
	sqDistance = sqDiffMat.sum(axis=1)#每行求和
	distance=sqDistance**0.5
	sortedDistIndices = distance.argsort()#argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
		sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	with open(filename) as fr:
	# fr=open(filename)
		arrayOfLines = fr.readlines()
		numberOfLines=len(arrayOfLines)
		returnMat=zeros((numberOfLines,3))
		classLabelVector=[]
		index=0
		for line in arrayOfLines:
			line=line.strip()
			listFromLine = line.split("\t")
			returnMat[index,:]=listFromLine[0:3]
			classLabelVector.append(int(listFromLine[-1]))
			index+=1
		return returnMat,classLabelVector

def autoNorm(dataSet):
	minValue=dataSet.min(0)
	maxValue=dataSet.max(0)
	ranges=maxValue-minValue
	normDataSet=zeros(shape(dataSet))
	c=dataSet.shape[0]
	normDataSet=dataSet-tile(minValue,(c,1))
	normDataSet=normDataSet/tile(ranges,(c,1))
	return normDataSet,ranges,minValue

def datingClassTest():
	hoRatio = 0.1
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minValue=autoNorm(datingDataMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRatio)
	errorCount=0.0
	for i in range(numTestVecs):
		classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],
			datingLabels[numTestVecs:m],3)
		print ("the classifier came back with :%d, the real answer is %d" 
			%(classifierResult,datingLabels[i]))
		if classifierResult != datingLabels[i]:
			errorCount+=1.0
	print ("total error rate is :%f" %(errorCount/float(numTestVecs)))

def img2vector(filename):
	returnVect=zeros((1,1024))
	with open(filename) as fr:
		for i in range(32):
			lineStr=fr.readline()
			for j in range(32):
				returnVect[0,32*i+j]=int(lineStr[j])
	return returnVect


def handwritingClassTest():
	hwLabels=[]
	trainingFileList=listdir("trainingDigits")
	m=len(trainingFileList)
	trainingMat=zeros((m,1024))
	for i in range(m):
		fileNameStr=trainingFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split("_")[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:]=img2vector('trainingDigits/%s' %fileNameStr)

	testFileList=listdir('testDigits')
	errorCount=0.0
	mTest=len(testFileList)
	for i in range(mTest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split("_")[0])
		vectorUnderTest=img2vector('testDigits/%s' %fileNameStr)
		classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print ("the classifier came back with :%d, the real answer is %d" 
			%(classifierResult,classNumStr))
		if classifierResult != classNumStr:
			errorCount+=1.0
	print ("# of errort is %d,total error rate is :%f" %(errorCount,errorCount/float(mTest)))










