import csv 
import random 
import math 
import pandas as pd
import sys

def splitDataset(dataset, splitRatio): 
	trainSize = int(len(dataset) * splitRatio) 
	trainSet = [] 
	copy = list(dataset) 
	while len(trainSet) < trainSize: 
		index = random.randrange(len(copy)) 
		trainSet.append(copy.pop(index)) 
	return [trainSet, copy] 

def separateByClass(dataset): 
	separated = {} 
	for i in range(len(dataset)): 
		vector = dataset[i] 
		if (vector[-1] not in separated): 
			separated[vector[-1]] = [] 
		separated[vector[-1]].append(vector) 
	return separated 

def mean(numbers): 
	return sum(numbers)/float(len(numbers)) 

def stdev(numbers): 
	avg = mean(numbers) 
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1) 
	return math.sqrt(variance) 

def summarize(dataset): 
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)] 
	del summaries[-1] 
	return summaries 

def summarizeByClass(dataset): 
	separated = separateByClass(dataset) 
	summaries = {} 
	for classValue, instances in separated.iteritems(): 
		summaries[classValue] = summarize(instances) 
	return summaries 

def calculateProbability(x, mean, stdev): 
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2)))) 
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent 

def calculateClassProbabilities(summaries, inputVector): 
	probabilities = {} 
	for classValue, classSummaries in summaries.iteritems(): 
		probabilities[classValue] = 1 
		for i in range(len(classSummaries)): 
			mean, stdev = classSummaries[i] 
			x = inputVector[i] 
			probabilities[classValue] *= calculateProbability(x, mean, stdev) 
	return probabilities 

def predict(summaries, inputVector): 
	probabilities = calculateClassProbabilities(summaries, inputVector) 
	bestLabel, bestProb = None, -1 
	for classValue, probability in probabilities.iteritems(): 
		if bestLabel is None or probability > bestProb: 
			bestProb = probability 
			bestLabel = classValue 
	return bestLabel ,probabilities
def getPredictions(summaries, testSet): 
	predictions = [] 
	prob_class=[]
	for i in range(len(testSet)): 
		result, prob = predict(summaries, testSet[i]) 
		predictions.append(result) 
		prob_class.append(prob)
	return predictions , prob_class
def getAccuracy(testSet, predictions): 
	correct = 0 
	for i in range(len(testSet)): 
		if testSet[i][-1] == predictions[i]: 
			correct += 1 
	return (correct/float(len(testSet))) * 100.0 

def main(): 
	splitRatio = 0.67 
	x_train = pd.read_csv(str(sys.argv[1]), header =None)
	y_train = pd.read_csv(str(sys.argv[2]), header =None)
	x_test = pd.read_csv(str(sys.argv[3]), header =None)
	temp = [x_train[i].append(y_train[i][0]) for i in range(len(x_train)) ]
	trainingSet, testSet = splitDataset(x_train, splitRatio) 
	summaries = summarizeByClass(trainingSet) 
	predictions,prob_class = getPredictions(summaries, testSet) 
	accuracy = getAccuracy(testSet, predictions) 
	print('Accuracy: {0}%').format(accuracy) 
	predictions2, prob_class2  = getPredictions(summaries, x_test) 
	
	class_Prob={}
	for i in range(len(prob_class2)):
		row=[]
		for j in range(len(prob_class2[i])):
			row.append(round(prob_class2[i][j]/sum(prob_class2[i].values()),2))
		class_Prob[i] = row
	output=pd.DataFrame.from_dict(class_Prob,orient='index')

	output.to_csv('probs_test.csv',index=False,header=None)


main()
