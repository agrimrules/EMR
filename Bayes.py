__author__ = 'Agrim Asthana'
#Load the dataset
import csv


def loadcsv(filename):
    lines=csv.reader(open(filename,"rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

# Split the dataset

import random



def splitDataset(data, splitratio):
    trainSize=int(len(dataset) * splitratio)
    trainSet=[]
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

# Separating the data by class

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if(vector[-1] not in separated):
            separated[vector[-1]]=[]
        separated[vector[-1]].append(vector)
    return separated



# Calculating the Mean Value
import math


def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg= mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries=[(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

dataset = [[1,20,0],[2,21,1]]

