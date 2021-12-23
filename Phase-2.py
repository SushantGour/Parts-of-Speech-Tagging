# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:05:14 2020

@author: Sushant Gour
"""



import collections
#import pandas and matplotlib.pyplot to plot confusion
import pandas as pd
import matplotlib.pyplot as plt
import os
#import numpy to play with arrays easily
import numpy as np
import itertools



train=open("FinalTrain.txt","r",errors="ignore")

word_dictionary=dict()
tag_dictionary=dict()
word_tag_dictionary=dict()



test=open("concatenatedTest.txt","r",errors="ignore")
predictedValues=[]
trueValues=[]
count=0



for line in train:
    line=line.strip()
    word,tag=line.split("_")
    
    if word in word_dictionary:
        word_dictionary[word]=word_dictionary[word]+1
        
    else:
        word_dictionary[word]=1
        
    if tag in tag_dictionary:
        tag_dictionary[tag]=tag_dictionary[tag]+1
        
    else:
        tag_dictionary[tag]=1
        
    if line in word_tag_dictionary:
        word_tag_dictionary[line]=word_tag_dictionary[line]+1
        
    else:
        word_tag_dictionary[line]=1
        

    
words=word_dictionary.keys()
tags=tag_dictionary.keys()



word_maximum_tag = dict()

for word in words:

    tagmax = "EmptyString";
    
    scoremax = 0;
    for tag in tags:

        word_tag = str(word) + "_" + str(tag)

        if word_tag in word_tag_dictionary.keys():
            score = word_tag_dictionary[word_tag]

        else:
            score = 0

        if score > scoremax:
            tagmax = tag
            scoremax=score

    word_maximum_tag[word] = tagmax


def most_probable_tag_model(word):
    if word in word_maximum_tag.keys():
        return word_maximum_tag[word]

    else:
        return max(tag_dictionary, key=tag_dictionary.get)

#print(most_probable_tag_model("the"))




for line in test:
    line = line.strip()
    word, tag = line.split("_")

    predictedValues.append(most_probable_tag_model(word))
    trueValues.append(tag)



#Calculating accuracy over test corpus
print("Accuracy=" + str(100 * ((sum(np.array(predictedValues) == np.array(trueValues))) / len(np.array(trueValues)))) + "%")
#print((sum(np.array(predictedValues) == np.array(trueValues))))
#print(len(np.array(trueValues)))


#Plotting the confusion matrix
tag_actu = pd.Series(trueValues, name='Actual')
tag_pred = pd.Series(predictedValues, name='Predicted')
confusionMatrix = pd.crosstab(tag_actu, tag_pred, rownames=['Actual Tags'], colnames=['Predicted Tags'], margins=True)
print("\nConfusion Matrix:\n")
print(confusionMatrix)



def confusion_matrix_plotting_function(confusionMatrix,title='Confusion Matrix',cmap=plt.cm.plasma):
    plt.matshow(confusionMatrix,cmap=cmap)
    plt.colorbar()
    tick_marks=np.arange(len(confusionMatrix.columns)) 
    plt.xticks(tick_marks,confusionMatrix.columns,rotation=45)
    plt.yticks(tick_marks,confusionMatrix.index)
    plt.ylabel(confusionMatrix.index.name)
    plt.xlabel(confusionMatrix.columns.name)
    plt.show()



confusion_matrix_plotting_function(confusionMatrix)







train.close
test.close 











