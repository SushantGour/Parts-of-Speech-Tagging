# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:04:08 2020

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



test=open("FinalTest.txt","r",errors="ignore")
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
        
print("The 10 Most Occured Words Are:")


sorted_word_frequency = sorted(word_dictionary.items(), key=lambda kv: kv[1], reverse=True)
for i in range(10):
    print(sorted_word_frequency[i])
        
print("\n")   

print("The 10 Most Occured Tags Are:")

sorted_tag_frequency = sorted(tag_dictionary.items(), key=lambda kv: kv[1], reverse=True)
for i in range(10):
    print(sorted_tag_frequency[i])



print("\n")   

print("The 10 Most Occured Word-Tag Pairs Are:")



sorted_word_tag_frequency = sorted(word_tag_dictionary.items(), key=lambda kv: kv[1], reverse=True)
for i in range(10):
    print(sorted_word_tag_frequency[i])   
    
    
