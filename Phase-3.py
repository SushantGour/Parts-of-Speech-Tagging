# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:07:35 2020

@author: Sushant Gour
"""




# Importing libraries

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt 

import pprint, time
 



#open train and test text files
train=open("concatenatedTrain.txt",encoding="utf8")
test=open("AN0.txt",encoding="utf8")

# create list of train and test tagged words
train_tagged_words = []        
for line in train:
    line=line.strip()
    
    word,tag= line.split("_")
    train_tagged_words.append((word,tag))
    
    

test_tagged_words = []
test_untagged_words=[]
for line in test:
    line=line.strip()
    
    word,tag= line.split("_")
    test_tagged_words.append((word,tag))
    test_untagged_words.append(word)
    
    

#print(len(train_tagged_words))
#print(len(test_tagged_words))



#use set datatype to check how many unique tags are present in training data
tags = {tag for word,tag in train_tagged_words}
#print(len(tags))
#print(tags)
 
# check total words in train-set
TotalUniqueWords = {word for word,tag in train_tagged_words}

#print(len(TotalUniqueWords)) 

# compute Emission Probability
def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
#now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
 
     
    return (count_w_given_tag, count_tag)



#print(word_given_tag("the","ART"))

# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)



#print(t2_given_t1("SUBST","SUBST"))

# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)
start=time.time()
tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
 
print(tags_matrix)
end=time.time()
difference=end-start
print("Time taken:",difference,"seconds")




# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
pd.display(tags_df)



def Viterbi(words, train_bag = train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
     
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = 0.001
            else:
                transition_p = tags_df.loc[state[-1], tag]
                 
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
             
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))


#print(len(test_untagged_words)) 
start=time.time()
tagged_seq = Viterbi(test_untagged_words)
end=time.time()
difference=end-start
print("Time taken:",difference,"seconds")
    
# accuracy
check = [i for i, j in zip(test_tagged_words, tagged_seq) if i == j] 
 
accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)


#Plotting the confusion matrix
trueValues=[tag for word,tag in test_tagged_words]
predictedValues=[state for word,state in tagged_seq]


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


 





