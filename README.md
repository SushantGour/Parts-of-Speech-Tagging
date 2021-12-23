# Parts-of-Speech-Tagging
This repository is about the Parts of Speech Tagging project mentioned in my resume. It is a project related to AI. All the material related to the project i.e. the training data, test data, code etc. are present in this repository. The goal of the project is to assign parts of speech tags to a given sequence of input word. The project uses Hidden Markov Model and I implemented the Viterbi Algorithm to find the best probable sequence of tags for a given sequence of words.


The overall code is divided into three phases:
1) Phase-1 includes retriving the word and tag data from the training dataset and storing them in python dictionaries. It also includes arranging the words and tags in decreasing order of their no of occurences in the training dataset.
2) Phase 2 includes finding the best tag for a given word.
3) Phase 3 includes implementation of the Viterbi Algorithm to find the best probable sequence of tags for a given sequence of words.
