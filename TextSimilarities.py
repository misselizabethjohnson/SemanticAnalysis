#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:41:44 2019

@author: elizabethjohnson

This script can calculate degrees of similarity between books (useful for predicting
shared authorship) by: 
    1) counting the sum of each word in the English language within the book 
    2) creating a vector with total dimensions equal to the total number of words
    in English and value within each of those dimensions = to the number of times
    that word is used in the book
    3) taking the dot product of that vector with another book's vector...
    identical books will have dot product = 1 and completely dissimilar books would
    have dot products = 0
    
Note that there are a lot of commonly used words in English like 'the', 'and', 
'a', etc. These are called 'stop words'. By elliminating these words, we can get 
a better measure of authorship because the books will be compared based on words 
with more substance. 

Of course, books with similar content use similar words. But another strategy this
script employs is to examine authorship based on shared sentences. Perhaps one 
author writes two books on completely different subjects, but has a similar writing
style such that certain phrases or small sentences appear often. 

This script can also simply plot Zipf's Law for all the books. Or just the number
of letters in each book. 

At the very end, the script uses a multidimensional scaling technique to plot a 
sort of similarity map where you can see which books are most alike. 

The script times itself and reports total computing time. 
Additional functionality welcomed. 
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn.manifold import MDS
import time 
start  = time.time()

## import file of all English words 
words_file = np.loadtxt('words_file.txt', dtype=np.str)
word_list = list(words_file)

### OPEN THE BOOK FILES
### get rid of puctuation attached to a word in a string
def open_book(filename):
    with open(filename) as f1: 
        text = [word for line in f1 for word in line.split()]
    text = [string.replace('.|,|!|;|:|"|?|_|/','').lower() for string in text]
    return text;

text1 = open_book('janeausten.txt')  
text2 = open_book('emma.txt')
text3 = open_book('senseandsensibility.txt')
text4 = open_book('waterfowlidentification.txt')
text5 = open_book('completecheese.txt')
text6 = open_book('flatland.txt')
text7 = open_book('intimeofdefense.txt')
text8 = open_book('odyssey.txt') 
text9 = open_book('internet.txt') 

### OPEN THE BOOK BUT SPLIT INTO SENTENCES
# -*- coding: utf-8 -*-
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

with open("janeausten.txt","r") as myfile: 
    b1=myfile.read().replace('\n','')
    
with open("emma.txt","r") as myfile: 
    b2=myfile.read().replace('\n','') 
    
with open("senseandsensibility.txt","r") as myfile: 
    b3=myfile.read().replace('\n','') 

with open("waterfowlidentification.txt","r") as myfile: 
    b4=myfile.read().replace('\n','')   

with open("completecheese.txt","r") as myfile: 
    b5=myfile.read().replace('\n','')   

with open("flatland.txt","r") as myfile: 
    b6=myfile.read().replace('\n','')   

with open("intimeofdefense.txt","r") as myfile: 
    b7=myfile.read().replace('\n','')   

with open("odyssey.txt","r") as myfile: 
    b8=myfile.read().replace('\n','')   

with open("internet.txt","r") as myfile: 
    b9=myfile.read().replace('\n','')       

def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        commonalities = a_set & b_set
    else: 
        commonalities = {0}
    return commonalities;


for i in range(0,9):
    globals()['s%s' % str(i+1)] = split_into_sentences(eval('b%s'% str(i+1)))



####### CALCULATE SIMILAR SENTENCES 
def GenerateSentenceCorrelationAndPlot():
    sentence_array = np.zeros([9,9])
    
    for x in range(0,9):
        for y in range(0,9): 
            if x == y:
                sentence_array[x][y] = np.nan
            else: 
                try: 
                    eval('s%s' % str(x+1)).replace('2.'|'3.'|'4.'|'5.'|'6.'|'7.'|'8.'|'9.'|'10.','')
                    eval('s%s' % str(y+1)).replace('2.'|'3.'|'4.'|'5.'|'6.'|'7.'|'8.'|'9.'|'10.','')
                except:
                    pass
                num_similar = (common_member(eval('s%s' % str(x+1)), eval('s%s' % str(y+1))))
            
                countx = [eval('s%s' % str(x+1)).count(i) for i in num_similar]
                county = [eval('s%s' % str(y+1)).count(i) for i in num_similar]
                 
                sentence_array[x][y] = sum(countx) + sum(county)

    labels=['Pride&Pred','Emma','Sense&Sense','DuckId','Cheese',
            'Flatland','TimeofEmergency','Odyssey','Internet']
    plt.imshow(sentence_array,cmap='Oranges')
    plt.xticks(np.arange(0,9,1),labels,rotation=90)
    plt.yticks(np.arange(0,9,1),labels)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


## function for making the vectors 
def create_vector(text, dictionary):
    
    uniq = np.unique(text, return_counts=True)
    
    for i in range(0, len(uniq[0])):
        
        if uniq[0][i] in dictionary.keys(): 
            dictionary[uniq[0][i]] = uniq[1][i]
        else: 
            continue
        
    vect = np.fromiter(dictionary.values(), dtype=float)
    return vect;


### split up all the letters
def get_letters(filename):
    with open(filename) as f1: 
        letter = [letter for line in f1 for word in line.split() for letter in word]
            
    letter = [string.replace('.|,|!|;|:|"|?|_|/|0|1|2|3|4|5|6|7|8|9','').lower() for string in letter]
    return letter;

def count_letters(letters_list):
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
                'q','r','s','t','u','v','w','x','y','z']
    for ii in alphabet: 
        globals()['%s' % ii] = letters_list.count('%s' % ii)
       
    
    hist_vals = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]
    hist_labels = alphabet 
    
    return hist_vals, hist_labels; 


### Plot the Sum of Each Letter in Each Book 
def PlotCountedLettersInEachBook():
    xs = list(np.arange(0,26,1))  

    ## get the letters in each book 
    letters1 = get_letters('janeausten.txt') 
    letters2 = get_letters('emma.txt')
    letters3 = get_letters('senseandsensibility.txt')
    letters4 = get_letters('waterfowlidentification.txt')
    letters5 = get_letters('completecheese.txt')
    letters6 = get_letters('flatland.txt')
    letters7 = get_letters('intimeofdefense.txt') 
    letters8 = get_letters('odyssey.txt') 
    letters9 = get_letters('internet.txt') 
      

    plt.bar(xs,count_letters(letters2)[0],width=0.1,align='edge',alpha=0.8,color='orangered',label='Emma')
    plt.bar([i+0.1 for i in xs],count_letters(letters8)[0],width=0.1,align='edge',alpha=0.8,color='darkorange',label='Odyssey')
    plt.bar([i+0.2 for i in xs],count_letters(letters1)[0],width=0.1,align='edge',alpha=0.8,color='gold',label='Pride&Pred')
    plt.bar([i+0.3 for i in xs],count_letters(letters3)[0],width=0.1,align='edge',alpha=0.8,color='greenyellow',label='Sense&Sense')
    plt.bar([i+0.4 for i in xs],count_letters(letters5)[0],width=0.1,align='edge',alpha=0.8,color='cyan',label='Cheese')
    plt.bar([i+0.5 for i in xs],count_letters(letters9)[0],width=0.1,align='edge',alpha=0.8,color='skyblue',label='Internet')
    plt.bar([i+0.6 for i in xs],count_letters(letters6)[0],width=0.1,align='edge',alpha=0.8,color='slateblue',label='Flatland')
    plt.bar([i+0.7 for i in xs],count_letters(letters7)[0],width=0.1,align='edge',alpha=0.8,color='mediumorchid',label='TimeofEmergency')
    plt.bar([i+0.8 for i in xs],count_letters(letters4)[0],width=0.1,align='edge',alpha=0.8,color='violet',label='DuckID')
    plt.xticks(xs, count_letters(letters1)[1])
    plt.legend()
    plt.ylabel('Number of Occurrences')
    plt.xlabel('Letter')
    plt.show()


#### letter-based entropy of each book: 

def compute_letter_entropy(letters_list):
    
    count_them = count_letters(letters_list)[0]
    
    max_num_of_letters = np.amax(count_letters(letters_list)[0])
    min_num_of_letters = np.amin(count_letters(letters_list)[0])
    
    probabilities = []
    for i in count_letters(letters_list)[0]:
        probabilities.append((i - min_num_of_letters)/(max_num_of_letters - min_num_of_letters))
   
    H = -sum([count_them[i]*probabilities[i]*np.log(probabilities[i]) for i in range(0, 
              len(probabilities)) if probabilities[i] > 0])/(sum(count_them))
    
    return H;

### Compute the letter-based entropy in each of the books
def ComputeLetterEntropyAndPlot():
    ent1 = compute_letter_entropy(letters1)
    ent2 = compute_letter_entropy(letters2)  
    ent3 = compute_letter_entropy(letters3)
    ent4 = compute_letter_entropy(letters4)
    ent5 = compute_letter_entropy(letters5)
    ent6 = compute_letter_entropy(letters6)
    ent7 = compute_letter_entropy(letters7)
    ent8 = compute_letter_entropy(letters8)
    ent9 = compute_letter_entropy(letters9)

    ents_x = np.arange(0,9,1)
    plt.bar(ents_x,[ent1,ent2,ent3,ent4,ent5,ent6,ent7,ent8,ent9],color='orange')
    plt.xticks(ents_x,['Pride&Pred','Emma','Sense&Sense','DuckId','Cheese',
            'Flatland','TimeofEmergency','Odyssey','Internet'],rotation=90)
    plt.ylabel('Total Entropy / Total Letters')        
    plt.ylim(0.26,0.29)
    plt.tight_layout()
    plt.show()


### Calculates level of similarity between the books with and without stop words (depending on which you use)
def GenerateCorrelationMatrixAndPlot():
    new_word_list = word_list
    stopwords = list(np.loadtxt('stopwords.txt', dtype=np.str))

    for i in stopwords: 
        try: 
            new_word_list.remove(i)
        except: 
            continue

    ## initialize dictionaries of all English words
    dict1 = {i : 0 for i in word_list}
    dict2 = {i : 0 for i in word_list}
    dict3 = {i : 0 for i in word_list}
    dict4 = {i : 0 for i in word_list}
    dict5 = {i : 0 for i in word_list}
    dict6 = {i : 0 for i in word_list}
    dict7 = {i : 0 for i in word_list}
    dict8 = {i : 0 for i in word_list}
    dict9 = {i : 0 for i in word_list}

    dictall = {i : 0 for i in word_list}

    ## initialize new dictionaries of English words without stop words
    newdict1 = {i : 0 for i in new_word_list}
    newdict2 = {i : 0 for i in new_word_list}
    newdict3 = {i : 0 for i in new_word_list}
    newdict4 = {i : 0 for i in new_word_list}
    newdict5 = {i : 0 for i in new_word_list}
    newdict6 = {i : 0 for i in new_word_list}
    newdict7 = {i : 0 for i in new_word_list}
    newdict8 = {i : 0 for i in new_word_list}
    newdict9 = {i : 0 for i in new_word_list}

    ## create vectors 
    vect1 = create_vector(text1,dict1)
    vect2 = create_vector(text2,dict2)
    vect3 = create_vector(text3,dict3)
    vect4 = create_vector(text4,dict4)
    vect5 = create_vector(text5,dict5)
    vect6 = create_vector(text6,dict6)
    vect7 = create_vector(text7,dict7)
    vect8 = create_vector(text8,dict8)
    vect9 = create_vector(text9,dict9)

    vectall = create_vector(text1+text2+text3+text4+text5+text6+text7+text8+text9,dictall)

    ## create new vectors 
    newvect1 = create_vector(text1,newdict1)
    newvect2 = create_vector(text2,newdict2)
    newvect3 = create_vector(text3,newdict3)
    newvect4 = create_vector(text4,newdict4)
    newvect5 = create_vector(text5,newdict5)
    newvect6 = create_vector(text6,newdict6)
    newvect7 = create_vector(text7,newdict7)
    newvect8 = create_vector(text8,newdict8)
    newvect9 = create_vector(text9,newdict9)


    ### create the 9x9 array 
    array_similarities = np.zeros([9,9])

    ### goes through and evaluates the dot product for each pair of vectors 
    ### right now this is using the WITHOUT stop words set... you can change it to 
    ### vect instead of new vect if you want with stop words 
    for x in range(0,9):
        for y in range(0,9):
            value = np.dot(eval('newvect%s'% str(x+1)), eval('newvect%s'% str(y+1)))
            array_similarities[x][y] = value
        
    minimum = np.amin(array_similarities)
    maximum = np.amax(array_similarities) 
    
    ### normalizes    
    for i in np.nditer(array_similarities, op_flags=['readwrite']):
        i[...] = (i - minimum)/(maximum - minimum) 


    plt.imshow(array_similarities,cmap='Oranges')
    labels=['Pride&Pred','Emma','Sense&Sense','DuckId','Cheese',
            'Flatland','TimeofEmergency','Odyssey','Internet']  
    plt.xticks(np.arange(0,9,1),labels,rotation=90)  
    plt.yticks(np.arange(0,9,1),labels)   
    plt.colorbar()
    plt.tight_layout()
    plt.show()


## Zipf's Law
def PlotZipfsLawWordsAll(individual=False):
    sort1 = np.sort(vect1)
    sort2 = np.sort(vect2)
    sort3 = np.sort(vect3)
    sort4 = np.sort(vect4)
    sort5 = np.sort(vect5)
    sort6 = np.sort(vect6)
    sort7 = np.sort(vect7)
    sort8 = np.sort(vect8)
    sort9 = np.sort(vect9)
    eachSortedBook = [sort1,sort2,sort3,sort4,sort5,sort6,sort7,sort8,sort9]
    
    sortall = np.sort(vectall)
    
    if individual == False:
        plt.plot(sortall[::-1],'.',color='orange',label='all 9 books')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Log Ordered Words')
        plt.ylabel('Log # Occurrences')
        plt.show()
    else:
        for i in eachSortedBook:
            plt.plot(i,'.')
        #plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Log Ordered Words')
        plt.ylabel('Log # Occurrences')
        plt.show()


### Multidimensional Scaling Plot
def createMDSPlot():
    labels=['Pride&Pred','Emma','Sense&Sense','DuckId','Cheese',
            'Flatland','TimeofEmergency','Odyssey','Internet']

    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(array_similarities)

    fig, ax = plt.subplots()
    ax.scatter(X_transformed[:,0], X_transformed[:,1])
    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (X_transformed[:,0][i], X_transformed[:,1][i]))
        
    plt.tight_layout()
    


end = time.time()
print('Computing time (sec): ', end - start)

