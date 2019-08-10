# SemanticAnalysis
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

*** More information in the PDF (this was a stats/data analysis homework assignment) 
