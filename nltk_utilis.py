import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stemm(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenised_sen,all_the_words):
    tokenised_sen=[stemm(w) for w in tokenised_sen]
    
    bag=np.zeros(len(all_the_words),dtype=np.float32)
    for idx,w, in enumerate((all_the_words)):
        if w in tokenised_sen:
            bag[idx]=1.0

    return bag      

#testing bag of words
# sentence=["hello","how","are","you"]
# words=["hi","hello","I","you","bye","thank","cool"]
#bag=[0,1,0,1,0,0,0]


# ans=bag=bag_of_words(sentence,words)
# print(ans)

#getting the right output so good

#test tokenize

# a="How long does shipping take?"
# print(a)
# a=tokenize(a)
# print(a)

#test stemmer
# words=['Organize','Organizes','organizing']
# stemmed_words=[stemm(w) for w in words]
# print(stemmed_words)



#def bag_of_words(tokenised_sen,all_words):
