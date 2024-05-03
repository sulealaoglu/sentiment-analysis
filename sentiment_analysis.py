# import libraries
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import re
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import nltk #stop wordlerden kurtulmak için kullanıldı.
nltk.download('stopwords')
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import nltk
import matplotlib.pyplot as plt
import matplotlib
plt.style.use(['seaborn-notebook'])
from sklearn.decomposition import _pca
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from spacy.lang.tr import Turkish
import jpype
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from TurkishStemmer import TurkishStemmer

def readExcel():
    df = pd.read_excel("./ANEW_Turkish.xlsx")
    
    for index, row in df.iterrows():
        turkish = row['TR']
        v = row['V']
        a = row['A']
        d = row['D']
        word_list[turkish] = {'V': v, 'A': a, 'D': d}

def getWordWeights():
    stn = pd.read_excel("./ANEW_Turkish.xlsx")
    for words in stn.index:
        if words in stn.index:
            if words is np.nan:
                continue
            for word in words.split(','):
                final_stn[word.strip()] = {'pos':stn.loc[words]['pos value'],'neg' : stn.loc[words]['neg value'] }

def Lemmatization(sentence):
    analysis: java.util.ArrayList = (morphology.analyzeAndDisambiguate(sentence).bestAnalysis())
    token = sentence.split() #Tokenization yapÄ±lÄ±r.
    pos=[]
    for index, i in enumerate(analysis):   
        if str(i.getLemmas()[0])=="UNK": #Kelime kÃ¶kÃ¼nÃ¼n bulamamasÄ± durumu.
            pos.append(token[index]) 
        else:
            pos.append(str(i.getLemmas()[0])) #Kelime kÃ¶kÃ¼ listeye eklenir.
        #print("lemma:")
        #print(str(i.getLemmas()[0]))  
    return pos 

def spellChecker(tokens):
    
    for index,token in enumerate(tokens):    
         
        #yazÄ±m yanlÄ±sÄ± varsa if'e girer
        if not spell_checker.check(JString(token)):
         
            if spell_checker.suggestForWord(JString(token)):
             
                  #kelimenin doÄŸru halini dÃ¶ndÃ¼rÃ¼r.
                  tokens[index] = spell_checker.suggestForWord(JString(token))[0]
                  #print((spell_checker.suggestForWord(JString(token))[0]))
     
    #Java liste yapÄ±sÄ± listeye eklenerek dÃ¼zeltilir. 
    corrected = [str(i) for i in tokens]
              
    return " ".join(corrected)

def feature_extraction(text):
    pos_val = 0
    neg_val = 0
    for token in text:
        word = token.lower()
        if word in word_list:
            pos_val += word_list[word]['V']
            neg_val +=  word_list[word]['A']
    print ('Positive Weight: ', pos_val , 'Negative Weight : ', neg_val)
    if pos_val > neg_val :
        return 'Positive'
    elif pos_val == neg_val :
        return 'Neutral'
    else:
        return 'Negative'



def remove_stopword(tokens): 
    filtered_tokens = [token for token in tokens if token not in stop_word_list]#stop word'lerden temizlenir.  
    return filtered_tokens
    
def sentiment_analysis(sentence):
    filtered = remove_stopword(sentence.split())
    corrected = spellChecker(filtered)
    stems = Lemmatization(corrected)
    return stems

#kelime ağırlıklarının tutulduğu liste - readExcel() fonksiyonu ile dolduruluyor
word_list = {}
#stop wordler için
stop_word_list = []
#nltk kütüphanesinden turkce dili icin stop wordleri al ve olmayanları kendin de ekle
stop_word_list = nltk.corpus.stopwords.words('turkish')    
stop_word_list.extend(["bir","kadar","sonra","kere","mi","ye","te","ta","nun","daki","nın","ten"])

#zemberek kütüphanesini tanımla , java virtual machine ile çalıştır
ZEMBEREK_PATH ='./zemberek-full.jar' 
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults() 
TurkishSpellChecker: JClass = JClass('zemberek.normalization.TurkishSpellChecker')
spell_checker: TurkishSpellChecker  = TurkishSpellChecker(morphology) 


readExcel()
example = 'senden nefret etmiyorum'
stems = sentiment_analysis(example)
print(stems)
print(feature_extraction(stems))