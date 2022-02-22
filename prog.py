# Requried Libraries
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

# Files locations
STOPWORDS_FILE = './StopWords_Generic.txt'
POSITIVEWORDS_FILE = './PositiveWords.txt'
NEGATIVEWORDS_FILE = './NegativeWords.txt'
UNCERTAINITY_DICT_FILE = './uncertainty_dictionary.txt'
CONSTRAINING_DICT_FILE = './constraining_dictionary.txt'
INPUT_FILE = './cik_list.xlsx'

# NETWORK CONFIGURATION -- HELPS NOT TO LIMIT TO THE THRESHOLD REQUESTS
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"}
PROXY={"http": "http://111.233.225.166:1234"}
BASE_URL='https://www.sec.gov/Archives/'

class EdgarAnalysis():

    def __init__(self):
        self.data=[]
        self.stopWordList=[]
        self.positiveWordList=[]
        self.negativeword=[]
        self.uncertainDictionary=[]
        self.constrainDictionary=[]
        self.output=None
        
    def remove_tags(self,html):
        soup = BeautifulSoup(html, "html.parser")
        return soup.text

    def rawdata_extract(self, inpFile):
        inpFile = pd.read_excel(inpFile)
        print("Scraping Process Started ..")
        pbar = tqdm(total=100,colour="green")
        for index, row in tqdm(inpFile.iterrows()):
            cik=row['CIK']
            coname=row['CONAME']
            fyrmo=row['FYRMO']
            fdate = row['FDATE']
            form = row['FORM']
            secfname=row['SECFNAME']

            res_dict = dict()
            res_dict['CIK'] = cik
            res_dict['CONAME'] = coname
            res_dict['FYRMO'] = fyrmo
            res_dict['FDATE'] = fdate
            res_dict['FORM'] = form
            res_dict['SECFNAME'] = secfname
            resp = requests.get(BASE_URL+secfname, proxies=PROXY,headers=HEADERS)
            content = resp.text
            res_dict['CONTENT']=self.remove_tags(content)
            self.data.append(res_dict)
            pbar.update(100/153)
            # if int((index+1)*100/153)%10==0 or int((index+1)*100/153)%10==5:
            #     print("Extracting : ",int((index+1)*100/153)," % Done ...")
        pbar.close()
        print("Scraping Process Done")
    def get_data(self):
        return self.data

    def tokenize_(self,text):
        text = text.lower()
        tokens =nltk.word_tokenize(text)
        filtered_words = list(filter(lambda token: token not in self.stopWordList, tokens))
        filtered_words=[ word for word in filtered_words if word.isalpha()]
        return filtered_words

    def load_stopwords(self):
        with open(STOPWORDS_FILE ,'r') as stop_words:
            stopWords = stop_words.read().lower()
        self.stopWordList = stopWords.split('\n')

    def load_positiveWordList(self):
        with open(POSITIVEWORDS_FILE,'r') as posfile:
            positivewords=posfile.read().lower()
        self.positiveWordList=positivewords.split('\n')
    
    def load_negativeWordList(self):
        with open(NEGATIVEWORDS_FILE ,'r') as negfile:
            negativeword=negfile.read().lower()
        self.negativeWordList=negativeword.split('\n')
    
    def positive_score(self,text):
        numPosWords = 0
        rawToken = self.tokenize_(text)
        for word in rawToken:
            if word in self.positiveWordList:
                numPosWords  += 1
        return numPosWords
    
    def negative_word(self,text):
        numNegWords=0
        rawToken = self.tokenize_(text)
        for word in rawToken:
            if word in self.negativeWordList:
                numNegWords -=1
        sumNeg = numNegWords 
        sumNeg = sumNeg * -1
        return sumNeg
    
    def polarity_score(self,positiveScore, negativeScore):
        pol_score = (positiveScore - negativeScore) / ((positiveScore + negativeScore) + 0.000001)
        return pol_score
    
    def average_sentence_length(self,text):
        sentence_list = [sent for sent in sent_tokenize(text) if len(sent.replace('\t',' '))>2]

        def f(text):
            chars = ['\\','`','*','_','{','}','[',']','(',')','>','#','+','-','.','!','$','\'',"=","  ","   ","    ","\n","\t"]
            for c in chars:
                text = text.replace(c, " ")
            return text
        
        sentence_list=[f(sent) for sent in sentence_list]
        sentence_list=[sent for sent in sentence_list if not sent.isnumeric()]
        tokens = self.tokenize_(text)
        totalWordCount = len(tokens)
        totalSentences = len(sentence_list)
        average_sent = 0
        if totalSentences != 0:
            average_sent = totalWordCount / totalSentences
        average_sent_length= average_sent
        return round(average_sent_length)
    
    def percentage_complex_word(self,text):
        tokens = self.tokenize_(text)
        complexWord = 0
        complex_word_percentage = 0
        for word in tokens:
            vowels=0
            if word.endswith(('es','ed')):
                pass
            else:
                for w in word:
                    if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                        vowels += 1
                if(vowels > 2):
                    complexWord += 1
        if len(tokens) != 0:
            complex_word_percentage = complexWord/len(tokens)
        return complex_word_percentage
    
    def fog_index(self,averageSentenceLength, percentageComplexWord):
        fogIndex = 0.4 * (averageSentenceLength + percentageComplexWord)
        return fogIndex
    
    def complex_word_count(self,text):
        tokens = self.tokenize_(text)
        complexWord = 0
        for word in tokens:
            vowels=0
            if word.endswith(('es','ed')):
                pass
            else:
                for w in word:
                    if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                        vowels += 1
                if(vowels > 2):
                    complexWord += 1
        return complexWord
    
    def total_word_count(self,text):
        tokens = self.tokenize_(text)
        return len(tokens)
    
    def load_uncertainDictionary(self):
        with open(UNCERTAINITY_DICT_FILE ,'r') as uncertain_dict:
            uncertainDict=uncertain_dict.read().lower()
        self.uncertainDictionary = uncertainDict.split('\n')
    
    def uncertainty_score(self,text):
        uncertainWordnum =0
        rawToken =self.tokenize_(text)
        for word in rawToken:
            if word in self.uncertainDictionary:
                uncertainWordnum +=1
        sumUncertainityScore = uncertainWordnum 
        return sumUncertainityScore
    
    def load_constrainDictionary(self):
        with open(CONSTRAINING_DICT_FILE ,'r') as constraining_dict:
            constrainDict=constraining_dict.read().lower()
        self.constrainDictionary = constrainDict.split('\n')
    
    def constraining_score(self,text):
        constrainWordnum =0
        rawToken = self.tokenize_(text)
        for word in rawToken:
            if word in self.constrainDictionary:
                constrainWordnum +=1
        sumConstrainScore = constrainWordnum 
        return sumConstrainScore
    
    def positive_word_prop(self,positiveScore,wordcount):
        positive_word_proportion = 0
        if wordcount !=0:
            positive_word_proportion = positiveScore / wordcount
        return positive_word_proportion
    
    def negative_word_prop(self,negativeScore,wordcount):
        negative_word_proportion = 0
        if wordcount !=0:
            negative_word_proportion = negativeScore / wordcount
        return negative_word_proportion
    
    def uncertain_word_prop(self,uncertainScore,wordcount):
        uncertain_word_proportion = 0
        if wordcount !=0:
            uncertain_word_proportion = uncertainScore / wordcount
        return uncertain_word_proportion
    
    def constraining_word_prop(self,constrainingScore,wordcount):
        constraining_word_proportion = 0
        if wordcount !=0:
            constraining_word_proportion = constrainingScore / wordcount
        return constraining_word_proportion
    
    def constrain_word_whole(self,text):
        constrainWordnumWhole =0
        rawToken = self.tokenize_(text)
        for word in rawToken:
            if word in self.constrainDictionary:
                constrainWordnumWhole +=1 
        return constrainWordnumWhole 
    
    def analyse(self):
        df = pd.DataFrame(self.data)
        print("Analysing...")
        pbar = tqdm(total=100,colour="#00e64d")

        df['positive_score'] = df.CONTENT.apply(self.positive_score)
        print("\nPositive Score Calculated")
        pbar.update(100/16)

        df['negative_score'] = df.CONTENT.apply(self.negative_word)
        print("\nNegative Score Calculated") 
        pbar.update(100/16)

        df['polarity_score'] = np.vectorize(self.polarity_score)(df['positive_score'],df['negative_score'])
        print("\npolarity Score Calculated") 
        pbar.update(100/16)
        
        df['average_sentence_length'] = df.CONTENT.apply(self.average_sentence_length)
        print("\naverage_sentence_length Score Calculated") 
        pbar.update(100/16)
        
        df['percentage_of_complex_words'] = df.CONTENT.apply(self.percentage_complex_word)
        print("\npercentage_of_complex_words Score Calculated") 
        pbar.update(100/16)
        
        df['fog_index'] = np.vectorize(self.fog_index)(df['average_sentence_length'],df['percentage_of_complex_words'])
        print("\nfog_index Calculated") 
        pbar.update(100/16)
        
        df['complex_word_count']= df.CONTENT.apply(self.complex_word_count)
        print("\ncomplex_word_count Calculated") 
        pbar.update(100/16)
        
        df['word_count'] = df.CONTENT.apply(self.total_word_count)
        print("\nword_count Calculated") 
        pbar.update(100/16)
        
        df['uncertainty_score']=df.CONTENT.apply(self.uncertainty_score)
        print("\nuncertainty_score Calculated") 
        pbar.update(100/16)
        
        df['constraining_score'] = df.CONTENT.apply(self.constraining_score)
        print("\nconstraining_score Calculated") 
        pbar.update(100/16)
        
        df['positive_word_proportion'] = np.vectorize(self.positive_word_prop)(df['positive_score'],df['word_count'])
        print("\npositive_word_proportion Calculated") 
        pbar.update(100/16)
        
        df['negative_word_proportion'] = np.vectorize(self.negative_word_prop)(df['negative_score'],df['word_count'])
        print("\nnegative_word_proportion Calculated") 
        pbar.update(100/16)
        
        df['uncertainty_word_proportion'] = np.vectorize(self.uncertain_word_prop)(df['uncertainty_score'],df['word_count'])
        print("\nuncertainty_word_proportion Calculated") 
        pbar.update(100/16)
        
        df['constraining_word_proportion'] = np.vectorize(self.constraining_word_prop)(df['constraining_score'],df['word_count'])
        print("\nconstraining_word_proportion Calculated") 
        pbar.update(100/16)
        
        df['constraining_words_whole_report'] = np.vectorize(self.constrain_word_whole)(df['CONTENT'])
        print("\nconstraining_words_whole_report Calculated") 
        pbar.update(100/16)

        df.drop(['CONTENT'],axis=1,inplace=True)

        print("\nAnalysing Process Done")
        pbar.update(100/16)
        self.output=df
        pbar.close()
        
    def get_output(self):
        return self.output

    def saveOutput(self):
        if self.output is not None:
            self.output.to_csv('submission.csv', sep=',', encoding='utf-8')
            print("\nOutput File Saved")

ed=EdgarAnalysis()
ed.rawdata_extract(INPUT_FILE)
ed.load_stopwords()
ed.load_positiveWordList()
ed.load_negativeWordList()
ed.load_uncertainDictionary()
ed.load_constrainDictionary()
ed.analyse()

# print(ed.get_output())

ed.saveOutput()