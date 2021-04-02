import pandas as pd
#import xlsxwriter
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
import re
import InvertedIndex
import fasttext
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
"""
import string
from nltk.corpus import wordnet
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
"""
##########
# STEP 1 #
##########

print("starting preprocessing")
stopwords = "StopWords.txt" #list stop words
tweets = "Trec_microblog11.txt" #txt of the original tweets

tknzr = TweetTokenizer()# create tokenizer

stopWordsList = pd.read_csv(stopwords, sep="\n", header=None, error_bad_lines=False) #stopwords dataset
data = pd.read_csv(tweets, sep="\t", header=None, error_bad_lines=False) #tweets dataset

stopWordsList.columns = ["words"]#set column name
data.columns = ["tweetID", "tweet"]#set column names
tweetList = data.loc[:,"tweet"]#create token array
tweetID = data.loc[:,"tweetID"]#create tweet ID
stops = stopWordsList.loc[:,"words"]#create stopword array

# tweetList = tweetList[0:1000]#TODO TEMPORARY TEST SET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# tweetID = tweetID[0:1000]#TODO TEMPORARY TEST SET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

tokenArray = []
tweetTokensCopy = []
for tweet in tweetList:
    tweetTokens = tknzr.tokenize(tweet) # tokenize tweets
    tweetTokensCopy = []
    for word in tweetTokens:
    
        word = word.lower() #set all tweet tokens lowercase
        word = re.sub("\W+","",word) #remove non-alphabet characters
        word = re.sub("[0-9]+","",word) #remove numbers
       # filter(lambda x:x[0]!='www', word.split())
       # print(word)

        if "http" not in word and "www" not in word and word != "" and word not in stopWordsList.values: # only add to output non-stopwords
            tweetTokensCopy.append(word)
    tokenArray.append(tweetTokensCopy) #add tweet tokens to output

print(tokenArray.length())

#print(tokenArray)#for test purposes 

#############################################################################
#                                A2 HERE                                    #
#############################################################################

print("Downloading model and returning it as object")
model_glove_twitter = api.load("glove-twitter-25")

print("Finding top similarity for each token")

counter = 0
for token in tokenArray:
    for word in token:
        if word in model_glove_twitter:
            counter = counter + 1
            print("word: ",word, counter)
            model_glove_twitter.most_similar(word,topn=1)

"""
add all tweetID and tweets to the Inverted Index
"""

print("adding to inverted index")
corpusInvertedIndex = InvertedIndex.InvertedIndex()
for i in range(len(tweetID)):
    corpusInvertedIndex.insertTokenList(tokenArray[i],tweetID[i])

print("testing queries")


##########
# STEP 4 #
##########
#write the top 1000 results
"""
topic_id = the topic/query number (use the numbers, such a 1 instead of MB001)
Q0 = an unused field (the literal 'Q0')
docno = the tweet id, rank is the rank assigned by your system to the segment (1 is the highest rank)
score = the computed degree of match between the segment and the topic
tag = a unique identifier you chose for this run (same for every topic).
"""

def WriteDownResults(query,topic_id,resultFile):
    # print("start of WriteDownResults")
    queryResults = corpusInvertedIndex.rankedRetrieval(query)#get all match scores and what tweet IDs they are connected to
    # print("trim list top 1000")
    queryResults = queryResults[:999]# trim list to 1000 results
    # print("before for loop")

    counter=0
    for (theTweetID,score) in queryResults:
        counter+=1
        topic_id, Q0, docno, rank, score, tag = topic_id, "Q0", theTweetID, counter, score, "myTag"#setting all variables
        resultFile.write("{}   {}   {}   {}   {}   {}\n".format(topic_id, Q0, docno, rank, score, tag))#formating and writing to file
        # print("{}   {}   {}   {}   {}\n".format(topic_id, Q0, docno, rank, score, tag))
    resultFile.write("\n\n")
    # topic_id=None
    # query=None

# make queries
counter=0
queryList = []
queryFileAddress = "topics_MB1-49.txt"#address of queries
queryFile = open(queryFileAddress, "r")#open the query file
line = queryFile.readline()#read line of query file
resultFile = open("Results.txt", "w")#open results file to write results
while line:#loop through getting the queries and the query number
    line = queryFile.readline()#read a new line
    
    topic_id_search = re.search('<num> Number: (.*) </num>',line,re.IGNORECASE)
    query = re.search('<title>(.*)</title>',line,re.IGNORECASE)
    if topic_id_search:
        topic_id = topic_id_search.group(1)
        topic_id = topic_id[2:]
        topic_id = topic_id.lstrip("0")
        #print("sssss" + topic_id)
    if query:
        query = query.group(1)
        query = tknzr.tokenize(query)
        # print(query)
        # print("sssss" + topic_id)
        WriteDownResults(query,topic_id,resultFile)

"""
f = open("topics_MB1-49.txt","r")
fout = open("topics_MB1-49.txt","w", encoding="utf-8")
stop_words=pd.read_csv(stopwords, sep="\n", header=None, error_bad_lines=False)

print("Before While")

while 1:
    print("In while")
    line=f.readline()
    if not line:
        break
    line=line.replace('\n','')
    line= line.split(" ",1)
    new_line=line[0]
    line[1]=line[1].lower()
    line[1]=line[1].translate(str.maketrans('','',string.punctuation))
    word_tokens = word_tokenize(line[1])
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    synonyms=[]

    count=0
    print("Before For")
    for x in filtered_sentence:
        
        print("In for 1")
        for syn in wordnet.synsets(x):
            for l in syn.lemmas() :
                print("In For 2")
                if(count<3):
                    if l.name() not in synonyms:
                        print("In inner If")
                        synonyms.append(l.name())
                        count+=1
                        
        count=0
    
    print("End For")
    synonyms_string=' '.join(synonyms)
    new_line=" ".join([str(new_line),synonyms_string])
    synonyms=[]
    fout.write(new_line)
    fout.write('\n')

    print("synonyms",synonyms)
        
f.close()
fout.close()
"""
resultFile.close()