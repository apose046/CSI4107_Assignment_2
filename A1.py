import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import InvertedIndex

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

tokenArray = []
for tweet in tweetList:
    tweetTokens = tknzr.tokenize(tweet) # tokenize tweets
    tweetTokens = nltk.word_tokenize(tweet)
    
    tweetTokensCopy = []
    for word in tweetTokens:
        word = re.sub("http(.*)","a",word) # remove links
        word = re.sub("[0-9]*","a",word) #remove numbers
        word = re.sub("\W+","a",word) #remove non-alphabet characters

        if word not in stopWordsList.values: # only add to output non-stopwords
            tweetTokensCopy.append(word)
    tokenArray.append(tweetTokensCopy) #add tweet tokens to output


#add all tweetID and tweets to the Inverted Index
print("adding to inverted index")
corpusInvertedIndex = InvertedIndex.InvertedIndex()
for i in range(len(tweetID)):
    corpusInvertedIndex.insertTokenList(tokenArray[i],tweetID[i])
print("vocabulary of Inverted Index is " +str(corpusInvertedIndex.vocabSize()))
print("Here is a sample size of words in the Inverted Index")
corpusInvertedIndex.tokenSample(100)
print("\n")

print("Testing queries")
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
    queryResults = corpusInvertedIndex.rankedRetrieval(query)#get all match scores and what tweet IDs they are connected to
    queryResults = queryResults[:999]# trim list to 1000 results

    counter=0
    for (theTweetID,score) in queryResults:
        counter+=1
        topic_id, Q0, docno, rank, score, tag = topic_id, "Q0", theTweetID, counter, score, "myTag"#setting all variables
        resultFile.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(topic_id, Q0, docno, rank, score, tag))#formating and writing to file
    resultFile.write("\n\n")


# make queries
counter=0
queryList = []
queryFileAddress = "topics_MB1-49.txt"#address of queries
queryFile = open(queryFileAddress, "r")#open the query file
line = queryFile.readline()#read line of query file
resultFile = open("Results.txt", "w")#open results file to write results
while line:#loop through getting the queries and the query number
    line = queryFile.readline()#read a new line
    
    topic_id_search = re.search('<num> Number: (.*) </num>',line,re.IGNORECASE)#check for number
    query = re.search('<title>(.*)</title>',line,re.IGNORECASE)#check for query
    if topic_id_search:
        topic_id = topic_id_search.group(1)#set the topic id from the number
        topic_id = topic_id[2:]#remove the starting MB
        topic_id = topic_id.lstrip("0")
    if query:
        query = query.group(1)#set query from title
        query = tknzr.tokenize(query)#tokenize query string
        WriteDownResults(query,topic_id,resultFile)#write to file the results.

resultFile.close()
