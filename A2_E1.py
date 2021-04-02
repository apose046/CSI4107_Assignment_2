import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import InvertedIndex
import numpy as np
from sent2vec.vectorizer import Vectorizer

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
tweetDict = {}
for i in range(len(tweetID)):
    tweetDict[tweetID[i]]= tweetList[i]
stops = stopWordsList.loc[:,"words"]#create stopword array

tokenArray = []
for tweet in tweetList:
    tweetTokens = tknzr.tokenize(tweet) # tokenize tweets
    tweetTokens = nltk.word_tokenize(tweet)
    
    tweetTokensCopy = []
    for word in tweetTokens:
        # word = re.sub("http(.*)","a",word) # remove links
        # word = re.sub("[0-9]*","a",word) #remove numbers
        # word = re.sub("\W+","a",word) #remove non-alphabet characters

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

def reduceTweetListSize(query, tweetList, tweetID, maxSize=len(tweetList)):
    print("start of reduceTweetListSize")
    #check for a query word
    smallTweetList = []
    smallTweetID = []
    for i in range(len(tweetList)-1):
        for word in query:
            if tweetList[i].find(word) != -1:
                smallTweetList.append(tweetList[i])
                smallTweetID.append(tweetID[i])
                continue
    return smallTweetList[:maxSize], smallTweetID[:maxSize]


def bert(query, tweetList):
    print("start of BERT")
    ##Added for A2 part 1.
    vectorizer = Vectorizer()
    queryString = ""
    for word in query:
        queryString = queryString +" "+word
    queryString = [queryString]
    queryString.extend(tweetList)
    print("Number of strings being processed "+str(len(queryString)))
    vectorizer.bert(queryString)
    vectors = vectorizer.vectors
    print("end of BERT")
    return vectors


"""
topic_id = the topic/query number (use the numbers, such a 1 instead of MB001)
Q0 = an unused field (the literal 'Q0')
docno = the tweet id, rank is the rank assigned by your system to the segment (1 is the highest rank)
score = the computed degree of match between the segment and the topic
tag = a unique identifier you chose for this run (same for every topic).
"""
def WriteDownResults(query,topic_id,resultFile):
    print("start of WriteDownResults")
    # queryResults = corpusInvertedIndex.rankedRetrieval(query)#get all match scores and what tweet IDs they are connected to
    # trim list to 1000 results
    
    smallTweetList, smallTweetID = reduceTweetListSize(query, tweetList, tweetID, 3000)
    vectors = bert(query, smallTweetList)
    qVect = vectors[0]
    vectToup = []
    #print(len(smallTweetID))
    #print(len(vectors))
    for i in range(len(smallTweetID)-1):
        vectToup.append((smallTweetID[i],vectors[i+1]))
    queryResults = CosineSim(vectors[0],vectToup)

    counter=0
    for (theTweetID,score) in queryResults:
        counter+=1
        topic_id, Q0, docno, rank, score, tag = topic_id, "Q0", theTweetID, counter, score, "myTag"#setting all variables
        resultFile.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(topic_id, Q0, docno, rank, score, tag))#formating and writing to file
    resultFile.write("\n\n")

def CosineSim(query, allVectors):
    print("start of CosineSim")
    CosinesSimValues = []
    for pair in allVectors:
        CosinesSimValues.append((np.dot(pair[1], query)/(np.linalg.norm(pair[1])*np.linalg.norm(query)),pair[0]))
    return CosinesSimValues

def part1 (query,topic_id,resultFile):
    print("start of part1")
    queryResults = corpusInvertedIndex.rankedRetrieval(query)#get all match scores and what tweet IDs they are connected to
    queryResults = queryResults[:1000]
    # trim list to 1000 results

    print("traversing dictionary")
    queryResultsTweetList = []
    for i in range(len(queryResults)):
        queryResultsTweetList.append((tweetDict[queryResults[i][0]],queryResults[i][0]))
    print("starting bert model")
    vectors = bert(query, queryResultsTweetList)
    print("starting bert model")
    qVect = vectors[0]
    vectToup = []
    for i in range(len(queryResultsTweetList)-1):
        vectToup.append((queryResultsTweetList[i],vectors[i+1]))
    queryResults = CosineSim(vectors[0],vectToup)

    counter=0
    for (theTweetID,score) in queryResults:
        counter+=1
        topic_id, Q0, docno, rank, score, tag = topic_id, "Q0", score[1], counter, theTweetID, "myTag"#setting all variables
        resultFile.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(topic_id, Q0, docno, rank, score, tag))#formating and writing to file
    resultFile.write("\n\n")

# make queries
counter=0
queryList = []
queryFileAddress = "topics_MB1-49.txt"#address of queries
queryFile = open(queryFileAddress, "r")#open the query file
line = queryFile.readline()#read line of query file
resultFile = open("Results_E1.txt", "w")#open results file to write results
qCount = 0
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
        #WriteDownResults(query,topic_id,resultFile)#write to file the results.#THIS IS FOR PART 3
        print("Start of Query "+ str(qCount))
        part1(query,topic_id,resultFile)#write to file the results. # THIS IS FOR PART 1
        print("end of Query "+ str(qCount))
        qCount +=1

resultFile.close()
