# CSI4107_Assignment_1

CSI4107 - Assignment 1 README

Theo Van der Burgt	300019142
Ariane Poserio		300011641
Anshu Sharma		300011600

-------------------------------------------------------------------------------------------------------------------------------
Work Division
-------------------------------------------------------------------------------------------------------------------------------
Part 1 = Theo
Part 2 = Anshu
Part 3 = Anshu and Ariane
Part 4 = Theo, Ariane
Part 5 = Theo, Ariane, Anshu
README = Theo and Ariane

-------------------------------------------------------------------------------------------------------------------------------
Run File Instructions
-------------------------------------------------------------------------------------------------------------------------------
On your windows machine, download the code from github and using the command prompt, change directory to the assignment folder.

Once inside the CSI4107_Assignment_1 folder, run the following command to load the assignment:

	Python A1.py

The file should then run and display the required results

-------------------------------------------------------------------------------------------------------------------------------
Explanation of Algorithm and Samples
-------------------------------------------------------------------------------------------------------------------------------
For this assignment pandas, nltk, and regular expression operations were libraries used to help preprocess and process the data. For this assignment, we had a vocabulary size of 95288 words consisting of tokens in the file named SampleVocab.txt.

These results would produce a list of topic IDs, queries, Tweet IDs, and the score. A sample of these results are shown below, with queries 3 and 20 being used to display the data. 

The data is separated by: topic_id, Q0, tweet_ID, counter, score

3   Q0   30207053444153345   1   1.0   myTag
3   Q0   33995136060882945   2   1.0   myTag
3   Q0   29367562181550080   3   0.6182559695547944   myTag
3   Q0   29108245628981248   4   0.6152929600049993   myTag
3   Q0   29780564701618176   5   0.5985355219313464   myTag
3   Q0   33388661584175104   6   0.579817788213962   myTag
3   Q0   30062788315447296   7   0.5301055415222135   myTag
3   Q0   31435766936633344   8   0.5268301777530708   myTag
3   Q0   34394616581066752   9   0.5193484905884698   myTag
3   Q0   34617019294814209   10   0.5036509714742771   myTag

20   Q0   29158340424634368   1   0.5096823535722769   myTag
20   Q0   30328331056447488   2   0.486045154621561   myTag
20   Q0   34134879876681728   3   0.4127545453527601   myTag
20   Q0   29906116062220290   4   0.4042029383569113   myTag
20   Q0   29853985930219520   5   0.3841140005855645   myTag
20   Q0   31082136219947008   6   0.37079230823829823   myTag
20   Q0   32443636847218688   7   0.3660051885239529   myTag
20   Q0   31742053993938945   8   0.34051081222868185   myTag
20   Q0   31161931205181440   9   0.3238057242369439   myTag
20   Q0   29913837511647232   10   0.31305151378701773   myTag

-------------------------------------------------------------------------------------------------------------------------------
Functionality
-------------------------------------------------------------------------------------------------------------------------------
#################
STEP 1 
#################
Main script
A1.py

A1.py is the main script which first parses through the tweets then splits it into the tweetIDs and the tweet content. The tweets are then tokenized and are then converted into all lowercase letters and removes all non-english characters, numbers, and stop words. 

Helper Class
InvertedIndex.py
Creates a and adds IDs and Tweet Tokens to an inverted index 

#################
STEP 2
#################
Classes:
InvertedIndex
Class containing various functions which creates the inverted index

Functions
__init__
Constructor for the inverted index class
insertToken
Takes token and tweetID as input and inserts tokens to the inverted index
insertTokenList
Takes tokenList and tweetID as input and returns a list of tokens with the same tweetID after inserting it into the inverted index
vocabSize
Returns the vocabulary of the inverted index
tokenSample	
Returns a sample of tokens from the inverted index

#################
STEP 3 
#################
Functions:
rankedRetrieval
Takes a query as input and returns a ranked list of the tweetIDs and the corresponding similarities for the given query by generating the cosine similarity for all the tweets from the data dictionary and then sorting them from highest to lowest

#################
STEP 4 
#################
Functions:
WriteDownResults
Takes query, topic_id, and resultFile as inputs and writes the top 1000 of the query into the designated file

#################
STEP 5 
#################

After running the trec_eval script, using the command:

	trec_eval Relevance.txt Results.txt

we received the following results with an overall MAP score of 17% accuracy

runid                 	all	myTag
num_q                 	all	49
num_ret               	all	48951
num_rel               	all	2640
num_rel_ret           	all	1879
map                   	all	0.1777
gm_map                	all	0.1195
Rprec                 	all	0.2184
bpref                 	all	0.1756
recip_rank            	all	0.4528
iprec_at_recall_0.00  	all	0.5259
iprec_at_recall_0.10  	all	0.3650
iprec_at_recall_0.20  	all	0.2856
iprec_at_recall_0.30  	all	0.2550
iprec_at_recall_0.40  	all	0.2233
iprec_at_recall_0.50  	all	0.1830
iprec_at_recall_0.60  	all	0.1445
iprec_at_recall_0.70  	all	0.1164
iprec_at_recall_0.80  	all	0.0713
iprec_at_recall_0.90  	all	0.0297
iprec_at_recall_1.00  	all	0.0098
P_5                   	all	0.2735
P_10                  	all	0.2531
P_15                  	all	0.2435
P_20                  	all	0.2490
P_30                  	all	0.2231
P_100                 	all	0.1614
P_200                 	all	0.1234
P_500                 	all	0.0716
P_1000                	all	0.0383
