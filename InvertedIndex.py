
import numpy as np
"""
The file contains the inverted index class. The inverted index
structure is used through this assignment for easier indexing and 
Retrieval and Ranking.
Author: Anshu Sharma
"""
##########
# STEP 2 #
##########
class InvertedIndex:

	"""
	This is the constructor for the inverted index class.
	"""
	def __init__ (self):
		self.index = {}
		self.idList = set()
		self.N = len(self.idList)

	"""
	Insert token to the inverted index.
	"""
	def insertToken(self, token, tweetId):
		self.idList.add(tweetId) #update set of tweetId's 
		self.N = len(self.idList) #update the size of the set of tweetId's
		if token in self.index: #if token token is in index already updates the token's value dictionary
			if tweetId in self.index[token]:
				self.index[token][tweetId] += 1
			else:
				self.index[token][tweetId] = 1
		else: #else add token to the index.
			self.index[token] = { tweetId:1 }


	"""
	Insert an list of token with same tweet Id
	into the inverted index.
	"""
	def insertTokenList(self, tokenList, tweetId):
		for token in tokenList: #loops throught the list of token calling the insertToken function.
			self.insertToken(token, tweetId)


	"""
	Return vocabulary of the Inverted Index.
	"""
	def vocabSize(self):
		return len(self.index)


	"""
	Returns a sample of the words in the inverted index
	"""
	def tokenSample(self, size):
		wordList = list(self.index.keys())
		for i in range(size):
			print(wordList[i])


	##########
	# STEP 3 #
	##########
	"""
	Returns a ranked list of tweetID and coresponding similarity
	values for a query.
	"""
	def rankedRetrieval(self, query):

		#Data create a dictionary of elements used to find the norm
		#and dot product for specific tweetID to generate it's cosine Similarity
		data = {}
		query_normsqr = 0
		for token in self.index:

			#Generate IDF for token in Index
			idf = np.log2(self.N/len(self.index[token]))

			#Find query term frequency
			tf_q = 0
			if token in query:
				tf_q = query.count(token)
				query_normsqr += (tf_q*idf)**2 

			#Fill up data dictionary
			for tweetId in self.index[token]:
				tf_d  = self.index[token][tweetId]
				if tweetId in data:
					data[tweetId][0] += tf_d*tf_q*(idf**2)
					data[tweetId][1] += (tf_d*idf)**2
				else:
					data[tweetId] = [tf_d*tf_q*(idf**2) , (tf_d*idf)**2]

		query_norm = np.sqrt(query_normsqr)
		rankedResults = []

		#generate Cosine sim for all tweets from data dictionary and query_nrom
		for i in data:
			document_norm = np.sqrt(data[i][1])
			if query_norm!=0:
				cos_sim = data[i][0]/(query_norm*document_norm)
			else:
				cos_sim = 0
			rankedResults.append((i, cos_sim))
		
		#Sort HERE
		rankedResults.sort(key=lambda x: x[1])
		rankedResults.reverse()
		return rankedResults	