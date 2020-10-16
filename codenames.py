import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')

import wikipedia
import numpy as np
import re
regex = re.compile('[^a-z]')

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import networkx as nx
import matplotlib.pyplot as plt
import tkinter
from itertools import combinations

#Gensime word2vec model
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=500000
)

#Additional words that need to be removed
removedWords = ["besides", "also", "likewise", "too", "aswell", "st", "th", "i", "ane" \
				"may", "many", "one", "new", "two", "first", "use", "used", "known", \
				"however", "often", "including", "three"]

#Extracts the words and their frequency of occurance, weighted 1-100
def getWikiWords(inputWord):
	#Find words that include the desired one
	wordPages = wikipedia.search(inputWord)
	#wordPages = wikipedia.search(inputWord, results=3)
	#wordPages = wikipedia.suggest(inputWord)
	content = dict()

	#Loop over the pages adding up their content
	for i in wordPages:
		#Check that the page exists
		page = wiki_wiki.page(i)
		if page.exists():
			#Extract the words from the page
			words_pres = page.text.split()
			#Loop through the words and add it to content
			for currWord in words_pres:
				#Convert to lower case
				currWord = currWord.lower();
				#If it has a space, skip it
				if ' ' in currWord:
					continue
				#Remove punctiation and numbers
				currWord = regex.sub('', currWord)
				#Remove empty words
				if(currWord):
					#If the word is in the dictionary, increment its count
					if currWord in content:
						content[currWord] = content[currWord] + 1
					#If not, add the word to the dictionary
					else:
						content[currWord] = 1

	#Remove the common words
	for removalWord in stopwords.words('english'):
		if removalWord in content:
			del content[removalWord]
	for removalWord in removedWords:
		if removalWord in content:
			del content[removalWord]

	if "may" in content:
		del content["may"]

	#Remove the words with few referances
	content = {k:v for k, v in content.items() if v > 4}

	#Remove the word itself
	content = {k:v for k,v in content.items() if not inputWord in k}

	#Adjust the occurance count to be a weightint 0-1
	valueRange = content.values()
	minVal = min(valueRange);
	maxVal = max(valueRange)
	newMin = 0
	newMax = 1
	for i in content:
		content[i] = float((((content[i] - minVal) * (newMax - newMin)) / (maxVal - minVal)) + newMin)

	#Sort and print the dictionary
	#counter = 0
	#sorted_content = sorted(content.items(), key=lambda x: x[1], reverse=True)
	#for i in sorted_content:
	#	print(i[0], i[1])
	#	counter += 1
	#	if counter > 10:
	#		break

	return content

#Uses word2Vec to get the most similar words
def getWord2VecWords(inputWord):
	#Get the 500 most similar words
	relatedWords = model.similar_by_word(inputWord, 10)

	#Create a dictionary and add all the words to it
	content = dict();
	#Loop through the gensim words, removing those with spaces and the actual word
	for i in relatedWords:
		print(i)
		#Covert the word to lowercase

		#If it has special characters, ignore it

		#If it is based on the original word, ignore it

		#Add it to the dictionary with the correct weight, if it is not already



#Returns synonyms for a word
def getSynonyms(inputWord):
	synonyms = []

	for syn in wordnet.synsets(inputWord):
		for l in syn.lemmas():
			newWord = l.name().lower()
			#If it has a space, skip it
			if not ' ' in newWord:
				newWord = regex.sub('', newWord)
				synonyms.append(newWord)

	#Remove words that are just the same word
	for i in list(synonyms):
		if inputWord in i:
			synonyms.remove(i)
	
	#Remove the common words
	for removalWord in stopwords.words('english'):
		if removalWord in synonyms:
			synonyms.remove(removalWord)
	for removalWord in removedWords:
		if removalWord in synonyms:
			synonyms.remove(removalWord)


	#print(set(synonyms))
	return set(synonyms)

#Adds a word inputWord and its related words to a word map G
def addWordToMap(G, inputWord):
	G.add_node(inputWord)
	#Add all the related words to the map with edges
	relatedWords = getWikiWords(inputWord)
	for i in relatedWords:
		G.add_node(i)
		G.add_edge(inputWord, i, weight = relatedWords[i])
		G.add_edge(i, inputWord, weight = relatedWords[i])
		#Add the synonyms for the related words
		#G = addSynonymsToNode(G, inputWord, i, relatedWords[i])

	return G

#Adds all the synonymns for synWord to an existing node inputNode in map G
#The weighting for synonyms is 90% of the actual connector, to avoid overshadowing the word
def addSynonymsToNode(G, inputNode, synWord, weighting):
	syns = getSynonyms(synWord)
	#Add the synonyms nodes and edges if they exist
	if(syns):
		for word in syns:
			G.add_node(word)
			G.add_edge(inputNode, word, weight = int(weighting*0.8))
			G.add_edge(word, inputNode, weight = int(weighting*0.8))
	return G

#Find connection between two words, assuming one deegree of seperation
def findConnection(G, inputNode1, inputNode2):
	node1Dir = list(G.successors(inputNode1))
	node2Dir = list(G.successors(inputNode2))
	connections = np.empty()
	#Check if there are any in common
	for nodes1 in node1Dir:
		if nodes1 in node2Dir:
			print(nodes1)


#Creates a drawing of a wrd map
def drawNetwork(G):
	nx.draw(G)
	plt.savefig("NetworkImage.png")
	plt.show()

#Class that generates and stores connections between words
class WordConnections:
	#Initialize it with a list of words to be linked and the network G
	#It must contain at least 2 elements
	def __init__(self, G, inputNodes):
		self.wordList = inputNodes
		pathList = set()
		#Iterate over the paths between the nodes
		for path in nx.all_simple_paths(G, source = self.wordList[0], target = self.wordList[1], cutoff=2):
			pathList.add(path[1])

		#If there are more nodes remaining, add the rest to the set
		if len(self.wordList) > 2:
			for i in self.wordList[2:]:
				#Create a new path list for comparing and add the new paths
				newPathList = set()
				for path in nx.all_simple_paths(G, source = self.wordList[0], target = i, cutoff=2):
					#If the new connection is present in prev set, add it to new set
					if path[1] in pathList:
						newPathList.add(path[1])
				#Make the old set this new set
				pathList.clear()
				pathList = newPathList

		#If the key nodes are in the set, remove them
		for removalWords in self.wordList:
			if removalWords in pathList:
				pathList.remove(removalWords)

		#Create a dictionary of the connecting nodes and their connecting strengths
		connectionDictionary = dict()
		#Iterate over the set of the connectors and our nodes, adding them to the dictionary
		for connectors in pathList:
			weights = [0] * len(self.wordList)
			for i in range(len(self.wordList)):
				weights[i] = G[self.wordList[i]][connectors]["weight"]
			connectionDictionary[connectors] = float(np.mean(weights))

		self.connections = connectionDictionary
		self.connectionNodes = pathList

	#Return the dictionary of all the connections and their strength
	def getNodes(self):
		return wordList

	#Retuns the connection words between the nodes
	def getConnectionNodes(self):
		return connectionNodes

	#Prints the top 10 connections, sorted by weight
	def printConnections(self):
		sorted_content = sorted(self.connections.items(), key=lambda x: x[1], reverse=True)
		counter = 0
		for i in sorted_content:
			print(i[0], i[1])
			counter += 1
			if(counter > 10):
				break

class Spymaster():
	#Initialize the class with all the input words
	def __init__(self, teamColorInput, redWordsInput, blueWordsInput, blackWordInput):
		self.teamColor = teamColorInput.lower()
		if(teamColor == "red"):
			self.teamWords = redWordsInput
			self.enemyWords = blueWordsInput
		else:
			self.teamWords = blueWordsInput
			self.enemyWords = redWordsInput
		self.blackWord = blackWordInput

		#Initialize the graph
		self.G = nx.DiGraph()

		#Add all the words to the graph
		for i in self.teamWords:
			print("Calculating map for", i)
			G = addWordToMap(G, i)
		for i in self.enemyWords:
			print("Calculating map for", i)
			G = addWordToMap(G, i)
		print("Calculating map for", self.blackWord)
		G = addWordToMap(G, self.blackWord)

	#Generates the array that shows all of the possible decisions and their strengths
	def generateDecisionMatrix(self):
		#The matrix has the following format for each row (13 columns)
		#ConnectionWord NumNodes Node1name Node2Name Node3Name Node4Name
		#Node1Strength Node2Strength Node3Strength Node4Strength
		#MaximumEnemyWordStrength MaxNeutralWordStrength BlackWordStrength
		
		#Loop over all the possible connections of words for the team
		#Calculate for combinations of 2, 3, 4
		comb2 = list(combinations(self.teamWords, 2))
		comb3 = list(combinations(self.teamWords, 3))
		comb4 = list(combinations(self.teamWords, 4))
		comb = comb2 + comb3 + comb4
		for i in comb:
			combConnection = WordConnections(G, i);



def main():

	#Create a graph of the word map
	G = nx.DiGraph()

	#words = ["cat", "dog", "fox"]
	#for i in words:
	#	print("Add", i, "to map")
	#	G = addWordToMap(G, i)
	#connectA = WordConnections(G, words)
	#connectA.printConnections()

	getWord2VecWords("cat")

if __name__ == "__main__":
	main()