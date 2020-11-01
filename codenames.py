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
import copy

#Gensime word2vec model
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=1000000
)

#Solving method, either wiki or word2Vec
solvingMethod = 'wiki'

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
				currWord = currWord.lower()
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
	minVal = min(valueRange)
	maxVal = max(valueRange)
	newMin = 0
	newMax = 1
	for i in content:
		content[i] = float((((content[i] - minVal) * (newMax - newMin)) / (maxVal - minVal)) + newMin)

	#Sort and print the dictionary
	counter = 0
	newContent = dict()
	sorted_content = sorted(content.items(), key=lambda x: x[1], reverse=True)
	for i in sorted_content:
		newContent[i[0]] = i[1]
		counter += 1
		if counter > 1000:
			break

	return newContent

#Uses word2Vec to get the most similar words
def getWord2VecWords(inputWord):
	#Get the 1000 most similar words
	try:
		relatedWords = model.similar_by_word(inputWord, 1000)
	except:
		#If the word was not present in this model, use wiki words
		return getWikiWords(inputWord)
	#Create a dictionary and add all the words to it
	content = dict()
	#Loop through the gensim words, removing those with spaces and the actual word
	for i in relatedWords:
		#Covert the word to lowercase
		currWord = i[0].lower()
		#If it has special characters, ignore it
		if ' ' in currWord:
			continue
		#Remove punctiation and numbers
		currWord = regex.sub('', currWord)
		#If it is based on the original word, ignore it
		if inputWord in currWord:
			continue
		#Add it to the dictionary with the correct weight, if it is not already
		#Remove empty words
		if(currWord):
			if currWord not in content:
				content[currWord] = i[1]

	#Return the dictionary of words
	return content


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
def addWordToMap(G, inputWord, method):
	G.add_node(inputWord)
	#Add all the related words to the map with edges
	if(method == 'wiki'):
		relatedWords = getWikiWords(inputWord)
	else:
		relatedWords = getWord2VecWords(inputWord)
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

	#Return a list of the nodes that are being connected
	def getNodes(self):
		return self.wordList

	#Retuns the connection words between the nodes
	def getConnectionNodes(self):
		return self.connectionNodes

	#Prints the top 10 connections, sorted by weight
	def printConnections(self):
		sorted_content = sorted(self.connections.items(), key=lambda x: x[1], reverse=True)
		counter = 0
		for i in sorted_content:
			print(i[0], i[1])
			counter += 1
			if(counter > 10):
				break

#Class for storing all the data related for decision making
class DecisionData():
	#Initializing the class
	def __init__(self, connectionWord, numNodes, nodeNames, nodeStrength, \
		enemyStrength, neutralStrength, blackStrength):
		#Word that connects the nodes
		self.connectionWord = connectionWord
		#Number of nodes that are being connected which is 1-4
		self.numNodes = numNodes
		#Names of the nodes being connected, 1-4 values
		self.nodeNames = nodeNames
		#Strength of the node to connection word, 1-4 values
		self.nodeStrength = nodeStrength
		#Maximum strength between an enemy word and the connection
		self.maxEnemyStrength = enemyStrength
		#Maximum strength between a neutral word and the connection
		self.maxNeutralStrength = neutralStrength
		#Maximum strength between the black word and the connection
		self.maxBlackStrength = blackStrength

		#Create the value function
		#The value is the sum of the node strengths 
		#If any of the nodeStrengths are less than neutral, enemy, or black
		#set the value to 0
		try:
			self.value = sum(self.nodeStrength)
			self.minNodeStrength = min(self.nodeStrength)
		except:
			self.value = self.nodeStrength
			self.minNodeStrength = self.nodeStrength
		
		strongestEnemy = max([self.maxEnemyStrength, self.maxNeutralStrength, self.maxBlackStrength])
		if(self.minNodeStrength < strongestEnemy):
			self.value = 0

#Helper function for sorting a list of decisions
#Returns the value of a decision
def getDecisionValue(inputDecision):
	return inputDecision.value

#Prints out a list of possible decisions given a list of DecisionData
def printDecisionList(decisionList):
	for i in decisionList:
		printDecision(i)

def printDecision(decision):
	output = decision.connectionWord + " " + str(decision.numNodes) + " [ "
	for j in decision.nodeNames:
		output = output + j + " "
	output = output + "] "
	output = output + str(decision.value)
	print(output)

#Stores a decision path through a game, keeping track of the game state
class DecisionPath():
	#Initialize the class with the list of team words
	def __init__(self, teamWordsInput):
		self.teamWords = copy.deepcopy(teamWordsInput)
		self.decisionList = []

	#Returns true if the path is complete, there are still words remaining
	def pathComplete(self):
		if(self.teamWords):
			return False
		return True
	
	#Returns the list of team words remaining
	def getRemainingWords(self):
		return self.teamWords

	#Adds a new decision to the path
	def addDecision(self, newDecision):
		self.decisionList.append(newDecision)
		#Update the list of team words to remove the ones in the decision
		for i in newDecision.nodeNames:
			self.teamWords.remove(i)

	#Removes the most recently added decision to the path
	def popDecision(self):
		removedDecision = self.decisionList[-1]
		#Add the words back to the list
		for i in removedDecision.nodeNames:
			self.teamWords.append(i)
		#Remove the decision
		self.decisionList.pop()

	#Returns the value
	def getValue(self):
		#Update the value of the path which is average of the decision values divided by the path length
		decisionValues = []
		for i in self.decisionList:
			decisionValues.append(i.value)
		self.value = sum(decisionValues)/len(self.decisionList)/len(self.decisionList)
		return self.value

	#Prints out the decision path
	def printDecisionPath(self):
		#printDecision(self.decisionList[1])
		printDecisionList(self.decisionList)

class Spymaster():
	#Initialize the class with all the input words
	def __init__(self):
		pass

	def addWords(self, teamColorInput, redWordsInput, blueWordsInput, neutralWordsInput, blackWordInput):
		print("Beginning game as spymaster")
		self.teamColor = teamColorInput.lower()
		if(teamColorInput == "red"):
			self.teamWords = redWordsInput
			self.enemyWords = blueWordsInput
		else:
			self.teamWords = blueWordsInput
			self.enemyWords = redWordsInput
		self.blackWord = blackWordInput
		self.neutralWords = neutralWordsInput

		#Initialize the graph
		self.G = nx.DiGraph()

		print("Calculating word map...")
		#Add all the words to the graph
		for i in self.teamWords:
			#print("Calculating map for", i)
			self.G = addWordToMap(self.G, i, solvingMethod)
		for i in self.enemyWords:
			#print("Calculating map for", i)
			self.G = addWordToMap(self.G, i, solvingMethod)
		for i in self.neutralWords:
			#print("Calculating map for", i)
			self.G = addWordToMap(self.G, i, solvingMethod)
		#print("Calculating map for", self.blackWord)
		self.G = addWordToMap(self.G, self.blackWord, solvingMethod)

		print("Map of all words created")

	def test(self):
		wordlist = ["car", "truck", "train"]
		combConnection = WordConnections(self.G, wordlist)
		print("Output Combos")
		print(combConnection.getConnectionNodes())

	#Generates the list of data that shows all of the possible decisions and their strengths
	def generateDecisionMatrix(self, teamWordsInput, enemyWordsInput, neutralWordsInput, blackWordsInput):
		decisionList = []
		#Loop over all the possible connections of words for the team
		#Calculate for combinations of 2, 3, 4
		comb = list()

		if(len(teamWordsInput) >= 2):
			comb.extend(list(combinations(teamWordsInput, 2)))
		if(len(teamWordsInput) >= 3):
			comb.extend(list(combinations(teamWordsInput, 3)))
		if(len(teamWordsInput) >= 4):
			comb.extend(list(combinations(teamWordsInput, 4)))

		#Where i is the words that need to be combinded over all possible combinations
		for i in comb:
			#Generate the possible connections over i
			combConnection = WordConnections(self.G, i)
			for connectionNode in combConnection.getConnectionNodes():
				#Get the strenth between the connection word and the others
				nodeStrength = []
				for j in i:
					nodeStrength.append(self.G[j][connectionNode]["weight"])

				#Find the max strength between enemy words and the connector
				enemyStrength = []
				for j in enemyWordsInput:
					#If the connection is there, add it
					if(connectionNode in self.G.neighbors(j)):
						enemyStrength.append(self.G[j][connectionNode]["weight"])
					#Otherwise, add a 0
					else:
						enemyStrength.append(0)
				if(not enemyStrength):
					enemyStrength = [0]
				#Find the max strength between the neutral words and the connector
				neutralStrength = []
				for j in neutralWordsInput:
					#if the connection is there, add it
					if(connectionNode in self.G.neighbors(j)):
						neutralStrength.append(self.G[j][connectionNode]["weight"])
					else:
						neutralStrength.append(0)
				if(not neutralStrength):
					neutralStrength = [0]
				#Find the black word strength to the connection node
				blackStrength = 0
				if(connectionNode in self.G.neighbors(blackWordsInput)):
					blackStrength = self.G[blackWordsInput][connectionNode]["weight"]
				
				#Create the decision data and push it to the list
				newDecisionData = DecisionData(connectionNode, len(i), i, nodeStrength, \
					max(enemyStrength), max(neutralStrength), blackStrength)

				#If the value is nonzero, add it to the list
				if(newDecisionData.value > 0):
					decisionList.append(newDecisionData)

		#If there is only one word left, add it to the decision list
		if(len(teamWordsInput) == 1 or ((not decisionList) and len(teamWordsInput) < 4)):
			nodeName = teamWordsInput[0]
			#Get the most relevant words
			if(solvingMethod == 'wiki'):
				relatedWords = getWikiWords(nodeName)
			else:
				relatedWords = getWord2VecWords(nodeName)
			
			#Add all the words to the decision list
			for i in relatedWords:
				nodeStrength = self.G[i][nodeName]["weight"]

				#Find the max strength between enemy words and the connector
				enemyStrength = []
				for j in enemyWordsInput:
					if(nodeName in self.G.neighbors(j)):
						enemyStrength.append(self.G[j][nodeName]["weight"])
					else:
						enemyStrength.append(0)
				if(not enemyStrength):
					enemyStrength = [0]
				#Find the max strength between the neutral words and the connector
				neutralStrength = []
				for j in neutralWordsInput:
					if(nodeName in self.G.neighbors(j)):
						neutralStrength.append(self.G[j][nodeName]["weight"])
					else:
						neutralStrength.append(0)
				if(not neutralStrength):
					neutralStrength = [0]
				#Find the black word strength to the connection node
				blackStrength = 0
				if(nodeName in self.G.neighbors(blackWordsInput)):
					blackStrength = self.G[blackWordsInput][nodeName]["weight"]

				#Create the decision data and push it to the list
				newDecisionData = DecisionData(i, 1, [nodeName], [nodeStrength], \
					max(enemyStrength), max(neutralStrength), blackStrength)
				
				if(newDecisionData.value > 0):
					decisionList.append(newDecisionData)

		#Sort the decisionList by value
		decisionList.sort(reverse=True, key=getDecisionValue)
		return decisionList

	#Returns the decision with the maximal value based on the current state of the game
	def makeGreedyDecision(self):
		#Generate list of possible decisions from current game state
		decisionList = self.generateDecisionMatrix(self.teamWords, self.enemyWords, self.neutralWords, self.blackWord)		
		#The first decision has the highest value
		decision = decisionList[0]
		#Print it out
		printDecision(decision)

	#Instead of only looking at the current game state, this executes a foreward search
	#to other possible game states
	#Returns a path where the first decision is what should be made in the current
	#state and the rest are possible decisions for the future
	def makeForwardSeekingDecision(self):
		#Number of actions to be explored, only the top decisions will be explored
		decisionCount = 5

		#Create a queue of paths to be explored in the tree
		queue = []
		#List of finished paths
		finishedPaths = []

		#Get the starting list of decisions
		decisionList = self.generateDecisionMatrix(self.teamWords, self.enemyWords, self.neutralWords, self.blackWord)
		#Extract the best decisions
		decisionList = decisionList[0:decisionCount]
		#Add them to the queue
		for i in decisionList:
			newPath = DecisionPath(self.teamWords)
			newPath.addDecision(i)
			if(newPath.pathComplete()):
				finishedPaths.append(newPath)
			else:
				queue.append(newPath)

		#Loop until the queue is empty
		while(queue):
			#Read the first path
			currentPath = queue[0]
			#Get list of decisions that could be made from this new state
			newTeamWords = currentPath.getRemainingWords()

			#Assume there are no enemy words
			newEnemyWords = []
			newDecisionList = self.generateDecisionMatrix(newTeamWords, newEnemyWords, self.neutralWords, self.blackWord)
			newDecisionList = newDecisionList[0:decisionCount]
			#Create the new paths and add them to the queue
			for i in newDecisionList:
				newPath = copy.deepcopy(currentPath)
				newPath.addDecision(i)
				#If the path is complete, add it to finished paths
				if(newPath.pathComplete()):
					finishedPaths.append(newPath)
				#Otherwise, add it to the queue
				else:
					queue.append(newPath)
			print(len(queue))
			#Remove the current element from the queue
			del queue[0]

		#Now, finished paths should be filled with all the possible valid paths
		#Find and return the one with the greatest value
		bestPath = finishedPaths[0]
		bestValue = finishedPaths[0].getValue()
		for i in finishedPaths:
			if(i.getValue() > bestValue):
				bestPath = i
				bestValue = i.getValue()

		#Return the final path
		bestPath.printDecisionPath()
		return bestPath
		

def setTeam():
	invalidInput = True
	#Enter user input
	#Get team color
	while(invalidInput):
		invalidInput = False
		team = input('Enter your team color [red] or [blue]: ').lower()
		if('r' in team):
			team = "red"
		elif('b' in team):
			team = "blue"
		else:
			print("Invalid input")
			invalidInput = True

	return team

def setColorWords(color):
	if(color == "black"):
		wordString = input('Enter the black word [a]: ').lower()
		return wordString
	else:
		wordString = input('Enter the '+color+' words, seperated by commas [a,b,c]: ').lower()
		words= wordString.split(",")
		return words

def printGameState(teamColor, redWords, blueWords, neutralWords, blackWord):
	print("-----------Game State-----------")
	print("Team Color: " + teamColor)
	line2 = "Red Words: ["
	for i in range(len(redWords)):
		line2 = line2 + redWords[i]
		if i < (len(redWords)-1):
			line2 = line2 + ","
	line2 = line2 + "]"
	print(line2)
	line3 = "Blue Words: ["
	for i in range(len(blueWords)):
		line3 = line3 + blueWords[i]
		if i < (len(blueWords)-1):
			line3 = line3 + ","
	line3 = line3 + "]"
	print(line3)
	line4 = "Neutral Words: ["
	for i in range(len(neutralWords)):
		line4 = line4 + neutralWords[i]
		if i < (len(neutralWords)-1):
			line4 = line4 + ","
	line4 = line4 + "]"
	print(line4)
	print("Black Word: " + blackWord)



def main():

	#newGame = Spymaster(team, redWords, blueWords, neutralWords, blackWord)

	#newGame.makeForwardSeekingDecision()
	
	#Start the game
	print("_______________________________")
	print("-----Starting Codenames AI-----")
	print("===============================")

	state = "init"
	playGame = True

	teamColor = ""
	redWords = []
	blueWords  = []
	neutralWords = []
	blackWord = ""
	spymaster = Spymaster()

	while(playGame):

		#Initialize the game state
		if(state == "init"):
			print("Starting New Game--------------")
			teamColor = setTeam()
			redWords = setColorWords("red")
			blueWords = setColorWords("blue")
			neutralWords = setColorWords("neutral")
			blackWord = setColorWords("black")
			#Output the data to make sure it looks alright
			printGameState(teamColor, redWords, blueWords, neutralWords, blackWord)

			#Check if the data is correct
			getInput = input('Is this data correct? [y/n]: ').lower()
			if('y' in getInput):
				state = "playing"
				spymaster.addWords(teamColor, redWords, blueWords, neutralWords, blackWord)
			#If the input is not correct, restart the init

		if(state == "playing"):
			#Evaluate if the game should keep running or exit
			if((not redWords) or (not blueWords)):
				break

			printGameState(teamColor, redWords, blueWords, neutralWords, blackWord)
			print("----------Choose an action ----------------------------------")
			print("[gg] Make the best guess with the current game state")
			print("[fs] Make the best guess while considering future game states")
			print("[rm word] Remove a word from the game")
			print("[add word team] Adds a word to a given team")
			userInput = input().lower()
			userInput = userInput.split(" ")
			if(userInput[0] == 'gg'):
				print("========HINT========")
				spymaster.makeGreedyDecision()
				print("====================")
			elif(userInput[0] == 'fs'):
				print("========HINT========")
				spymaster.makeForwardSeekingDecision()
				print("====================")
			elif(userInput[0] == 'rm'):
				removalWord = userInput[1]
				if(removalWord in redWords):
					redWords.remove(removalWord)
				if(removalWord in blueWords):
					blueWords.remove(removalWord)
				if(removalWord in neutralWords):
					neutralWords.remove(removalWord)
			elif(userInput[0] == 'add'):
				teamtoadd = userInput[2]
				if(teamtoadd == "red"):
					redWords.append(userInput[1])
				elif(teamtoadd == "blue"):
					blueWords.append(userInput[1])
				elif(teamtoadd == "neutral"):
					neutralWords.append(userInput[1])
				else:
					print("Invalid input")
			else:
				print("Invalid input entered")

	print("Thanks for playing")
	
	
	


if __name__ == "__main__":
	main()