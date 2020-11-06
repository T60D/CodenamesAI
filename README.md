# Codenames Spymaster AI

This is a algorithm that is capable of playing the spymaster role in a game of codenames. It takes user data and outputs the hint that will yield the best result

## Installation

Download this repository
Install all of the packages required
Download the pretrained Word2Vec model

## Usage

Set the solver on line 25 to either wiki or word2vec
```python
#Solving method, either wiki or word2Vec
solvingMethod = 'word2Vec'
```

Run the codenames.py program
When it asks for it, enter the data about the game (example below)

```python
_______________________________
-----Starting Codenames AI-----        
===============================        
Starting New Game--------------        
Enter your team color [red] or [blue]: red
Enter the red words, seperated by commas [a,b,c]: chair,table,bench
Enter the blue words, seperated by commas [a,b,c]: car,dog,rabbit
Enter the neutral words, seperated by commas [a,b,c]: apple,banana,pear
Enter the black word [a]: piano
'''

Once the program initializes the map(this may take awhile if using wikipedia), you can choose decisions to make:
gg : Returns the best decision given the current game state
fs : Returns the best decision given current and future game states (may take awhile to compute)
rm word : Removes the given "word" from the game
add word team : Adds a given "word" to a specific team. You cannot add words that were not in the originally calculated map

The game will end once one of the the teams is out of words.
