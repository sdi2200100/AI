# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # worst cases
        if successorGameState.isLose(): return(float('-inf'))                     # if with our action we lose, return lowest number possible
        ghosts = successorGameState.getGhostPositions()
        for ghost in ghosts:
            if util.manhattanDistance(ghost,newPos) <= 3: return(float('-inf')+1) # slightly better (we don't lose but we are very close to ghosts)

        # best cases
        if successorGameState.isWin(): return(float('inf'))                       # if with our action we win, return highest number possible
        x , y = newPos
        currentFood = currentGameState.getFood()
        if currentFood[x][y]: return(float('inf')-1)                              # slightly worse (we don't win but we eat food and we are not close to any ghost)

        # other cases             
        dist = []                                                                 # else find the food closest to our position
        for food in newFood.asList():
            dist.append(util.manhattanDistance(food,newPos))
        dist.sort()     

        return -dist[0]                                                           # we return it's negative because if -x > -y => x < y and 
                                                                                  # we want the one with to lowest value to be the bigger number

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # throughout the code we don't take under consideration the stop move as it is objectively a bad move 

        minimax_action = []
        actions = gameState.getLegalActions(0)                                                 # generate possible actions for pacman
        for action in actions:
            if action == "Stop" : continue
            game = gameState.generateSuccessor(0,action)
            eval = self.minimax(game,0,1)        
            minimax_action.append((eval,action))
        minimax_action.sort()
        minimax_value , action = minimax_action[len(minimax_action)-1]
        return action

    def minimax(self, gamestate : GameState, depth, agent_index):
        if(gamestate.isLose() or gamestate.isWin()): return self.evaluationFunction(gamestate)  # terminal cases


        actions = gamestate.getLegalActions(agent_index)

        if(agent_index == 0):                                                                   # max agent
            if(depth == self.depth - 1): return self.evaluationFunction(gamestate)
            maxe = float('-inf')
            for action in actions:
                if action == "Stop" : continue
                game = gamestate.generateSuccessor(agent_index,action)
                eval = self.minimax(game, depth + 1, 1)
                maxe = max(maxe, eval)
            return maxe
        
        else:                                                                                    # min agents
            mine = float('inf')
            for action in actions:
                if action == "Stop" : continue
                game = gamestate.generateSuccessor(agent_index,action)
                if(agent_index == gamestate.getNumAgents() - 1):                                 # check if there are other agents to visit or go to pacman 
                    eval = self.minimax(game, depth,0)
                else:
                    eval = self.minimax(game, depth,agent_index + 1)
                mine = min(mine, eval)
            return mine

            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        minimax_value = float('-inf')
        infi = float('inf')
        minf = float('-inf')
        actions = gameState.getLegalActions(0)                # generate possible actions for pacman
        for action in actions:
            if action == "Stop" : continue
            game = gameState.generateSuccessor(0,action)
            eval = self.minimax_ab(game,0,1,minf,infi)        
            if(eval > minimax_value):
                minimax_value = eval
                minimax_action = action
                minf = max(minf , eval)                       # change lowest possible value for max agent if necessary
        return minimax_action

    def minimax_ab(self, gamestate : GameState, depth, agent_index, a, b):
        if(gamestate.isLose() or gamestate.isWin()): return self.evaluationFunction(gamestate)

        actions = gamestate.getLegalActions(agent_index)

        if(agent_index == 0):
            if(depth == self.depth - 1): return self.evaluationFunction(gamestate)
            maxe = float('-inf')
            for action in actions:
                if action == "Stop" : continue
                game = gamestate.generateSuccessor(agent_index,action)
                eval = self.minimax_ab(game, depth + 1, 1, a, b)
                maxe = max(maxe, eval)

                if(maxe > b): return maxe                                    # pruning
                a = max(a , maxe)

            return maxe
        
        else:
            mine = float('inf')
            for action in actions:
                if action == "Stop" : continue
                game = gamestate.generateSuccessor(agent_index,action)
                if(agent_index == gamestate.getNumAgents() - 1):
                    eval = self.minimax_ab(game, depth,0,a,b)
                else:
                    eval = self.minimax_ab(game, depth,agent_index + 1,a,b)
                mine = min(mine, eval)

                if(mine < a): return mine                                   # pruning
                b = min(b , mine)

            return mine

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        
        minimax_action = []
        eval2 = float('-inf')
        action2 = None
        actions = gameState.getLegalActions(0)                                              # generate possible actions for pacman
        for action in actions:
            if action == "Stop" : continue
            game = gameState.generateSuccessor(0,action)
            eval = self.expectimax(game,0,1)   
            minimax_action.append((eval,action))
        minimax_action.sort()
        minimax_value , action = minimax_action[len(minimax_action)-1]
        return action

    def expectimax(self, gamestate : GameState, depth, agent_index):
        if(gamestate.isLose() or gamestate.isWin()): return self.evaluationFunction(gamestate)   # terminal cases


        actions = gamestate.getLegalActions(agent_index) 

        if(agent_index == 0):                                                                    # max agent
            if(depth == self.depth - 1): return self.evaluationFunction(gamestate)
            maxe = float('-inf')
            for action in actions:
                if action == "Stop" : continue
                game = gamestate.generateSuccessor(agent_index,action)
                eval = self.expectimax(game, depth + 1, 1)
                maxe = max(maxe, eval)
            return maxe
        
        else:                                                                                    # min agents
            mine = 0
            for action in actions:
                if action == "Stop" : continue
                game = gamestate.generateSuccessor(agent_index,action)
                if(agent_index == gamestate.getNumAgents() - 1):                                 # check if there are other agents to visit or go to pacman 
                    eval = self.expectimax(game, depth,0)
                else:
                    eval = self.expectimax(game, depth,agent_index + 1)
                mine += eval 
            return mine/len(actions)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    we take into consideration:
        the current score
        if we win or if we lose
        if we are very close to the ghosts
        our distance from our nearest food
        the number of foods we have left 
        the remaining actions where we can eat the ghosts 
    """
    "*** YOUR CODE HERE ***"
    # worst case
    if currentGameState.isLose(): return(float('-inf'))      # if in this state we lose, return lowest number possible

    # best case
    if currentGameState.isWin(): return(float('inf'))        # if with this state we win, return highest number possible

    eval = currentGameState.getScore()

    # Useful information you can extract from a GameState (pacman.py)
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    
    # other cases
    ghosts = currentGameState.getGhostPositions()
    for ghost in ghosts:
        e = util.manhattanDistance(ghost,Pos)
        if e == 3: eval -= 1                                 # we don't lose but we are very close to ghosts   
        elif e == 2: eval -=2
        elif e == 1: eval -=3                            
         
    dist = []                                                # find the food closest to our position
    for food in Food.asList():
        dist.append(util.manhattanDistance(food,Pos))
    dist.sort()  

    eval -= dist[0]                                          # the less distance the better

    eval += sum(ScaredTimes)                                 # the more more left for ghosts to be edible the better

    return eval                                      

# Abbreviation
better = betterEvaluationFunction
