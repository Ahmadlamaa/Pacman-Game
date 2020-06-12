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
import math
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        closest =  -math.inf
        for i in newFood.asList():
            x = util.manhattanDistance(newPos, i)
            if closest == -math.inf or closest >=x :
                closest = x
        score1 = 1/closest
        y,z = 0,1
        for j in successorGameState.getGhostPositions():
            x = util.manhattanDistance(newPos, j)
            z+=x
            if x==1 or x <1:
                y+=1
        score2 = y + 1/z

        return successorGameState.getScore() + score1 - score2

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        def getmin(s,i,x):
            a = s.getLegalActions(i)
            if len(a)==0:
                return self.evaluationFunction(s)
            if i==s.getNumAgents() -1:
                return min(getmax(s.generateSuccessor(i,k),x) for k in a)
            else:
                return min(getmin(s.generateSuccessor(i,k),i+1,x) for k in a)

        def getmax(s,x):
            a = s.getLegalActions(0)
            if len(a)==0 or x == self.depth:
                return self.evaluationFunction(s)
            return max(getmin(s.generateSuccessor(0,k),1,x+1) for k in a)

        return max(gameState.getLegalActions(0), key=lambda action: getmin(gameState.generateSuccessor(0,action),1,True))

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """


    def getmin(self, s, i, x, a, b):
        k = math.inf
        action = Directions.STOP
        actions = s.getLegalActions(i)

        if s.isLose() or len(actions)==0:
            return self.evaluationFunction(s), Directions.STOP


        for m in actions:
            s2 = s.generateSuccessor(i, m)
            if i == s.getNumAgents() - 1:
                c = self.getmax(s2, x + 1, a, b)[0]
            else:
                c = self.getmin(s2, i + 1, x, a, b)[0]

            if k > c:
                k = c
                action = m
            if a > k:
                return k, action
            b = min(b, k)

        return k, action

    def getmax(self,s,x,a,b):
        k = -math.inf
        action = Directions.STOP
        actions = s.getLegalActions(0)
        if s.isWin() or x > self.depth or len(actions)==0:
	           return self.evaluationFunction(s), Directions.STOP
        for m in actions:
            s2 = s.generateSuccessor(0, m)
            c = self.getmin(s2, 1, x, a, b)[0]
            if c > k:
                k = c
                action = m
            if k > b:
                return k, action
            a = max(a, k)
        return k, action


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"


        return self.getmax(gameState, 1, -math.inf, math.inf)[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def getexp(self,s,i,x):
        k = 0
        if s.isLose() or s.isWin():
            return self.evaluationFunction(s)
        p = 1/len(s.getLegalActions(i))
        for action in s.getLegalActions(i):
            if action == Directions.STOP:
                continue
            next = s.generateSuccessor(i, action)
            temp = self.getexpectimax(next, i+1, x)
            k += (temp * p)
            actionValue = action
        return k


    def getmax(self, s, i, x):

      if s.isWin() or s.isLose():
        return self.evaluationFunction(s)

      actionValue= "Stop"
      k=-math.inf

      for action in s.getLegalActions(i):

        if action == Directions.STOP:
          continue

        next = s.generateSuccessor(i, action)
        temp = self.getexpectimax(next, i+1, x)

        if temp > k:
          k = temp
          actionValue = action

      if x == 0:
        return actionValue
      else:
        return k


    def getexpectimax(self,s,i,x):
        n = s.getNumAgents()
        if n < i+1:
            x+=1
            i=0
        if x == self.depth:
            return self.evaluationFunction(s)
        if i!=self.index:
            return self.getexp(s, i, x)
        else:
            return self.getmax(s, i, x)
        return 0




    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        return self.getexpectimax(gameState,0,0)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    The returned score is positively related to:
    1- The distance to the nearest ghost(if present wihtin a Manhattan distance of 20) (x)
    2- Durations during which the ghosts are scared of pacman (times)
    3- The number of scared ghosts(scared)
    4- The state's default score evaluation(score)
    And is negatively related to:
    1- Distance to the nearest food pellet(y)
    2- Number of ghosts (count)
    3- Number of food pellets remaining (count2)
    4- Average distance to all ghosts (avg)
    """
    "*** YOUR CODE HERE ***"

    position = currentGameState.getPacmanPosition()
    gs = currentGameState.getGhostStates()
    food = currentGameState.getFood().asList()
    scare_duration = [ghostState.scaredTimer for ghostState in gs]
    gp = currentGameState.getGhostPositions()

    x = math.inf
    scared = 0
    times = 1
    for time in scare_duration:
      times += time
      if time > 2:
        scared+=1

    count = 0
    avg = 1
    for ghost in gp:
      d = manhattanDistance(ghost, position)
      avg += d
      if d <= 20:
        count+=1
      if d<x:
          x = d
    avg =avg/len(gp)
    y = math.inf

    for f in food:
      d = manhattanDistance(f, position)
      if d<y:
          y = d
    y+=1
    count2 = len(food) + 1
    if x>1:
        x = 0
    else:
        x = -math.inf

    score = currentGameState.getScore() + 1
    return x + score - 1/times  + 20*scared+ 1/y  + 10/count2 + 1/avg - 20*count

    util.raiseNotDefined()

# Abbreviation #not TOO helpful
better = betterEvaluationFunction
