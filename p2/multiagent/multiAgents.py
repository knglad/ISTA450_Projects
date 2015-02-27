# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

from util import manhattanDistance
import util
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
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    " UNDER CONSTRUCTION "

    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """

    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)

    # (row, column)
    newPos = successorGameState.getPacmanPosition()

    # Boolean grid of if food is in a particular position
    foodGrid = currentGameState.getFood().asList()

    # Ghost information that we will need
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
    ghostDistances = [manhattanDistance(newPos, thisGhostsDist) for thisGhostsDist in ghostPositions]


    # Food information that we will also need
    activeFood = [manhattanDistance(newPos, food) for food in foodGrid if food]
    closestActiveFood = min(activeFood)

    # We want a larger number if the ghosts are far away, and if food is close.
        # Smaller if ghosts are closer and or food is farther away.
        # Looks like ghost dist / food dist properly shows this behavior



    modifier = 0

        # TEST sum of all (ghost dist / closest food dist)
    for ghostDist in ghostDistances:
      modifier += ghostDist / (closestActiveFood + 1)

    if newScaredTimes[0] > 0:
      modifier += ghostDist + closestActiveFood


        # TEST closest ghost dist / closest food dist
    #modifier += min(ghostDistances) / closestActiveFood

        # TEST sum of ghost dists / sum of food dists
    #for ghostDist in ghostDistances:
        #sumGhostDist += ghostDist
    #for foodDist in activeFood:
        #sumFoodDist += foodDist
    #return modifier += sumGhostDist / sumFoodDist


    return modifier


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

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    self.index = 0  # Pacman is always agent index 0
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """

    # We need to find our oppositions next best move, so we can predict it
    def miner(aState, currentDepth, numGhostLayer):
      # For clarity: numGhostLayer is the ghost number we are working on,
      # when building this function I thought about each ghost as a layer of successor moves in a tree

      # Helps add some reflex to the agent, without this we lost every time
      if aState.isWin() or aState.isLose():
        return self.evaluationFunction(aState)

      # Set our base minimum score very high, since we want the smallest
      m = 1000000.0

      # Again we loop through ALL possible successors at this depth
      # we want to find the smallest possible value (his best move)
      # and we will recursively call this function to look at all ghosts
      for ghostAction in aState.getLegalActions(numGhostLayer):
        # Compare lowest 'm' found with the next legal action
        # maxer moves the depth forward, and is what controls it
        if numGhostLayer == aState.getNumAgents() - 1:
          # We've checked all the ghosts recursively, NOW we want to get the next max
          m = min(m, maxer(aState.generateSuccessor(numGhostLayer, ghostAction), currentDepth))
        else:
          # This allows us to use the depth recursion to check EACH ghost
          m = min(m, miner(aState.generateSuccessor(numGhostLayer, ghostAction), currentDepth, numGhostLayer + 1))

      # We've looked at ALL the ghosts and found the smallest one
      return m


    # We need to get the max value of our possible moves
    def maxer(aState, currentDepth):
      futureDepth = currentDepth + 1

      # Stops the recursion, or the insanity before python breaks, also speeds it up just enough to win (I believe)
      if aState.isWin() or aState.isLose() or futureDepth == self.depth:
        return self.evaluationFunction(aState)

      # Default the max to be tiny, so anything is better
      m = -1000000.0

      # Look through all the possible actions of PACMAN
      # and compare the max possible next move from the ghosts 'm'
      for pacManAction in aState.getLegalActions(0):
        # Update 'm' possibly from the successors of all legal actions
        # '1' starts checking the smallest next move from the FIRST ghost
        # miner will recursively look at ALL ghosts and give me their best move
        m = max(m, miner(aState.generateSuccessor(0, pacManAction), futureDepth, 1))

      # We've looked through ALL possible successors currently
      return m


    # Okay now we use those helper functions
    allPacManLegalActions = gameState.getLegalActions(0)
    # I'm seeing a pattern in these saved variables...
    currMax = -1000000
    # Seriously, it fought me for ever until I facepalmed at what was wrong
    storedBestAction = 'see python, its initialized!'

    # Look at ALL actions pacman can take currently
    for action in allPacManLegalActions:
      # We always start at depth 0
      startDepth = 0
      # Get the BEST next move, maxer looks through the ghosts next moves for us
      bestMove = maxer(gameState.generateSuccessor(0, action), startDepth)

      # Obligatory update based on findings so far
      if currMax < bestMove:
        currMax = bestMove
        storedBestAction = action

    return storedBestAction



class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    alpha = -1000000.0
    beta = 1000000.0

    # We need to find our oppositions next best move, so we can predict it
    def miner(aState, currentDepth, numGhostLayer, a, b):
      # For clarity: numGhostLayer is the ghost number we are working on,
      # when building this function I thought about each ghost as a layer of successor moves in a tree


      if aState.isWin() or aState.isLose():
        return self.evaluationFunction(aState)

      # Set our base minimum score very high, since we want the smallest
      m = 1000000.0

      for ghostAction in aState.getLegalActions(numGhostLayer):
        if numGhostLayer == aState.getNumAgents() - 1:
          # We've checked all the ghosts recursively, NOW we want to get the next max
          m = min(m, maxer(aState.generateSuccessor(numGhostLayer, ghostAction), currentDepth, a, b))
        else:
          # This allows us to use the depth recursion to check EACH ghost
          m = min(m, miner(aState.generateSuccessor(numGhostLayer, ghostAction), currentDepth, numGhostLayer + 1, a, b))

        # Is our calculated 'm' smaller or atleast the same as our alpha value?
        if m <= a:
          return m
        # oh it isn't, well then lets see if we can update our beta value and try to speed it up next time.
        b = min(b, m)
      # We've looked at ALL the ghosts and found the smallest one
      return m


    # ========================================================================================
    # Nice Comment based spacer for ease of reading
    # =========================================================================================

    # We need to get the max value of our possible moves
    def maxer(aState, currentDepth, a, b):
      futureDepth = currentDepth + 1

      if aState.isWin() or aState.isLose() or futureDepth == self.depth:
        return self.evaluationFunction(aState)

      m = -1000000.0
      for pacManAction in aState.getLegalActions(0):
        m = max(m, miner(aState.generateSuccessor(0, pacManAction), futureDepth, 1, a, b))

        # is our calculated m or max value better than our beta? if so use it! otherwise
        # just set alpha if 'm' is higher
        if m >= b:
          return m
        a = max(m, a)
      # We've looked through ALL possible successors currently
      return m


    # Okay now we use those helper functions
    allPacManLegalActions = gameState.getLegalActions(0)
    # I'm seeing a pattern in these saved variables...
    currMax = -1000000
    # Seriously, it fought me for ever until I facepalmed at what was wrong
    storedBestAction = 'see python, its initialized!'

    # Look at ALL actions pacman can take currently
    for action in allPacManLegalActions:
      # We always start at depth 0
      startDepth = 0
      # Get the BEST next move, maxer looks through the ghosts next moves for us
      bestMove = maxer(gameState.generateSuccessor(0, action), startDepth, alpha, beta)

      # Obligatory update based on findings so far
      if currMax < bestMove:
        currMax = bestMove
        storedBestAction = action

    return storedBestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

