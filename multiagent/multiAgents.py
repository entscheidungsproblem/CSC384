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
import random, util, sys

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
        oldFoodScores = distanceToFood(currentGameState)
        newFoodScores = distanceToFood(successorGameState)
        oldGhostScores = distanceToGhosts(currentGameState)
        newGhostScores = distanceToGhosts(successorGameState)

        EAT_FOOD = 10
        EAT_GHOST = 200
        TIME_PENALTY = 1
        EPSILON = 0.001

        # End game
        if successorGameState.isLose() or newGhostScores[0][0] == 1.0:
            return -sys.maxint - 1
        if successorGameState.isWin():
            return sys.maxint

        # Score starts as difference of states scores
        score = successorGameState.getScore() - currentGameState.getScore()

        # Not moving penalty
        if action == Directions.STOP:
            score -= 1000*TIME_PENALTY

        # Ghost calculation
        ghostScore = 0.0
        # If there are ghosts
        if len(newGhostScores) > 0:
            closestGhost = newGhostScores[0]
            oldClosestGhost = oldGhostScores[0]

            # If the closest ghost is scared
            if newScaredTimes[closestGhost[1]] > 0:
                # Try to eat
                if newGhostScores[0][0] == 1.5:
                        ghostScore += 2.5*EAT_GHOST
                # If the ghost is closer and reachable 
                elif oldClosestGhost[0] < closestGhost[0] and closestGhost[0] < newScaredTimes[closestGhost[1]]:
                    change = 2/(oldClosestGhost[0] - closestGhost[0])
                    ghostScore += change
            else:

                if successorGameState.getCapsules() < currentGameState.getCapsules():
                    cap = EAT_FOOD/closestGhost[0]
                    ghostScore += cap

                # If the ghost is further away 
                if oldClosestGhost[0] > closestGhost[0]:
                    change = 1/(oldClosestGhost[0] - closestGhost[0])
                    ghostScore += change
        score += ghostScore
                
        # Food calculation
        closestFood = newFoodScores[0]
        oldClosestFood = oldFoodScores[0]
        foodScore = 0
        # General food distance calculation
        for food in newFoodScores:
            foodScore += 1/(food[0] + EPSILON)
        for food in oldFoodScores:
            foodScore -= 1/(food[0] + EPSILON)
        # Min food distance
        if oldClosestFood[0] > closestFood[0]:
            foodScore += 5*(oldClosestFood[0] - closestFood[0])
        #print foodScore
        score += foodScore

        return score


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

    _min = -sys.maxint - 1
    _max = sys.maxint

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isTerminal(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == self.depth*gameState.getNumAgents()

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
        """
        "*** YOUR CODE HERE ***"
        # Default bests
        bestMove = None
        bestValue = self._min
        # For each current possible action
        for action in gameState.getLegalActions(self.index):
            nxtState = gameState.generateSuccessor(self.index, action)
            # Run the ghosts starting at depth 1
            # ASSUMES THERE IS AT LEAST 1 GHOST!!!
            potential = self.DFMiniMax(nxtState, 1, 1)
            if potential > bestValue:
                bestValue = potential
                bestMove = action
        return bestMove

    def DFMiniMax(self, gameState, agentIndex, depth=0):
        #print agentIndex, self.index
        best_move = None
        #if self.isTerminal(gameState, depth) and agentIndex == self.index:
        if self.isTerminal(gameState, depth):
            x = self.evaluationFunction(gameState)
            #print x
            return x
    
        value = self._max
        if agentIndex == self.index:
            value = self._min

        for action in gameState.getLegalActions(agentIndex):
            nxtState = gameState.generateSuccessor(agentIndex, action)

            # If pacman turn
            if self.index == agentIndex:
                nxt_val = self.DFMiniMax(nxtState, agentIndex + 1, depth + 1)
                if value < nxt_val:
                    value = nxt_val
            # Ghost turn
            else:
                # If last ghost, next turn is pacmans
                if agentIndex == gameState.getNumAgents() - 1:
                    nxt_val = self.DFMiniMax(nxtState, self.index, depth + 1)
                else:
                    nxt_val = self.DFMiniMax(nxtState, agentIndex + 1, depth + 1)

                if value > nxt_val:
                    value = nxt_val

        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
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
        """

        # Default bests
        bestMove = None
        bestValue = self._min

        alpha = self._min
        beta = self._max
        # For each current possible action
        for action in gameState.getLegalActions(self.index):
            nxtState = gameState.generateSuccessor(self.index, action)
            # Run the ghosts starting at depth 1
            # ASSUMES THERE IS AT LEAST 1 GHOST!!!
            potential = self.DFMiniMax(nxtState, 1, alpha, beta, 1)
            #print potential, bestValue, beta
            if potential > bestValue:
                bestValue = potential
                bestMove = action
            if potential >= beta:
                return bestMove
            alpha = max(alpha, potential)
            #beta = min(beta, bestValue)
        #print bestMove
        return bestMove

    def DFMiniMax(self, gameState, agentIndex, alpha, beta, depth=0):
        #print agentIndex, self.index
        #if self.isTerminal(gameState, depth) and agentIndex == self.index:
        if self.isTerminal(gameState, depth):
            x = self.evaluationFunction(gameState)
            #print x
            return x
    
        value = self._max
        if agentIndex == self.index:
            value = self._min

        for action in gameState.getLegalActions(agentIndex):
            nxtState = gameState.generateSuccessor(agentIndex, action)

            # If pacman turn
            if self.index == agentIndex:
                nxt_val = self.DFMiniMax(nxtState, agentIndex + 1, alpha, beta, depth + 1)
                if value < nxt_val:
                    value = nxt_val
                if value >= beta: return value
                #if value >= beta: return value
                alpha = max(alpha, value)
            # Ghost turn
            else:
                # If last ghost, next turn is pacmans
                if agentIndex == gameState.getNumAgents() - 1:
                    nxt_val = self.DFMiniMax(nxtState, self.index, alpha, beta, depth + 1)
                else:
                    nxt_val = self.DFMiniMax(nxtState, agentIndex + 1, alpha, beta, depth + 1)

                if value > nxt_val:
                    value = nxt_val
                if value <= alpha: return value
                beta = min(beta, value)
        return value


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
        bestMove = None
        bestValue = self._min

        for action in gameState.getLegalActions(self.index):
            nxtState = gameState.generateSuccessor(self.index, action)
            # Run the ghosts starting at depth 1
            # ASSUMES THERE IS AT LEAST 1 GHOST!!!
            potential = self.Expectimax(nxtState, 1, 1)
            if potential > bestValue:
                bestValue = potential
                bestMove = action
        return bestMove

    def Expectimax(self, gameState, agentIndex, depth=0):
        # print agentIndex, self.index
        # if self.isTerminal(gameState, depth) and agentIndex == self.index:
        if self.isTerminal(gameState, depth):
            x = self.evaluationFunction(gameState)
            # print x
            return x

        value = 0
        if agentIndex == self.index:
            value = self._min

        for action in gameState.getLegalActions(agentIndex):
            nxtState = gameState.generateSuccessor(agentIndex, action)

            # If pacman turn eg player == MAX
            if self.index == agentIndex:
                nxt_val = self.Expectimax(nxtState, agentIndex + 1, depth + 1)
                if value < nxt_val:
                    value = nxt_val
                # if value >= beta: return value
            # Ghost turn
            else:
                # If last ghost, next turn is pacmans
                if agentIndex == gameState.getNumAgents() - 1:
                    nxt_val = self.Expectimax(nxtState, self.index, depth + 1)
                else:
                    nxt_val = self.Expectimax(nxtState, agentIndex + 1, depth + 1)

                value = value + (2/len(gameState.getLegalActions(agentIndex))) * nxt_val
        return value






def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: What I did in this question was set up a weighted reward system to tell Pacman what he should
      prioritize. I then used those weights in different scenarios splitting them up into value modifiers that involve
      food and ghosts (since eating ghosts and food are the only two ways Pacman can attain points. Once all the
      modifiers are taken into account return the value for that instance and keep an ongoing count till the stage
      is complete.
    """

    # We get the position of Pacman, the food on the board and position of the ghosts
    presentPos = currentGameState.getPacmanPosition()
    allfood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    # Set up a weighted reward system for Pacman to follow.
    REWARD_FOOD = 10.0
    REWARD_GHOST = 10.0
    REWARD_SCARED_GHOST = 100.0

    # current score
    value = currentGameState.getScore()

    # Winning should be the ultimate reward and losing be the ultimate penalty
    if currentGameState.isWin():
        return sys.maxint
    if currentGameState.isLose():
        return -sys.maxint

    # Score for ghost stuff
    valueGhost = 0
    for ghost in ghostStates:
        dist = manhattanDistance(presentPos, ghostStates[0].getPosition())
        if dist > 0:
            if ghost.scaredTimer > 0:
                valueGhost += REWARD_SCARED_GHOST / dist
            else:
                valueGhost -= REWARD_GHOST / dist
    value += valueGhost

    # score for food stuff
    foodDist = [manhattanDistance(presentPos,food) for food in allfood.asList()]
    if len(foodDist):
        value += REWARD_FOOD / min(foodDist)

    return value




# Abbreviation
better = betterEvaluationFunction

def distanceToFood(gameState):
    pos = gameState.getPacmanPosition()
    food = gameState.getFood().asList()
    scores = [(manhattanDistance(pos, food[i]), i) for i in range(len(food))]
    return sorted(scores)

def distanceToGhosts(gameState):
    pos = gameState.getPacmanPosition()
    ghosts = [ghost.getPosition() for ghost in gameState.getGhostStates()]
    distances = [()]
    scores = [(manhattanDistance(pos, ghosts[i]), i) for i in range(len(ghosts))]
    return sorted(scores)
