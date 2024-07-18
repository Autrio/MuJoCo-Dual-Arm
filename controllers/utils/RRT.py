import numpy as np
import math
from typing import Iterable,List

class StateSpace:
    """
    Joint Space states of the robot
    """    
    def __init__(self,low,high):
        self.low = low
        self.high = high

    def InitModel(self,model,data):
        self.model = model
        self.data = data

    def UniformSampling(self,rng = np.random):
        return rng.uniform(low=self.low, high=self.high)
    
    def computeDistance(self, state0,state1):
        return np.linalg.norm(state1-state0,axis=-1)
    
    def computeDistances(self,state0,state1):
        return self.computeDistance(state0,state1)
    
    def Interpolate(self,state0,state1,w):
        return state0 + (state1 - state0) * w
    
    def IsCollision(self):
        if(len(self.data.contact[:].geom1)!=0):
            if(self.data.contact[:].geom1[0]!=0):
                return True
        return False
    
class Goal:
    def __init__(self, goal, state_space: StateSpace, threshold, seed=None):
        self.goal = np.array(goal)
        self.state_space = state_space
        self.threshold = threshold
        self.rng = np.random.RandomState(seed)

    def sample(self):
        return self.goal
    
    def IsSatisfied(self,state):
        return self.state_space.computeDistance(state,self.goal) <= self.threshold
    

class GoalSpace(Goal):
    def sample(self):
        ind = self.rng.choice(len(self.goal))
        return self.goal[ind]
    
    def IsSatisfied(self, state):
        return np.any(self.state_space.computeDistances(state,self.goal)<=self.threshold)
    
    
class Node:
    def __init__(self,state,parent=None):
        self.state = state
        self.parent = parent

    def tracePath(self):
        node = self
        path = []
        while(node is not None):
            path.append(node.state)
            node = node.parent

        return path


class RRT:
    """
    implementation of RRT-connect or bi-RRT algorithm
    """
    def __init__(self,state_space:StateSpace):
        self.stateSpace = state_space

    def setParams(self,startStates,goal_iter: Iterable, maxDist, maxIter,
                  startStateRange,startStateMaxTrials,seed=None):
        self.startStates = startStates
        self.goal_iter = goal_iter
        self.maxDist = maxDist
        self.maxIter = maxIter
        self.startStateRange = startStateRange
        self.startStateMaxTrials = startStateMaxTrials

        self.rng = np.random.RandomState(seed)
        self.startTree : List[Node] = []
        self.goalTree : List[Node] = []

        self.nIter = 0
        self.status = Node

    def solve(self):
        """
        """