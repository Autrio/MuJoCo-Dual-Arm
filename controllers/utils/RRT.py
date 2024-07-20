import numpy as np
import math
from typing import Iterable,List
import mujoco
import time

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
        return rng.uniform(low=self.low, high=self.high,size=9)
    
    def computeDistance(self, state0,state1):
        return np.linalg.norm(state1-state0,axis=-1)
    
    def computeDistances(self,state0,state1):
        return self.computeDistance(state0,state1)
    
    def Interpolate(self,state0,state1,w):
        return state0 + (state1 - state0) * w
    
    def IsCollision(self,state):
        self.data.qpos[:9] = state
        mujoco.mj_forward(self.model,self.data)
        if(len(self.data.contact[:].geom1)!=0):
            if(self.data.contact[:].geom1[0]!=0):
                return True
        return False
    
class Goal:
    def __init__(self, goal, stateSpace: StateSpace, threshold, seed=None):
        self.goal = np.array(goal)
        self.stateSpace = stateSpace
        self.threshold = threshold
        self.rng = np.random.RandomState(seed)

    def sample(self):
        return self.goal
    
    def IsSatisfied(self,state):
        return self.stateSpace.computeDistance(state,self.goal) <= self.threshold
    

class GoalSpace(Goal):
    def sample(self):
        ind = self.rng.choice(len(self.goal))
        return self.goal[ind]
    
    def IsSatisfied(self, state):
        return np.any(self.stateSpace.computeDistances(state,self.goal)<=self.threshold)
    
    
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
    def __init__(self,stateSpace:StateSpace):
        self.stateSpace = stateSpace

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
        for startState in self.startStates:
            if self.checkStateValidity(startState):
                node = Node(startState)
                self.startTree.append(node)

        if len(self.startTree) == 0:
            if(self.startStateRange == 0.0):
                print("there are no valid initial states")
                return None
            
        for _ in range(self.startStateMaxTrials):
                offset = self.rng.uniform(
                    -self.startStateRange, self.startStateRange
                )
                nearbyStartState = startState + offset
                if self.checkStateValidity(nearbyStartState):
                    self.startTree.append(Node(nearbyStartState))

        if len(self.startTree) == 0:
            print("There are no valid (nearby) initial states!")
            self.status = "invalid start"
            return None
        
        for goalState in self.goal_iter:
            if self.checkStateValidity(goalState):
                node = Node(goalState)
                self.goalTree.append(node)
        
        if len(self.goalTree) == 0:
            print("There are no valid goal states!")
            self.status = "invalid goal"
            return None

        IsStartTree = False

        while not self.shouldTerminate():
            IsStartTree = not IsStartTree
            tree = self.startTree if IsStartTree else self.goalTree
            otherTree = self.goalTree if IsStartTree else self.startTree

            # Sample random state
            rstate = self.sample_uniform()

            # From current tree to other tree
            node, status = self.growTree(tree, rstate)

            # Try another random state to grow tree
            if status == "TRAPPED":
                continue

            # Attempt to connect trees
            otherNode, status = self.growTree(otherTree, node.state)
            while status == "ADVANCED":
                otherNode, status = self.growTree(
                    otherTree, node.state, nnode=otherNode
                )

            # If we connected the trees in a valid way
            if status == "REACHED":
                print("Find solution at %d steps", self.nIter)
                path = node.tracePath()[::-1] + otherNode.tracePath()
                if not IsStartTree:
                    path = path[::-1]
                self.status = "success"
                return path
        else:
            self.status = "failure"
            return []
        

    def checkStateValidity(self, state) -> bool:
        self.nIter += 1
        return not self.stateSpace.IsCollision(state)

    def shouldTerminate(self):
        return self.nIter >= self.maxIter

    def sample_uniform(self):
        return self.stateSpace.UniformSampling(self.rng)

    def getNearestNode(self, tree: List[Node], state) -> Node:
        node_states = [node.state for node in tree]
        dist = self.stateSpace.computeDistances(state, node_states)
        return tree[np.argmin(dist)]

    def growTree(self, tree, rstate, add_node=True, nnode=None):
        if nnode is None:
            # Find closest state in the tree
            nnode = self.getNearestNode(tree, rstate)
        nstate = nnode.state

        # Assume we can reach the state we go towards
        reach = True

        # Find state to add
        dstate = rstate
        dist = self.stateSpace.computeDistance(nstate, rstate)
        if dist > self.maxDist:
            dstate = self.stateSpace.Interpolate(nstate, rstate, self.maxDist / dist)
            reach = False

        is_valid = self.checkStateValidity(dstate)
        if not is_valid:
            return None, "TRAPPED"

        node = Node(dstate, parent=nnode)
        if add_node:
            tree.append(node)
        return node, ("REACHED" if reach else "ADVANCED")
    