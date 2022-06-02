"""Mixture model for collaborative filtering"""
from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

up = 0
down = 1
left = 2
right = 3

numCol = 4
numRow = 3

not_valid = -100
coord = [up, down, left, right]
cost = -0.04

class StateEntry:
    def __init__(self, a):
        self.T = np.ones(4)
        self.Q = np.zeros((2, 4, 4))
        self.a = a
        self.R = cost
        self.currentQ = 0
        ## compile T
        for i in coord:
            if (a[i] == not_valid):
                self.T[i] = 0


def updateQ(stateNum, movement):
    s = stateList[stateNum]
    if (s.currentQ % 2 == 0):
        Q_old = s.Q[0]
        Q_new = s.Q[1]
    new_state = stateList[s.a[movement]]
    if (new_state != not_valid):
        temp = s.R + gamma * new_state.Q

stateList = []
# first row
stateList.append(StateEntry(np.array([not_valid, 4, not_valid, 1])))
stateList.append(StateEntry(np.array([not_valid, not_valid, 0, 2])))
stateList.append(StateEntry(np.array([not_valid, 6, 1, 3])))
stateList.append(StateEntry(np.array([not_valid, 7, 2, not_valid])))
# 2nd row
stateList.append(StateEntry(np.array([0, 8, not_valid, not_valid])))
stateList.append(StateEntry(np.array([not_valid, not_valid, not_valid, not_valid])))
stateList.append(StateEntry(np.array([2, 10, not_valid, 7])))
stateList.append(StateEntry(np.array([3, 11, 6, not_valid])))
# 3rd row
stateList.append(StateEntry(np.array([4, not_valid, not_valid, 9])))
stateList.append(StateEntry(np.array([not_valid, not_valid, 8, 10])))
stateList.append(StateEntry(np.array([6, not_valid, 9, 11])))
stateList.append(StateEntry(np.array([7, not_valid, 10, not_valid])))

stateList[3].R = stateList[3].R + 1
stateList[7].R = stateList[7].R - 1

print('completed')
