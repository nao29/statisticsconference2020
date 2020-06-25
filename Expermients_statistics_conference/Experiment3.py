# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:16:05 2019

@author: nao
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import copy
from scipy.stats import entropy

######################constant values######################################################
CONDITION = 4
D_MAX_ARR = np.array([7,8,9,10])#MaxDepth of MetaTree
B_ARR = np.array([10,10,10,10])
D_MAX_TRUE = 10#MaxDepth of true model
K = 20
N = 1500
THETA = 50
M =  100#the number of generating true model
TEST = 100
BRANCH_NUM = 2
Y_VALUE = 2
N_WIDTH = 100
BETA = np.ones(Y_VALUE) / Y_VALUE
######################木構造######################################################
class Node:
    def __init__(self, depth):
        self.childs = [None for i in range(BRANCH_NUM)]
        self.depth = depth
        self.feat = -1
        self.division_index_list = [0 for i in range(K - self.depth)]
        self.g = 1/2#division probability on each branch
        self.n = np.zeros(Y_VALUE)   
        self.P = 1 / 2#probability of y=0
        self.q = 1 / 2
        self.theta = np.ones(Y_VALUE) / Y_VALUE
        self.division = -1  #0:division 1:no division

    ######################generate true tree######################################################
    def make_true_tree(self, depth, flist):#true tree
        if depth < D_MAX_TRUE:
            self.feat = np.random.choice(flist, 1)[0]#select feature values at random from flist
            flist_copied = flist.copy()
            flist_copied.remove(self.feat)
            self.division = np.random.binomial(1, self.g, 1)#determine whether division ocuurs according with self.g
        else:
            self.g = 1    
            self.division = 1
    
        if self.division != 1:
            for branch in range(BRANCH_NUM):
                self.childs[branch] = Node(depth)
                self.childs[branch].make_true_tree(depth+1, flist_copied)
                
    def make_theta(self, depth):
        if self.division == 1:
            self.theta = np.random.dirichlet(BETA,1)[0]
        else:
            for branch in range(BRANCH_NUM):
                self.childs[branch].make_theta(depth+1)

    ######################division index################################################
    def division_index_cal(self, data, flist):#data is the tuple of xy
        for i in range(len(flist)):
            for branch in range(BRANCH_NUM):
                data_selected = np.delete(data.copy(), np.where(data[flist[i]][:] != branch), axis=1)#delete the rows except for x_i=branch
                if data_selected.shape[1] == 0:#if no data
                    self.division_index_list[i] += np.log2(data.shape[1])
                else:
                    self.division_index_list[i] += data_selected.shape[1] * entropy([np.count_nonzero(data_selected[-1][:] == y_value) /data.shape[1]  for y_value in range(Y_VALUE)], base=2) / data.shape[1]
        self.feat = flist[np.argmin(self.division_index_list)]
                            
    ######################generate trees for classification##########
    def make_tree_random_feat(self,depth, flist, condition):
        if depth < D_MAX_ARR[condition]:
            self.feat = np.random.choice(flist, 1)[0]#select feature values at random from flist
            flist_copied = flist.copy()
            flist_copied.remove(self.feat)
            for branch in range(BRANCH_NUM):
                self.childs[branch] = Node(depth+1)
                self.childs[branch].make_tree_random_feat(depth+1, flist_copied, condition)

    ######################with renewal of poseterior distribution###############################################
    def q_cal(self, y_value):
        return (self.n[y_value] + BETA[y_value]) / (self.n.sum() + np.sum(BETA))
    
    def n_add(self, data):
        self.n[data[-1]] += 1
    
    def g_cal(self,data, condition):
        if self.depth < D_MAX_ARR[condition]:
            self.g *= self.childs[data[self.feat]].P / self.P

    def P_cal_with_update(self, data, condition):
        if self.depth == D_MAX_ARR[condition]:
            self.P = self.q_cal(data[-1])
        else:
            self.P = (1 - self.g) * self.q_cal(data[-1]) + self.g * self.childs[data[self.feat]].P_cal_with_update(data, condition)
        self.g_cal(data, condition)
        self.n_add(data)    
        return self.P

    ######################only classification######################################################
    def P_cal(self,data_x, condition, y_value):
        if self.depth == D_MAX_ARR[condition]:
            tmp_P = self.q_cal(y_value)
        else:
            tmp_P = (1 - self.g) * self.q_cal(y_value) + self.g * self.childs[data_x[self.feat]].P_cal(data_x, condition, y_value)
        return tmp_P

######################generate data######################################################
def y_decide(node, data):
    if node.division== 1:
        theta = node.theta
    elif node.division != 1:
        theta = y_decide(node.childs[data[node.feat]], data)
    return theta

######################Feature Trees######################################################
class Feature_trees:
    def __init__(self, condition, train_data):
        self.root_list = [None for i in range(B_ARR[condition])]
        for i in range(B_ARR[condition]):
            self.root_list[i] = Node(0)
            self.root_list[i].make_tree_random_feat(0, [i for i in range(K)], condition)
        self.posterior = np.ones(B_ARR[condition]) / B_ARR[condition]

    ######################renew posterior distribution#####################################################
    def posterior_cal(self,data, condition):
        for i in range(B_ARR[condition]):
            self.root_list[i].P_cal_with_update(data, condition)
            self.posterior[i] *= self.root_list[i].P
        self.posterior /= self.posterior.sum()
    
    ######################classification######################################################
    def prediction(self,data, condition):
        tmp = np.zeros(Y_VALUE)
        for y_value in range(Y_VALUE):
            for i in range(B_ARR[condition]):
                tmp[y_value] += self.posterior[i] * self.root_list[i].P_cal(data, condition, y_value)
        return np.argmax(tmp)

######################main######################################################
correct = np.zeros((CONDITION, int(N/N_WIDTH)), dtype = 'int')
incorrect = np.zeros((CONDITION, int(N/N_WIDTH)), dtype = 'int')
for m in range(M):
    print('m= %s' % m)
    #generate true tree
    true_root = Node(0)
    true_root.make_true_tree(0, [i for i in range(K)])
    for theta in range(THETA):
        true_root.make_theta(0)    
        #generate data
        train_data = np.array([[random.randint(0, BRANCH_NUM - 1) for j in range(N)] for i in range(K+1)],dtype = 'int')
        test_data = np.array([[random.randint(0, BRANCH_NUM - 1) for j in range(TEST)] for i in range(K+1)],dtype = 'int')
        for j in range(N):
            train_data[-1,j] = np.where(np.random.multinomial(1, y_decide(true_root, train_data[:,j])) == 1)[0][0]
        for j in range(TEST):
            test_data[-1,j] = np.where(np.random.multinomial(1, y_decide(true_root, test_data[:,j])) == 1)[0][0]

        #generate trees for classification
        feature_trees = Feature_trees(-1, train_data) #condition=-1
        for condition in range(CONDITION):
            feature_trees_deepcopy = copy.deepcopy(feature_trees)
            for n in range(N_WIDTH, N + 1, N_WIDTH):#10,20,30,...
                #renew posterior distribution
                for j in range(n - N_WIDTH, n):
                    feature_trees_deepcopy.posterior_cal(train_data[:,j], condition)
                    
                #classification
                for j in range(TEST):
                    if feature_trees_deepcopy.prediction(test_data[:,j], condition) == test_data[-1,j]:
                        correct[condition][int(n/N_WIDTH)-1] += 1
                    else:
                        incorrect[condition][int(n/N_WIDTH)-1] += 1

print('correct= %s' % correct)
print('incorrect= %s' % incorrect)
num = np.zeros(int(N/N_WIDTH), dtype = 'int')
error_rate = np.zeros((CONDITION, int(N/N_WIDTH)), dtype = 'float')
for n in range(N_WIDTH, N + 1, N_WIDTH):
    num[int((n-1)/N_WIDTH)] = n
for condition in range(CONDITION):
    for n in range(N_WIDTH, N + 1, N_WIDTH):
        error_rate[condition][int((n-1)/N_WIDTH)] = incorrect[condition][int((n-1)/N_WIDTH)] / (THETA * TEST * M)
    error_rate_list = error_rate[condition].tolist()
    print(error_rate_list)
    fig, ax = plt.subplots()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot(num,error_rate_list)
                         