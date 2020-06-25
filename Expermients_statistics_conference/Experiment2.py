# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:16:05 2019

@author: nao
"""

import numpy as np
import random
from scipy.stats import entropy

######################constant values######################################################
CONDITION = 1
D_MAX_ARR = np.array([2])#MaxDepth of MetaTree
B_ARR = np.array([3])#fix
D_MAX_TRUE = 2#MaxDepth of true model
K = 10
N = 200
THETA = 50
M =  100
TEST = 100
BRANCH_NUM = 2
Y_VALUE = 2
BETA = np.ones(Y_VALUE) / Y_VALUE
B_feat = np.array([[0,2,3],[3,4,5],[4,5,6]])#the order of the feature values on the true model
######################tree structure######################################################
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
        self.division = 0  #（0:divide,１:not devide） 

                
    def make_true_tree_biased_feat_fix(self, depth):#make complete tree
        if depth < D_MAX_TRUE:
            self.feat = self.depth#fix
        else:
            self.g = 1    
            self.division = 1
    
        if self.division != 1:
            for branch in range(BRANCH_NUM):
                self.childs[branch] = Node(depth)
                if branch != 0:
                    self.childs[branch].division = 1
                self.childs[branch].make_true_tree_biased_feat_fix(depth+1)


    def make_theta(self, depth):
        if self.division == 1:
            self.theta = np.random.dirichlet(BETA,1)[0]
        else:
            for branch in range(BRANCH_NUM):
                self.childs[branch].make_theta(depth+1)
    ######################division index################################################
    def division_index_cal(self, data, flist):#data represents the tuple of xy
        for i in range(len(flist)):
            for branch in range(BRANCH_NUM):
                data_selected = np.delete(data.copy(), np.where(data[flist[i]][:] != branch), axis=1)#delete the rows except for x_i=branch
                if data_selected.shape[1] == 0:#if no data
                    self.division_index_list[i] += np.log2(data.shape[1])
                else:
                    self.division_index_list[i] += data_selected.shape[1] * entropy([np.count_nonzero(data_selected[-1][:] == y_value) /data.shape[1]  for y_value in range(Y_VALUE)], base=2) / data.shape[1]
        self.feat = flist[np.argmin(self.division_index_list)]
                            
    ######################make tree for classification##########
    def make_tree_fix(self,depth, i, condition, feat):
        if depth < D_MAX_ARR[condition]:            
            if self.depth == 0:
                self.feat = B_feat[i][0]
            else:
                self.feat = feat

            for branch in range(BRANCH_NUM):
                self.childs[branch] = Node(depth+1)
                if branch == 0:
                    feat_depth1 = B_feat[i][1]
                else:
                    feat_depth1 = B_feat[i][2]
                self.childs[branch].make_tree_fix(depth+1, i, condition, feat_depth1)

    ######################with posterior distribution renewal###############################################
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
            self.root_list[i].make_tree_fix(0, i, condition, 0)
        self.posterior = np.ones(B_ARR[condition]) / B_ARR[condition]

    ######################renew posterior distribution######################################################
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

##############copy parameters on tree############################################
def feature_trees_copy_node(node,node2,condition):
    node2.feat = node.feat
    if node.depth < D_MAX_ARR[condition] :
        for branch in range(BRANCH_NUM):
            feature_trees_copy_node(node.childs[branch], node2.childs[branch],condition)


######################main######################################################
correct = np.zeros((CONDITION, int(N/10)), dtype = 'int')
incorrect = np.zeros((CONDITION, int(N/10)), dtype = 'int')
posterior_average = np.zeros((int(N/10), (B_ARR[0])), dtype = 'float')
for m in range(M):
    print('m= %s' % m)
    #generate true tree
    true_root = Node(0)
    true_root.make_true_tree_biased_feat_fix(0)
    for theta in range(THETA):
        true_root.make_theta(0)    
        #generate data
        train_data = np.array([[random.randint(0, BRANCH_NUM - 1) for j in range(N)] for i in range(K+1)],dtype = 'int')
        test_data = np.array([[random.randint(0, BRANCH_NUM - 1) for j in range(TEST)] for i in range(K+1)],dtype = 'int')
        for j in range(N):
            train_data[-1,j] = np.where(np.random.multinomial(1, y_decide(true_root, train_data[:,j])) == 1)[0][0]
        for j in range(TEST):
            test_data[-1,j] = np.where(np.random.multinomial(1, y_decide(true_root, test_data[:,j])) == 1)[0][0]

        #generate tree for classification
        feature_trees = Feature_trees(-1, train_data) #condition=-1

        for condition in range(CONDITION):
            for n in range(10, N + 1, 10):#10,20,30,...
                #renew poseterior distribution
                for j in range(n - 10, n):
                    feature_trees.posterior_cal(train_data[:,j], condition)
                    
                for i in range(B_ARR[0]):
                    posterior_average[int(n/10)-1][i] += feature_trees.posterior[i]

for i in range(B_ARR[0]):
    for n in range(int(N/10)):
        posterior_average[n][i] /= (THETA * M)
list0 =posterior_average.T[0,:].tolist()
list1 =posterior_average.T[1,:].tolist()
list2 =posterior_average.T[2,:].tolist()
print(list0)
print(list1)
print(list2)
