#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:50:06 2021

@author: fatihdurmaz
"""
import random
import pandas as pd
import numpy as np
import time
import itertools
from scipy.stats import sem, t
#%%
"""
        HEURISTIC ALGORITHM
        GREEDY MINSETCOVER
"""
def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(set(subset))
        covered |= subset
    return cover
#%%
"""
        RANDOM INPUT GENERATOR 
        FOR THE GREEDY ALGORITHM
"""
def random_subset(s,D_size):
    D = set(random.sample(s, k=D_size))
    return D

def rand_input(S_size,C_size,D_size):
    subsets=[]
    universe = set(random.sample(range(1,S_size*3), k=S_size))
    
    for i in range(1,C_size+1):
        subset = random_subset(universe,D_size)
        if len(subset) != 0:
            subsets.append(subset)
    return universe, subsets

#%%

"""
        RANDGEN -> GREEDY
        THIS FUNCTION RUNS THE ALGORITHM
"""
def main(S_size,C_size,D_size):
    universe,subsets = rand_input(S_size,C_size,D_size) 
    cover = set_cover(universe, subsets)
    return cover
    #print("#####################\n\n",cover,"\n\n#####################\n\n",universe,"\n\n#####################\n\n")
    
#%%
"""
        COST COMPUTATION LOOP
"""
log = {}
count =0
for j in range(1,11):
    log[j*50] = []
    i=0
    while (i<1000):
        
        start = time.time()
        cover = main(1000,1000,50*j) 
        stop = time.time()
        cost = stop - start
        count +=1
        if((count %100) == 0):
            print(count,"\n")
        if cover != None:
            i += 1;
            log[j*50].append(cost) #seconds
#%%
"""
        DATA MINING
"""
data = pd.DataFrame.from_dict(log).swapaxes(0,1)
mean = data.mean(numeric_only=True, axis=1)
std = data.std(axis=1)
data["Mean (sec.)"] = mean
data[ "STD"] = std;
data.reset_index(inplace=True)
data = data.rename(columns = {'index':'Size(D)'})
data = data[['Size(D)', 'Mean (sec.)',"STD"]]
data["Sm"] = (data["STD"]/np.sqrt((data['Size(D)'])))
data["h_95"] =  (data["Sm"] * t.ppf((1 + 0.95) / 2, data["Size(D)"] - 1))
data["h_90"] = (data["Sm"] * t.ppf((1 + 0.90) / 2, data["Size(D)"] - 1))
data.plot.line(x="Size(D)", y="Mean (sec.)")

#%%

"""
        ISCOVER CHECKS IF
        IT IS COVER 
        FOR THE SAKE OF CORRECTNESS
        MAIN_TEST GUARANTEES CORRECT INPUT
"""
def isCover(universe,Cover):
    for subset in Cover:
            universe = universe - subset
    if len(universe) == 0:
        return True
    return False
def main_test(S_size,C_size,D_size):
    universe,subsets = rand_input(S_size,C_size,D_size) 
    while(isCover(universe,subsets) != True):
        universe,subsets = rand_input(S_size,C_size,D_size) 
    cover = set_cover(universe, subsets)
    return isCover(universe,cover)
#%%
"""
        CHECKING THE COVER CORRECTNESS
"""
count = 0
for i in range(1,10000):
    value = main_test(100,100,20)
    count +=1
    if((count %100) == 0):
        print(count,"\n")
    if(value == False):
        print("false")
        
#%%
"""
        SELF EXPLANATORY BRUTEFORCE
"""
def brute_force(subsets_dict, universal_set, upper_bound=None):
    """
    Brute forces all combinations of subsets of size < upper_bound to find the exact solution.
    """
    for size in range(1, upper_bound+1):
        keys_of_subsets_to_test = itertools.combinations(subsets_dict,size)
        for keys_ls in keys_of_subsets_to_test:
            covered = set().union(*[key for key in keys_ls])
            if covered == universal_set:
                return keys_ls
    return []
#%%
"""
        REAL CORRECTNESS TESTING FUNCTION
"""
total= 0
correct = 0
minsum =0
greedysum = 0
def test(S_size,C_size,D_size):
    global total
    global correct
    global minsum
    global greedysum
    universe,subsets = rand_input(S_size,C_size,D_size) 
    while(isCover(universe,subsets) != True):
        universe,subsets = rand_input(S_size,C_size,D_size) 
    approx_sol = set_cover(universe, subsets)
    min_sol = brute_force(subsets, universe, len(approx_sol))
    total +=1
    print(len(approx_sol),len(min_sol))
    if (len(approx_sol) == len(min_sol)):
        correct+=1
    return len(approx_sol),len(min_sol)
#%%
"""
        TESTING THE CORRECTNESS VIA TEST
"""
dfg = []
dfm = []
for i in range(1,101):
    greedysol,minsol = test(15,15,2)
    dfg.append(greedysol)
    dfm.append(minsol)
    
#%%
"""

            THIS PART TESTS RATIO BOUND AND GIVES A DATAFRAME

"""
import math
total= 0
correct = 0
minsum =0
greedysum = 0

iscover_count = 0
def test(S_size,C_size,D_size):
    global total
    global correct
    global minsum
    global greedysum
    global iscover_count
    iscover_count=0
    universe,subsets = rand_input(S_size,C_size,D_size) 
    while(isCover(universe,subsets) != True):
        iscover_count+=1
        #if((iscover_count %10) == 0):
            #print(iscover_count,"\n")
        universe,subsets = rand_input(S_size,C_size,D_size) 
    approx_sol = set_cover(universe, subsets)
    min_sol = brute_force(subsets, universe, len(approx_sol))
    total +=1
    minsum += len(min_sol)
    greedysum += len(approx_sol)
    #print(len(approx_sol),len(min_sol))
    #print("Approximate solution --> ",approx_sol,"\nPerfect solution --> ",min_sol) 
    if (len(approx_sol) == len(min_sol)):
        correct+=1
    return len(approx_sol),len(min_sol)
log = []


numberOfTestst =[100,500,1000]
Ssizet = [5,10,15]
Csizet = [15,20]
Dsizet = [2,5]
for numberOfTests in numberOfTestst:
    #print("#################################numberOfTests",numberOfTests)
    for Ssize in Ssizet:
        #print("#################################Ssize",Ssize)
        for Csize in Csizet:
            #print("#################################Csize",Csize)
            for Dsize in Dsizet:
                #print("#################################Dsize",Dsize)
                dfg = []
                dfm = []
                correct2 = 0
                for i in range(1,numberOfTests+1):
                    greedysol,minsol = test(Ssize,Csize,Dsize)
                    if (greedysol/minsol <= math.log2(Ssize)):
                      correct2 +=1
                    dfg.append(greedysol)
                    dfm.append(minsol)
                
                
                rboundcorrectness= numberOfTests*100/correct2
                rbound = greedysum/minsum
                frame = (numberOfTests,Ssize,Csize,Dsize,rboundcorrectness,rbound)
                log.append(frame)
                
                print("Number of test cases = ",numberOfTests,"\nS size = ",Ssize,"\nC size = ",Csize,"\nD size",Dsize)  
                print("Ratio Bound Correctness  %" ,rboundcorrectness)
                print("Ratio ",rbound)
                total= 0
                correct = 0
                minsum =0
                greedysum = 0

iscover_count = 0

#%%
df = pd.DataFrame(log, columns=['numberOfTests', 'Ssize', 'Csize','Dsize','rbCorrectness','quality'])
df["quality"] = (1/df.quality)
df.sort_values(by=['numberOfTests'], ascending =True,inplace=True)
df.reset_index()
print(df.head())
#%%