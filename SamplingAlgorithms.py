__author__ = 'Chulaka'

import random
import time
import networkx as nx
import numpy
import math
import array
import csv
from collections import deque
import sys



def randomSampling(G_, keptNodes):
    random.seed(time.clock())
    picked = random.sample(set(G_.nodes()), keptNodes)
    return picked


def randomDegreeSampling(G_, keptNodes):
    probs = []
    picked = []
    edgecount = float(len(G_.edges()))
    for node in G_.nodes():
        probs.append(G_.degree(node)/(2*edgecount))
    cumSumProbs = cumulative_sum(probs)
    cumSumProbs[len(cumSumProbs)-1] = 1.0
    num = 0
    while num < keptNodes:
        random.seed(time.clock())
        number = random.random()
        for node in range(0, len(G_.nodes())):
            if (number <= cumSumProbs[node]):
                if(G_.nodes()[node] not in picked):
                    print "Adding node "+ str(G_.nodes()[node])
                    picked.append(G_.nodes()[node])
                    num = num+1
                    break
                else:
                    #print "Collision"
                    break
    return picked

def randomEigenvectorSampling(G_, keptNodes):
    sumEigen = 0.0
    eigenvector = nx.eigenvector_centrality_numpy(G_)
    for node in G_.nodes():
        sumEigen = sumEigen+eigenvector[node]
    probs = []
    picked = []
    for node in G_.nodes():
        probs.append(eigenvector[node]/sumEigen)
    cumEigenProbs = cumulative_sum(probs)
    cumEigenProbs[len(cumEigenProbs)-1] = 1.0
    num = 0
    while num < keptNodes:
        random.seed(time.clock())
        number = random.random()
        for node in range(0, len(G_.nodes())):
            if (number <= cumEigenProbs[node]):
                if(G_.nodes()[node] not in picked):
                    print "Adding node "+ str(G_.nodes()[node])
                    picked.append(G_.nodes()[node])
                    num = num+1
                    break
                else:
                    #print "Collision"
                    break
    return picked

def core_number_reachability(G_, keptNodes):

    if G_.is_multigraph():
        raise nx.NetworkXError(
                'MultiGraph and MultiDiGraph types not supported.')

    if G_.number_of_selfloops()>0:
        raise nx.NetworkXError(
                'Input graph has self loops; the core number is not defined.',
                'Consider using G.remove_edges_from(G.selfloop_edges()).')

    if G_.is_directed():
        import itertools
        def neighbors(v):
            return itertools.chain.from_iterable([G_.predecessors_iter(v),
                                                  G_.successors_iter(v)])
    else:
        neighbors=G_.neighbors_iter

    degrees=G_.degree()
    # sort nodes by degree
    nodes=sorted(degrees,key=degrees.get)
    # where degrees change
    bin_boundaries=[0]
    curr_degree=0
    for i,v in enumerate(nodes):
        if degrees[v]>curr_degree:
            bin_boundaries.extend([i]*(degrees[v]-curr_degree))
            curr_degree=degrees[v]

    #degree dictrionary
    node_pos = dict((v,pos) for pos,v in enumerate(nodes))
    # initial guesses for core is degree
    core=degrees
    nbrs=dict((v,set(neighbors(v))) for v in G_)
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos=node_pos[u]
                bin_start=bin_boundaries[core[u]]
                node_pos[u]=bin_start
                node_pos[nodes[bin_start]]=pos
                nodes[bin_start],nodes[pos]=nodes[pos],nodes[bin_start]
                bin_boundaries[core[u]]+=1
                core[u]-=1

    #coreSorted = sorted(core.items(), key=lambda x: x[1], reverse=True)
    coreSorted=sorted(core,key=core.get, reverse=True)
    return coreSorted[:keptNodes]
    #return core

def cumulative_sum(L):
     CL = []
     csum = 0
     for x in L:
         csum += x
         CL.append(csum)
     return CL

def topDegreeSubset(fileName, keptNodes ):
    sortedDegreeList = []
    sortedEigenvectorList = []
    sortedClosenessList = []
    sortedBetweennessList = []

    #fileName = "All_lesmis.csv.csv"
    val = 0
    with open(fileName, 'rU') as csvfile:
            filereader = csv.reader(csvfile)
            for row in filereader:
                val = val+1
                #print "Reading row "+str(val)
                if row is not None and len(row) > 0:
                    sortedDegreeList.append(int(row[0]))
                    sortedEigenvectorList.append(int(row[1]))
                    sortedClosenessList.append(int(row[2]))
                    sortedBetweennessList.append(int(row[3]))
            csvfile.close()

    #topPer = int(round(numNodes*precentToKeep))

    degreeTopPer = sortedDegreeList[:keptNodes]
    return degreeTopPer

def SimilaritySample(G_, keptNodes):
    lowerbound = 0.95*keptNodes
    upperbound = 1.05*keptNodes
    print "Range "+str(lowerbound)+"-"+str(upperbound)
    similarityMeasure = 0.81
    nodeSet = G_.nodes()
    while True:
        random.seed(time.clock())
        random.shuffle(nodeSet)
        sample = set([])
        sample.add(nodeSet[0])
        #Add first node to sample
        for node in nodeSet:
        #   For each node v in the network
            matched = False
            for sampledNode in sample:
                nodeNeighbors = G_.neighbors(node)
                #nodeNeighbors.append(node)
                sampleNeighbors = G_.neighbors(sampledNode)
                #sampleNeighbors.append(sampledNode)
                #   If v does not have more than simMeasure similarity with any node in the sample
                #   Add v to Sample
                intersection = list(set(nodeNeighbors).intersection(set(sampleNeighbors)))
                intersectionPer = 0.0
                if len(nodeNeighbors) < len(sampleNeighbors):
                    intersectionPer = len(intersection)/float(len(nodeNeighbors))
                else:
                    intersectionPer = len(intersection)/float(len(sampleNeighbors))
                #print len(nodeNeighbors)
                #print len(sampleNeighbors)
                #print intersectionPer

                if (intersectionPer > similarityMeasure):
                    if (len(nodeNeighbors) <= len(sampleNeighbors)):
                        #print "Found match for "+str(node)+ " in node "+str(sampledNode)
                        matched = True
                        break
                    else:
                        #print "Replacing "+str(sampledNode)+ " by node "+str(node)
                        sample.remove(sampledNode)
                        sample.add(node)
                        matched = True
                        break
                else:
                    continue
            if (not matched) and (node not in sample):
                #print "Adding "+str(node)+" to sample"
                sample.add(node)

        sampleSize = len(sample)
        #print sampleSize

        if (sampleSize > lowerbound) and (sampleSize < upperbound):
            print "Final size of the sample "+str(sampleSize)
            break
        else:
            print "Discarding sample of size "+str(sampleSize)
            continue

    return sample


def multiRandomWalk(G_, keptNodes):
    picked = []
    random.seed(time.clock())
    random.shuffle(G_.nodes())
    seeds = random.sample(set(G_.nodes()), 5)
    sinNodes = int(math.floor(keptNodes/5.0))
    for seed in seeds:
        current = seed
        picked.append(seed)
        c = 0.85
        temp = 0
        num = 1
        #print "Added "+str(seed)+ " num is "+str(num)
        temp_list = []
        num_true = int(100*c)
        num_false = 100-num_true
        for t_val in range (num_true):
            temp_list.append(1)
        for f_val in range (num_false):
            temp_list.append(0)
        while num < sinNodes:
            random.shuffle(temp_list)
            info_accepted = random.choice(temp_list)
            if (info_accepted==1):
                neighborlist = G_.neighbors(current)
                #prevCurr = current
                oneSelected = False
                for ind in range(0, len(neighborlist)):
                    random.seed(time.clock())
                    random.shuffle(neighborlist)
                    walkingTo = random.choice(neighborlist)
                    if (walkingTo not in picked):
                        num = num+1
                        temp = num-1
                        #print "Added "+str(walkingTo)+ " num is "+str(num)
                        picked.append(walkingTo)
                        current = walkingTo
                        oneSelected = True
                        break
                if not oneSelected:
                    val = temp-1
                    temp -= 1
                    current = picked[val]
                    continue
                    #temp = temp-1
            else:
                val = temp-1
                temp -= 1
                current = picked[val]
                continue

    return picked

def randomWalk(G_, keptNodes):
    picked = []
    random.seed(time.clock())
    random.shuffle(G_.nodes())
    seed = random.choice(G_.nodes())
    current = seed
    picked.append(seed)
    c = 0.85
    temp = 0
    num = 1
    #print "Added "+str(seed)+ " num is "+str(num)
    temp_list = []
    num_true = int(100*c)
    num_false = 100-num_true
    for t_val in range (num_true):
        temp_list.append(1)
    for f_val in range (num_false):
        temp_list.append(0)
    while num < keptNodes:
        random.shuffle(temp_list)
        info_accepted = random.choice(temp_list)
        if (info_accepted==1):
            neighborlist = G_.neighbors(current)
            #prevCurr = current
            oneSelected = False
            for ind in range(0, len(neighborlist)):
                random.seed(time.clock())
                random.shuffle(neighborlist)
                walkingTo = random.choice(neighborlist)
                if (walkingTo not in picked):
                    num = num+1
                    temp = num-1
                    #print "Added "+str(walkingTo)+ " num is "+str(num)
                    picked.append(walkingTo)
                    current = walkingTo
                    oneSelected = True
                    break
            if not oneSelected:
                val = temp-1
                temp -= 1
                current = picked[val]
                continue
                #temp = temp-1


        else:
            val = temp-1
            temp -= 1
            current = picked[val]
            continue

    print "Size of picked "+str(len(picked))
    return picked


def randomWalk_with_jump(G_, keptNodes):
    c = 0.85
    temp_list = []
    num_true = int(100*c)
    num_false = 100-num_true
    for t_val in range (num_true):
        temp_list.append(1)
    for f_val in range (num_false):
        temp_list.append(0)

    nmax = nx.number_of_nodes(G_)
    picked = []
#    random.seed(time.clock())
#    random.shuffle(G_.nodes())
#    seed = random.choice(G_.nodes())
#    current = seed
#    picked.append(seed)

    temp = 0
    num = 0
    n_iteration=0
    #print "Added "+str(seed)+ " num is "+str(num)
  

    # The first time at the n_iterations to sample the nodes
    # while n_iteration <= nmax and num < keptNodes:
    #     print n_iteration
    #     n_iteration+=1
    #     random.shuffle(temp_list)
    #     info_accepted = random.choice(temp_list)
    #     if (info_accepted==1):
    #         neighborlist = G_.neighbors(current)
    #         #prevCurr = current
    #         oneSelected = False
    #         for ind in range(0, len(neighborlist)):
    #             random.seed(time.clock())
    #             random.shuffle(neighborlist)
    #             walkingTo = random.choice(neighborlist)
    #             if (walkingTo not in picked):
    #                 num = num+1
    #                 temp = num-1
    #                 #print "Added "+str(walkingTo)+ " num is "+str(num)
    #                 picked.append(walkingTo)
    #                 current = walkingTo
    #                 oneSelected = True
    #                 break
    #         if not oneSelected:
    #             val = temp-1
    #             temp -= 1
    #             current = picked[val]
    #             continue
    #             #temp = temp-1
    #     else:
    #         val = temp-1
    #         temp -= 1
    #         current = picked[val]
    #         continue

    while (num <= keptNodes):
        # If the nodes are enough.
        if num == keptNodes:
            print('It achieves the goal.')
            break;

        # If the nodes are enough.
        elif num < keptNodes:
            random.seed(time.clock())
            random.shuffle(G_.nodes())
            seed = random.choice(G_.nodes())
            current = seed
            picked.append(seed)
            num = num+1
            
            while n_iteration <= nmax and num < keptNodes:
                print n_iteration
                n_iteration+=1
                random.shuffle(temp_list)
                info_accepted = random.choice(temp_list)
                if (info_accepted==1):
                    neighborlist = G_.neighbors(current)
                    #prevCurr = current
                    oneSelected = False
                    for ind in range(0, len(neighborlist)):
                        random.seed(time.clock())
                        random.shuffle(neighborlist)
                        walkingTo = random.choice(neighborlist)
                        if (walkingTo not in picked):
                            num = num+1
                            temp = num-1
                            #print "Added "+str(walkingTo)+ " num is "+str(num)
                            picked.append(walkingTo)
                            current = walkingTo
                            oneSelected = True
                            break
                    if not oneSelected:
                        val = temp-1
                        temp -= 1
                        current = picked[val]
                        continue
                        #temp = temp-1
                else:
                    val = temp-1
                    temp -= 1
                    current = picked[val]
                    continue

    print "Size of picked "+str(len(picked))
    return picked

def forest_fire_sampler(g, target_nodes):
    total_nodes = nx.number_of_nodes(g)
    to_burn = deque()
    
    finished = False
    #this is an import factor for forest_fire, its average value pf/1-pf is the number of vertices that a node need to burn its neighbours.
    p = 2.0 / 3.0
    
    print p
    
    while not(finished):
        to_burn.clear()
    
        to_burn.append(random.choice(g.nodes()))
    
        while to_burn:                
            burning = to_burn.popleft()   
            nodes = g.neighbors(burning)
            g.remove_node(burning)
            
            total_nodes = total_nodes - 1
            if (total_nodes <= target_nodes):        
                finished = True
                break
    
            nodes = [node for node in nodes if node not in to_burn and not(node == burning)]
            x = numpy.random.geometric(p)
            if len(nodes) > x:                   
                    nodes = random.sample(nodes, x)
         
    return g

