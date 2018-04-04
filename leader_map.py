from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy 
import os
import sys 
import operator
if __name__ == '__main__':
    
    s = os.sep
    cwd = os.getcwd()
#    parent_path = os.path.dirname(cwd)
    rootdir = cwd + s + "Networks_with_ground_truth_communities" + s;

#    parent=  rootdir + 'karate' +s 
#    parent=  rootdir + 'polblogs' +s
#    parent=  rootdir + 'dolphins' +s
#    parent=  rootdir + 'dblp' +s
#    parent=  rootdir + 'com_LiveJournal' +s       
    parent=  rootdir + 'football' +s 
    filename = 'network_v1.dat'
    print 'start to draw %s, its parent is %s' %(filename, parent)
    full_name = os.path.join(parent, filename);     
    G = nx.read_edgelist(full_name, nodetype=int) 
    #G = nx.read_gml(filename)
    
    degree_dictionary = {} #recording the degree of nodes
    distance_dictionary = {} #recording the min distance in which node v can reach a node with higher degree
    statue_dictionary = {} #recording if a node is active or deactive

    nodes_list = G.nodes()
    
    #step I: initialization
    for v in nodes_list:
        degree_dictionary[v] = len(G.neighbors(v))
        distance_dictionary[v] = 100000
        statue_dictionary[v] = 1;

    #print "node list:"
    #print nodes_list
    #print "degree dictionary:"
    #print degree_dictionary
    #print "distance dictionary:"
    #print distance_dictionary
    #print "statue dictionary:"
    #print statue_dictionary

    #step II: turn (1) : Coarse Method
    uncertain_list = [];

    while nodes_list:
        u = nodes_list[0]
        neighbor_of_u =  G.neighbors(u)
        activated_list = []     
        deactivated_list = []
        for v in neighbor_of_u:
            if (statue_dictionary[v] == 1):
                activated_list.append(v)
            else:
                deactivated_list.append(v)

        for v in activated_list:
            if (degree_dictionary[u] > degree_dictionary[v]):
                statue_dictionary[v] = 0
                distance_dictionary[v] = 1
                nodes_list.remove(v)
            elif (degree_dictionary[u] < degree_dictionary[v]):
                statue_dictionary[u] = 0
                distance_dictionary[u] = 1
         
        if (statue_dictionary[u] == 1): #so we should check its deactived neighbors
            for v in deactivated_list:
                if (degree_dictionary[u] < degree_dictionary[v]):
                    statue_dictionary[u] = 0
                    distance_dictionary[u] = 1
                    break;

        if (statue_dictionary[u] == 1): #Coarse Method cannot solve this node, record it and waiting for the Specific Method
            uncertain_list.append(u)
       
        nodes_list.remove(u)

    uncertain_list = sorted(uncertain_list, reverse = True)

    #print "node list:"
    #print nodes_list
    #print "degree dictionary:"
    #print degree_dictionary
    #print "distance dictionary:"
    #print distance_dictionary
    #print "statue dictionary:"
    #print statue_dictionary
    #print "uncertain list:"
    #print uncertain_list

    #Step III: Specific Method
    i = 1;

    while (i < len(uncertain_list)):
        min_distance = 100000;
        j = 0;
        while (degree_dictionary[uncertain_list[j]]  > degree_dictionary[uncertain_list[i]]):
           distance = nx.dijkstra_path_length(G, uncertain_list[i], uncertain_list[j]) 
           if (distance < min_distance):
              min_distance = distance

           if (min_distance == 2):
              break;
           j = j + 1

        distance_dictionary[uncertain_list[i]] = min_distance
        statue_dictionary[uncertain_list[i]] = 0

        i = i + 1
    
    print "degree dictionary:"
    print degree_dictionary
    print "distance dictionary:"
    print distance_dictionary
    print "statue dictionary:"
    print statue_dictionary
    
    Dlists = sorted(degree_dictionary.items(), key=operator.itemgetter(1))
    # sorted by key, return a list of tuples
    Dlists_x, Dlists_y = zip(*Dlists) # unpack a list of pairs into two tuples
    
    Mlists = sorted(distance_dictionary.items(), key=operator.itemgetter(1)) # sorted by key, return a list of tuples
    Mlists_x, Mlists_y = zip(*Mlists) # unpack a list of pairs into two tuples
        
    print "list degree dictionary:"
    print Dlists_y
    
    print "list distance dictionary:"
    print Mlists_y
    #x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(Dlists_y, Mlists_y,c='b', marker='o')
    plt.show()
    
    

