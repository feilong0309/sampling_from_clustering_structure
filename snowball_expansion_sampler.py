from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy 
import random
import os
import sys


#########################################################################
#paraments:
#    neighbors_of_sampling_set: N(S), |N(S)| = len(neighbors_of_sampling_set)
#    sampling_set:S, |S| = len(sampling_set)
#return:
#    |N(S)|/|S|
##########################################################################
def calc_expansion_factor(neighbors_of_sampling_set, sampling_set):
    return len(neighbors_of_sampling_set) / len(sampling_set)


##########################################################################
#return:
#    N(v) - (N(S) U S)
##########################################################################
def calc_extra_neighbors_of_v(G, v, neighbors_of_sampling_set, sampling_set):
      
    neighbors_of_v = G.neighbors(v) #N(v)
    extra_neighbors_of_v = copy.deepcopy(neighbors_of_v)

    for i in neighbors_of_v:  
        #if v belongs to (N(S) U S), remove it from N(v)
        if ((i in neighbors_of_sampling_set) or (i in sampling_set)): 
            extra_neighbors_of_v.remove(i)
      
      
    return (extra_neighbors_of_v)


def update_set(sampling_set, neighbors_of_sampling_set, selected_node, extra_neighbors_of_selected_node):

    if selected_node not in neighbors_of_sampling_set:
        print "selected_node:"
        print selected_node
        print "neighbors_of_sampling_set"
        print neighbors_of_sampling_set
        print "sampling set:"
        print sampling_set 
         
    sampling_set.append(selected_node)
    neighbors_of_sampling_set.remove(selected_node)
    for i in extra_neighbors_of_selected_node:
        neighbors_of_sampling_set.append(i)
            

def build_sample_graph(node_list, edge_list):
    S = nx.Graph()
    S.add_nodes_from(node_list)
    S.add_edges_from(edge_list)
    return S

def snowball_expansion_sampling(G, sample_ratio):
    
    node_list = G.nodes() 
    edge_list = G.edges() 
    sample_size = len(node_list) * sample_ratio 
    
    sampling_nodes = []  # S
    neighbors_of_sampling_nodes = [] # N(S)

    v = random.choice(node_list) #chose node from V at random     

    sampling_nodes.append(v) # S = S U {v}
    neighbors_of_sampling_nodes = G.neighbors(v) 

    print "turn 0:"
    # print sampling_nodes
    # print neighbors_of_sampling_nodes
 
    #step II: nodes sampling
    while (len(sampling_nodes) < sample_size):
       	
        selected_node = 0
        extra_neighbors_of_selected_node = []
        number_of_extra_neighbors_of_selected_node = 0

        i = 0
        while (i < len(neighbors_of_sampling_nodes)):
          
            v =  neighbors_of_sampling_nodes[i]	
            extra_neighbors_of_v = calc_extra_neighbors_of_v(G, v, neighbors_of_sampling_nodes, sampling_nodes)

             # print "v = %d, v's |N(v) - (N(S) U S)| = %d, selected node's |N(v) - (N(S) U S)| = %d" %(v, len(extra_neighbors_of_v), number_of_extra_neighbors_of_selected_node)

            if (len(extra_neighbors_of_v) >= number_of_extra_neighbors_of_selected_node): #sample a node based on max |N(v) - (N(S) U S)|
                selected_node = v
                extra_neighbors_of_selected_node = extra_neighbors_of_v
                number_of_extra_neighbors_of_selected_node  = len(extra_neighbors_of_v)

            i = i + 1
 
        #update related information
        update_set(sampling_nodes, neighbors_of_sampling_nodes, selected_node, extra_neighbors_of_selected_node)

        # print "turn %d:" %(len(sampling_nodes))
        # print sampling_nodes
        # print neighbors_of_sampling_nodes

    number_of_sampled_nodes= len(sampling_nodes)
    #zjp add if the number_of_sampled_nodes is larger than sample_size
    while (number_of_sampled_nodes > sample_size):
        min_degree = sys.maxint
        for sampled_node in sampling_nodes:
            # neighbours = G.neighbors(sampled_node)
            node_degree = G.degree(sampled_node)
            if (node_degree  < min_degree): 
                min_degree = node_degree
                temp_node = sampled_node
            else:
                continue
        sampling_nodes.remove(temp_node)
        number_of_sampled_nodes= number_of_sampled_nodes-1

    #step III: edges sampling
    sampling_edges = []
    for i in edge_list:
        node_sour = i[0]
        node_dest = i[1]
        if ((node_sour in sampling_nodes) and (node_dest in sampling_nodes)):
            sampling_edges.append(i)

    #stepIV: draw a graph
    S = build_sample_graph(sampling_nodes, sampling_edges)

    print S.nodes()
    print "%d nodes" %S.number_of_nodes()
    print S.edges()
    print "%d edges" %S.number_of_edges()
    return S

if __name__ == '__main__':


    #filename = './karate/karate.gml'
    #filename = './polbooks/polbooks.gml'
#==============================================================================
#     filename = './football/network_v1.dat'
#     ge=open(filename, 'rb')
#     original_=nx.read_edgelist(ge, nodetype=int, create_using=nx.Graph())    
#     #option_conditions_in_initialization = ['--not_direct_neighbors']
#     option_conditions_in_initialization = []
#     top_k_snowball_sampling_algorithm_balanced(original_, 0.5, 12, option_conditions_in_initialization)
#==============================================================================
 
    s = os.sep
    cwd = os.getcwd()
    parent_path = os.path.dirname(cwd)
    rootdir = cwd + s + "Networks_with_ground_truth_communities" + s+ 'football';

    list_dirs = os.walk(rootdir)
    for parent, dirnames, filenames in list_dirs:
        #for dirname in dirnames:
        #    print 'parent is %s' %parent
        #    print 'dirname is %s' %dirname
        for filename in filenames:
            split_filename = filename.split('.');
            if filename == 'network_v1.dat':
                print 'start to draw %s, its parent is %s' %(filename, parent)
                full_name = os.path.join(parent, filename);
                original_ = nx.read_edgelist(full_name, nodetype=int)  
                sample_rate= 0.5
                sample_ = snowball_expansion_sampling(original_, sample_rate)
                
          
                fh=open("test.edgelist",'wb')
#                nx.write_edgelist(S,fh,data=False)
                nx.write_edgelist(sample_, parent+ os.sep+"network_sample_p"+str(int(100*float(sample_rate)))+"_v1.dat", data=False)   
                
                fh.close()
                
#            elif filename == 'network_v1_subgraph_speed_up.dat':
#                print 'start to draw %s, its parent is %s' %(filename, parent)
#                full_name = os.path.join(parent, filename);
#                G = nx.read_edgelist(full_name, nodetype=int)                
#                S = top_k_snowball_sampling_algorithm(G, 0.15, 5000)
#                fh=open("test.edgelist",'wb')
#                nx.write_edgelist(S,fh,data=False)
#                nx.write_edgelist(S, full_name+"_test.edgelist", data=False)
            else:            
                continue
