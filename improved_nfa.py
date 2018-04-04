from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy 
import os
import operator

from PIL import Image  
from pylab import * 

def DrawPlot(G, C):
    first_node = (G.nodes())[0]
    node_label = {} 
    C_list = C.values()
    id_color = 0
    for partition in C_list:
        for v in partition:
            node_label[v - first_node] = id_color
        id_color = id_color + 1

    print "Node labels:"
    print node_label

    pos = nx.spring_layout(G)  # compute graph layout
    plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    plt.axis('off')
    # nx.draw_networkx_nodes(G, pos, node_size = len(G.nodes()), cmap=plt.cm.RdYlBu, node_color=list(node_label.values()))
    nx.draw_networkx_nodes(G, pos, node_size = 200, cmap=plt.cm.RdYlBu, node_color=list(node_label.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)         
    
def LeaderSelection(leader_list, G, degree_dictionary):

    distance_dictionary = {} #recording the min distance in which node v can reach a node with higher degree
    statue_dictionary = {} #recording if a node is active or deactive

    nodes_list = G.nodes()    

    #step I: initialization
    for v in nodes_list:
        distance_dictionary[v] = len(G.nodes())
        statue_dictionary[v] = 1;

    #step II: turn (1) : Coarse Method
    uncertain_dictionary = {};

    while nodes_list:
        u = nodes_list[0]

        if (degree_dictionary[u] == 0): #u is isolated
             statue_dictionary[u] = 0
        else:
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
                uncertain_dictionary[u] = degree_dictionary[u]
       
        nodes_list.remove(u)

    uncertain_list = sorted(uncertain_dictionary.items(), key=lambda uncertain_list:uncertain_list[1], reverse=True)

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
   
    print "Step III: Specific Method..."

    global_max_distance = 0

    i = 1;

    while (i < len(uncertain_list)):
        
        min_distance = len(G.nodes());
        j = 0;
        while (degree_dictionary[uncertain_list[j][0]]  > degree_dictionary[uncertain_list[i][0]]):
           
           distance = nx.dijkstra_path_length(G, uncertain_list[i][0], uncertain_list[j][0]) 
           if (distance < min_distance):
              min_distance = distance

           if (min_distance == 2):
              break;
           j = j + 1
       
        if (min_distance != len(G.nodes())):
            distance_dictionary[uncertain_list[i][0]] = min_distance
            statue_dictionary[uncertain_list[i][0]] = 0
        
            if (min_distance > global_max_distance):
	        global_max_distance = min_distance

        i = i + 1
    
    for v in distance_dictionary:
  	if (distance_dictionary[v] == len(G.nodes())):
	    distance_dictionary[v] = global_max_distance
  
    #print "global_max_distance = %d" %global_max_distance
    #print "degree dictionary:"
    #print degree_dictionary
    #print "distance dictionary:"
    #print distance_dictionary
    #print "statue dictionary:"
    #print statue_dictionary

    #Step IV: extract leaders
    #length = nx.single_source_shortest_path_length(G, 34, 2); this function can be used to get SO neighbors
    leader_list = []

    delta = 2.2
    
    mean_density_product = 0
    variance_density_product = 0
    standard_variance_density_product = 0

    density_product_dictionary = {}
     
    for v in degree_dictionary:
        density_product_dictionary[v] = degree_dictionary[v] * distance_dictionary[v]
        mean_density_product = mean_density_product + density_product_dictionary[v]

    #Caculate the mean and std by Kaijie
    mean_density_product = mean_density_product / (len(G.nodes()))   
    for v in density_product_dictionary:
	temp = density_product_dictionary[v] - mean_density_product
        temp = temp * temp
        variance_density_product = variance_density_product + temp
    variance_density_product = variance_density_product / (len(G.nodes()))
    standard_deviation_density_product =  sqrt(variance_density_product)

    #Caculate the mean and std by Jianpeng, and We should use list
    test_mean = np.mean(list(density_product_dictionary.values()), axis=0)
    test_std = np.std(list(density_product_dictionary.values()), axis=0)
    print "mean:"
    print test_mean
    print "variance:"
    print test_std


    for v in density_product_dictionary:
	if ((density_product_dictionary[v] - mean_density_product) > (delta * standard_deviation_density_product)):
	    leader_list.append(v)

    #print "density product:"
    #print density_product_dictionary
    print "kaijie mean:"
    print mean_density_product
    # print "kaijie variance:"
    # print variance_density_product
    print "standard_variance_density_product:"
    print standard_deviation_density_product
    print "leader_list:"
    print leader_list


    #Step V: Draw the Decision
    Dlists = sorted(degree_dictionary.items(), key=operator.itemgetter(1))   # sorted by key, return a list of tuples
    Dlists_x, Dlists_y = zip(*Dlists) # unpack a list of pairs into two tuples
    Mlists = sorted(distance_dictionary.items(), key=operator.itemgetter(1)) # sorted by key, return a list of tuples
    Mlists_x, Mlists_y = zip(*Mlists) # unpack a list of pairs into two tuples
    # print "list degree dictionary:"
    # print Dlists_y  
    # print "list distance dictionary:"
    # print Mlists_y

    #x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.figure(0)
    plt.plot(Dlists_y, Mlists_y,c='b', marker='o', markersize=5, label='Decision Graph')
    plt.grid()
    # plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Local neighbours')
    plt.ylabel('Minimum distance')
    plt.ylim([0, 6.0])
    # plt.show()

    plt.figure(1)
    Density_product_lists = sorted(density_product_dictionary.items(), key=operator.itemgetter(1))   
    print 'Density_product_lists:', Density_product_lists
    Density_product_lists_x, Density_product_lists_y = zip(*Density_product_lists) # unpack a list of pairs into two tuples
    x_value=linspace(1,len(Density_product_lists_y), len(Density_product_lists_y))
    plt.plot(x_value, Density_product_lists_y, color='green', linestyle='--', marker='s', markersize=5, label='Density Product')

    plt.fill_between(x_value, test_mean + delta*test_std, test_mean, alpha=0.15, color='blue')
    plt.grid()
    # plt.xscale('log')
    plt.legend(loc='upper left')
    plt.xlabel('Sorted order')
    plt.ylabel('Density product')
    # plt.show()

    return leader_list
   

def GetCluster (C, leader):
    i = 0;
    while (i < len(C)):
	if (leader in C[i]):
	    break
        i = i + 1
    return i
    
def maximum_connected_components(H):
    """creat maximum_connected_components"""
        
    while True:
        connectedComponents = nx.number_connected_components(H)
        print "Num components: "+str(connectedComponents)
        if connectedComponents > 1:
            print "Taking the highest sub graph"
            nbef = len(H.nodes())
            print "Nodes before - "+str(len(H.nodes()))
            highestCompNodes = 0
            for comp in nx.connected_component_subgraphs(H):
                compNodes = len(comp.nodes())
                if compNodes > highestCompNodes:
                    highestCompNodes = compNodes
                    H = comp
            print "Nodes after - "+str(len(H.nodes()))
            naft = len(H.nodes())
            if naft > int(nbef/connectedComponents):
                return H
                break

            else:
                print "try again"
                #G_ = nx.convert_node_labels_to_integers(G, first_label=start)
                continue
        else:
            return H
            break

if __name__ == '__main__':

#    filename = './football/football.gml' #2 communities
#    G = nx.read_gml(filename)


    s = os.sep
    cwd = os.getcwd()
#    parent_path = os.path.dirname(cwd)
    rootdir = cwd + s + "Networks_with_ground_truth_communities" + s;
    # parent=  rootdir + 'karate' +s 
    # parent=  rootdir + 'polblogs' +s
    # parent=  rootdir + 'dolphins' +s
    parent=  rootdir + 'football' +s 
    filename = 'network_v1.dat'

    # parent=  rootdir + 'dblp' +s       
    # parent=  rootdir + 'com_LiveJournal' +s 
    # filename = 'network_v1_subgraph_speed_up.dat'

    print 'start to draw %s, its parent is %s' %(filename, parent)
    full_name = os.path.join(parent, filename);     
    original_ = nx.read_edgelist(full_name, nodetype=int, create_using=nx.Graph())


#    orig_nodes = nx.number_of_nodes(original_)
#    print 'orig: number of unique nodes:',orig_nodes          
#    orig_edges = nx.number_of_edges(original_)
#    print 'orig: number of unique edges:',orig_edges
    G=maximum_connected_components(original_)
    
    
    ###NFA 
    fc = {};  #record how many followers node i has
    following = {}; #record which one node i follows
    status = {}; #record the role of node i, i.e, 1:follower 2:leader
    degree = {}; #record the degree of nodes

    r = 0.1 
    local_degree_order = 1 #the order of local degree

    nodes_list = G.nodes()

    #initialization
    for v in nodes_list:
        fc[v] = 0
        following[v] = 0
        status[v] = 0;
        degree[v] = len(nx.single_source_shortest_path_length(G, v, local_degree_order)) - 1 #we use seond-order degree
  
    #Process I: Calculate the following relationships 
    for v in nodes_list:    

        #k<--arg max fc(j) where j belong to N(x) U x
        max_fc= -1;
        max_degree = 0;
        k = 0 

        neighbor = G.neighbors(v)
        neighbor.append(v)    
        for u in neighbor:
            if (fc[u] > max_fc):
                max_fc = fc[u]
                max_degree = degree[u]
                k =  u
	    elif (fc[u] == max_fc):
		if (degree[u] > max_degree):
                    max_degree = degree[u]
                    k = u

        #print "the following of node %d is %d" %(v, k)                                    

        #following(x) <-k
        following[v] = k;  
        #fc(k)++
        fc[k] = fc[k] + 1;
       
        if (status[v] == 0):
	    status[v] = 1;
      
    print "degree:"
    print degree
    print "fc:"
    print fc
    print "following"
    print following
   

    #Process II: Calculate leader_list
    leader_list = []
    leader_list = LeaderSelection(leader_list, G, degree)   
   
    print "leaders:"
    print leader_list 


    for leader in leader_list:
	status[leader] = 2

 
    print "status:"
    print status

    #Process III: Calculate cluster relations
    print "Process III start...."
    C = {}
    for v in nodes_list:
	cur = v
	while (status[cur] == 1):
            print "Node %d follows %d" %(cur, following[cur])
	    cur = following[cur]
            if (cur == v):
		status[cur] = 2;
		break;
            ################prevent test####################
            if (cur == following[cur]):
		status[cur] = 2
                break;
            #################################################

        #id_cluster = GetCluster(C, cur);
        if (C.has_key(cur)):
       	    C[cur].append(v)     
        else:
            cluster = []
            cluster.append(v)
            C[cur] = cluster
      

    print "%d cluster in total:" %(len(C))
    print C
        

    # Process IV: Draw Community Result
    DrawPlot(G, C)

