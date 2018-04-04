from __future__ import division
import networkx as nx
import community
import matplotlib.pyplot as plt
import time
import copy 

from PIL import Image  
from pylab import *  

import os


def Verification(G_last_snapshot, G_current_snapshot, degree):
    
    #print "%d edges in G_last_snapshot" %(len(G_last_snapshot.edges()))
    #print "%d edges in G_current_snapshot" %(len(G_current_snapshot.edges()))

    #print "%d nodes in G_last_snapshot" %(len(G_last_snapshot.nodes()))
    #print "%d nodes in G_current_snapshot" %(len(G_current_snapshot.nodes()))

    for node in G_current_snapshot.nodes():
        
        realistic_degree = len(nx.single_source_shortest_path_length(G_current_snapshot, node, 2)) - 1
        record_degree = degree[node]

        if (record_degree != realistic_degree):
            print "now we check node %d, record_degree = %d, realistic_degree = %d" %(node, record_degree, realistic_degree)
            time.sleep(1000)
	    return 0

    print "every thing is OK!"
    return 1

def CountCommonNeighbors(G, u, v):
    
    common_neighbors_list = []
    common_neighbors_list = GetCommonNeighbors(common_neighbors_list, G, u, v)
    
    #print "%d and %d has following common neighbors:" %(u,v)
    #print common_neighbors_list
    #print "neighbors of %d:" %u
    #print G.neighbors(u)
    #print "neighbors of %d:" %v
    #print G.neighbors(v)
  
    return len(common_neighbors_list)
    
def GetCommonNeighbors(common_neighbors_list, G, u, v):

    temp =  nx.common_neighbors(G, u, v)
               
    for node in temp:
        common_neighbors_list.append(node)

    return common_neighbors_list


def update_dynamic_edge_stream(dynamic_edge_stream, last_snapshot, current_snapshot):
    #firstly, we collect all the edges from last_snapshot which have already been deleted in current_snapshot
    #these edges should been added into dynamic_edge_stream as the edge-deleting request in the time window of current_snapshot
    
    j = 0; ##
    
    lower_bound = 0
    
    size_current_snapshot = len(current_snapshot)
    size_last_snapshot = len(last_snapshot)
    
    for edge in last_snapshot:
       
        while (lower_bound < size_current_snapshot): #aimed range of edges has not been reached, we should increase lower bound
            if (edge[0] <= current_snapshot[lower_bound][0]):
                break;
            lower_bound = lower_bound + 1

        if (lower_bound >= size_current_snapshot):  #aimed range has been passed, edge in last snapshot is not in current snapshot
            del_edge_stream = "! " + str(edge[0]) + ' ' +  str(edge[1])
            dynamic_edge_stream.append(del_edge_stream)

        else:
            if (edge[0] < current_snapshot[lower_bound][0]):   #aimed range has been passed, edge in last snapshot is not in current snapshot
                del_edge_stream = "! " + str(edge[0]) + ' ' +  str(edge[1]) 
                dynamic_edge_stream.append(del_edge_stream)
            elif (edge[0] == current_snapshot[lower_bound][0]): #aimed range has been reached , we should check whether the edge in last snapshot is still in current snapshot
            
                i = lower_bound
                while (i < size_current_snapshot):
                    if (edge[0] < current_snapshot[i][0]):
                        break;
                    if (edge[1] == current_snapshot[i][1]):
                        break;
                    i = i + 1
                        
                if (i >= size_current_snapshot): #current snapshot has come to its end, so the edge in last edge in last snapshot is not in current snap
                    del_edge_stream = "! " + str(edge[0]) + ' ' +  str(edge[1])  
                    dynamic_edge_stream.append(del_edge_stream)
                else:
                    if (edge[0] < current_snapshot[i][0]): #edge in last snapshot is not in current snapshot
                        del_edge_stream = "! " + str(edge[0]) + ' ' +  str(edge[1])
                        dynamic_edge_stream.append(del_edge_stream) 


        #if (j % 10000 == 0): ##
        #    print "j = %d" %j
        j = j + 1

    #then, we collect all the edges from current_snapshot which does not exist in last_snapshot
    #these edges should been added into dynamic_edge_stream as the edge-adding request in the time window of current_snapshot

    k = 0; ##

    lower_bound = 0
    
    for edge in current_snapshot:
    
        while (lower_bound < size_last_snapshot):
            if (edge[0] <= last_snapshot[lower_bound][0]):
                break;
            lower_bound = lower_bound + 1

        if (lower_bound >= size_last_snapshot):
            edge_stream = str(edge[0]) + ' ' +  str(edge[1])
            dynamic_edge_stream.append(edge_stream)
        else:
            if (edge[0] < last_snapshot[lower_bound][0]): 
                edge_stream = str(edge[0]) + ' ' +  str(edge[1])
                dynamic_edge_stream.append(edge_stream)
            elif (edge[0] == last_snapshot[lower_bound][0]):
                i = lower_bound
                while (i < size_last_snapshot):
                    if (edge[0] < last_snapshot[i][0]):
                        break;
                    if (edge[1] == last_snapshot[i][1]):
                        break;
                    i = i + 1
                
                if (i >= size_last_snapshot):      
                    edge_stream = str(edge[0]) + ' ' +  str(edge[1])
                    dynamic_edge_stream.append(edge_stream)
                else:   
                    if (edge[0] < last_snapshot[i][0]): #edge in last snapshot is not in last snapshot, so it a new edge
                        edge_stream = str(edge[0]) + ' ' +  str(edge[1]) 
                        dynamic_edge_stream.append(edge_stream)
            

	   
        #if (k % 10000 == 0):
        #    print "k = %d" %k ##   
        k = k + 1 ##

    return dynamic_edge_stream

def VerifyReachability(G, u, v):
    flag_reachable = 0;
    components_list = nx.connected_component_subgraphs(G)
    for component in components_list:
        
        if ((u in component.nodes()) and (v in component.nodes())):
            flag_reachable = 1;
            break;

    return flag_reachable;

def DrawPlot(G, C):
    first_node = (G.nodes())[0]
  
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
    nx.draw_networkx_nodes(G, pos, node_size = len(G.nodes()), cmap=plt.cm.RdYlBu, node_color=node_label)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)         
    
def GetClusterStructure(C, following, status, leader_list):
    for v in nodes_list:
	cur = v
	while (status[cur] == 1):
            #print "Node %d follows %d" %(cur, following[cur])
	    cur = following[cur]
            if (cur == v):
                print "I:add %d in leader list" %cur
		status[cur] = 2;

                ############################
                if (cur not in leader_list):
                    leader_list.append(cur)
                ############################

		break;
            ##################prevent test###################
            if (cur == following[cur]):
                #print "II:add %d in leader list" %cur
		status[cur] = 2

                ############################
                if (cur not in leader_list):
                    leader_list.append(cur)
                ############################

                break;
            #################################################

        #id_cluster = GetCluster(C, cur);
        if (C.has_key(cur)):
       	    C[cur].append(v)     
        else:
            cluster = []
            cluster.append(v)
            C[cur] = cluster

    return C

def LeaderSelection(leader_list, G, degree_dictionary):

    distance_dictionary = {} #recording the min distance in which node v can reach a node with higher degree
    statue_dictionary = {} #recording if a node is active or deactive

    nodes_list = G.nodes()    

    if (len(nodes_list) == 0):
        return []

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
 
    global_max_distance = 0

    i = 1;

    while (i < len(uncertain_list)):
        
        min_distance = len(G.nodes());

        j = 0;

        while (degree_dictionary[uncertain_list[j][0]]  > degree_dictionary[uncertain_list[i][0]]):
           
            if (VerifyReachability(G, uncertain_list[i][0], uncertain_list[j][0])):

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

    #length = nx.single_source_shortest_path_length(G, 34, 2); this function can be used to get SO neighbors

    #Step IV: extract leaders
    leader_list = []

    delta = 3
    
    mean_density_product = 0
    variance_density_product = 0
    standard_variance_density_product = 0

    density_product_dictionary = {}
    
    #print "distance:"
    #print distance_dictionary
    #print "nodes:"
    #print nodes_list
    
    for v in distance_dictionary:
        density_product_dictionary[v] = degree_dictionary[v] * distance_dictionary[v]
        mean_density_product = mean_density_product + density_product_dictionary[v]
   
    mean_density_product = mean_density_product / (len(G.nodes()))
    
    for v in distance_dictionary:
	temp = density_product_dictionary[v] - mean_density_product
        temp = temp * temp
        variance_density_product = variance_density_product + temp
   
    variance_density_product = variance_density_product / (len(G.nodes()))

    standard_deviation_density_product =  sqrt(variance_density_product)

    for v in density_product_dictionary:
	if ((density_product_dictionary[v] - mean_density_product) > (delta * standard_deviation_density_product)):
	    leader_list.append(v)

    
    print "density product:"
    print density_product_dictionary

    #time.sleep(1000)

    #print "mean:"
    #print mean_density_product
    #print "variance:"
    #print variance_density_product
    #print "standard_variance_density_product:"
    #print standard_deviation_density_product
    #print "%d leaders in total:" %(len(leader_list))
    #print leader_list
  
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
    
    s = os.sep
    cwd = os.getcwd()
#    parent_path = os.path.dirname(cwd)
    rootdir = cwd + s + "Networks_with_ground_truth_communities" + s;
    parent=  rootdir + 'new_enronYY_u_45' +s 
    
    
    prefix = "separated_output-prefix.t0000"
    suffix = ".graph"
    id_snapshot = 0
    
    filename =  prefix + str(id_snapshot) + suffix    
    

    print 'start to draw %s, its parent is %s' %(filename, parent)
    full_name = os.path.join(parent, filename);     
    original_ = nx.read_edgelist(full_name, nodetype=int, create_using=nx.Graph())
#    orig_nodes = nx.number_of_nodes(original_)
#    print 'orig: number of unique nodes:',orig_nodes          
#    orig_edges = nx.number_of_edges(original_)
#    print 'orig: number of unique edges:',orig_edges
    G_current_snapshot=maximum_connected_components(original_)
#    G_current_snapshot = nx.read_edgelist(filename, nodetype = int)
 
    fc = {};  #record how many followers node i has
    following = {}; #record which one node i follows
    status = {}; #record the role of node i, i.e, 1:follower 2:leader
    degree = {}; #record the degree of nodes

    leader_list = []
    node_label = []
    nodes_list = G_current_snapshot.nodes()

    local_degree_order =  2  #the order of local degree

    #Pre-Process:initialization
    for v in nodes_list:
        fc[v] = 0
        following[v] = 0
        status[v] = 0;

        node_label.append(0)

        degree[v] = len(nx.single_source_shortest_path_length(G_current_snapshot, v, local_degree_order)) - 1 #we use seond-order degree

    while (1):
     
        #Process I: Calculate leaders
      
        #print "%d nodes in current snapshot" %(len(nodes_list))
        #print nodes_list
        #print degree

        if (len(leader_list) == 0):  #this is the first turn, we initialize a leader list for t = 0
            leader_list = LeaderSelection(leader_list, G_current_snapshot, degree)

            print "%d leaders has been selected" %(len(leader_list))
        else:
            temp = leader_list
            leader_list = []
	    for v in temp:
                if ((v in nodes_list) and (degree[v] != 0)): #in following turns, if a leader is deleted or if it becomes the isolated one, it should be removed from leader list
		    leader_list.append(v)
            print "%d leaders still effective in this snapshot" %(len(leader_list))
        print leader_list

        for leader in leader_list:
	    status[leader] = 2
            following[leader] = leader
            fc[leader] = 1

        #Process II: Calculate the following relationships
        for v in nodes_list:    

            #k<--arg max fc(j) where j belong to N(x) U x
            max_fc= -1;
            max_degree = 0;
            k = 0 

            neighbor = G_current_snapshot.neighbors(v)
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

            #following(x) <-k
            following[v] = k;  
            #fc(k)++
            fc[k] = fc[k] + 1;
       
            if (status[v] == 0):
	        status[v] = 1;

        #Process III: Calculate cluster relations
        #print "Process III start...."

        C = {}
        C = GetClusterStructure(C, following, status, leader_list)

        
        print "%d cluster in total:" %(len(C))
        print "%d leaders in total:" %(len(leader_list))
        print leader_list

        #print C
    
        #Process IV: prepare the snapshot file for next time step
        G_last_snapshot = G_current_snapshot

        id_snapshot = id_snapshot + 1
        filename =  prefix + str(id_snapshot) + suffix
        if (os.path.exists(os.path.join(parent, filename)) == False):
            break;
  
        print "filename: %s" %filename

        G_current_snapshot = nx.read_edgelist(os.path.join(parent, filename), nodetype = int)
        


        last_snapshot =  G_last_snapshot.edges();
        current_snapshot = G_current_snapshot.edges();

        dynamic_edge_stream = []
        dynamic_edge_stream = update_dynamic_edge_stream(dynamic_edge_stream, last_snapshot, current_snapshot) #generate the update of edge list

        #Process V: dynamic update for
        
        for item in dynamic_edge_stream:
           
            #print item

            split_items = item.split(' ')

            if (split_items[0] != '!'):

		id_from = int(split_items[0])
                id_to = int(split_items[1])
                 
                if(id_from not in G_last_snapshot):
                    G_last_snapshot.add_node(id_from)

                if(id_to not in G_last_snapshot):
                    G_last_snapshot.add_node(id_to)


                if (degree.has_key(id_from) == False):
                    #print "Node %d is added" %id_from
		    degree[id_from] = 0

                if (degree.has_key(id_to) == False):
                    #print "Node %d is added" %id_to
		    degree[id_to] = 0

                #if (id_to not in G_last_snapshot.nodes()):
                #    print "%d not in !" %id_to
                #    time.sleep(1000)
    
                ret = CountCommonNeighbors(G_last_snapshot, id_from, id_to)
                               
                if (ret == 0): #this indicates cannot reach each other in 2 hops until this edge is added:
                    #print "Node %d and %d cannot reach each other until this edge is added, so their SO-degree increase" %(id_from, id_to)
		    degree[id_from] = degree[id_from] + 1
                    degree[id_to] = degree[id_to] + 1
            

                neighbors_from = G_last_snapshot.neighbors(id_from)
                neighbors_to = G_last_snapshot.neighbors(id_to)

                for node in neighbors_from:
                    neighbors_node = G_last_snapshot.neighbors(node)
                    if (id_to not in neighbors_node):  #indicating 'node' cannot reach 'to' in 1 hop
                        ret = CountCommonNeighbors(G_last_snapshot, node, id_to)
                        if ( ret == 0 ): #indicating 'node' can only reach 'to' in 2 hops through the edge (from, to)
			    degree[node] = degree[node] + 1
                            degree[id_to] = degree[id_to] + 1


                for node in neighbors_to:
                    neighbors_node = G_last_snapshot.neighbors(node)
                    if (id_from not in neighbors_node):
                        ret = CountCommonNeighbors(G_last_snapshot, node, id_from)
		        if ( ret == 0):
			    degree[node] = degree[node] + 1
                            degree[id_from] = degree[id_from] + 1
       
                G_last_snapshot.add_edge(id_from, id_to)

            else:

                id_from = int(split_items[1])
                id_to = int(split_items[2])

                G_last_snapshot.remove_edge(id_from, id_to)
 
                ret = CountCommonNeighbors(G_last_snapshot, id_from, id_to)
                
                if( ret == 0 ): #this indicates cannot reach each other in 2 hops after this edge is added:
                    #print "Node %d and %d cannot reach each other after this edge is added, so their SO-degree increase" %(id_from, id_to)
		    degree[id_from] = degree[id_from] - 1
                    degree[id_to] = degree[id_to] - 1  

                neighbors_from = G_last_snapshot.neighbors(id_from)
                neighbors_to = G_last_snapshot.neighbors(id_to) 

                for node in neighbors_from:
                    neighbors_node = G_last_snapshot.neighbors(node)
                    if (id_to not in neighbors_node):  #indicating 'node' cannot reach 'to' in 1 hop
                        ret = CountCommonNeighbors(G_last_snapshot, node, id_to)
                        if ( ret == 0 ): #indicating 'node' can only reach 'to' in 2 hops through the edge (from, to)
			    degree[node] = degree[node] - 1
                            degree[id_to] = degree[id_to] - 1

                for node in neighbors_to:
                    neighbors_node = G_last_snapshot.neighbors(node)
                    if (id_from not in neighbors_node):
                        ret = CountCommonNeighbors(G_last_snapshot, node, id_from)
		        if ( ret == 0 ):
			    degree[node] = degree[node] - 1
                            degree[id_from] = degree[id_from] - 1    

#        Verification(G_last_snapshot, G_current_snapshot, degree)
        
        #Process VI: Clean the data structure
        nodes_list = G_current_snapshot.nodes()
        for v in nodes_list:
            fc[v] = 0
            following[v] = 0
            status[v] = 0;
 
            node_label[0] = 0
 
    print "It's OK."

   
   


 
   


    
