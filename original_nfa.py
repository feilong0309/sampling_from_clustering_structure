from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy 

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

def GetCluster (C, leader):
    i = 0;
    while (i < len(C)):
	if (leader in C[i]):
	    break
        i = i + 1
    return i

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
 
    fc = {};  #record how many followers node i has
    following = {}; #record which one node i follows
    status = {}; #record the role of node i, i.e, 1:follower 2:leader
    degree = {}; #record the degree of nodes

    node_label = []

    r = 0.1 #0.1, 0.5, 0.9

    nodes_list = G.nodes()

    #initialization
    for v in nodes_list:
        fc[v] = 0
        following[v] = 0
        status[v] = 0;
        degree[v] = len(G.neighbors(v))
    
        node_label.append(0)

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
        if ((status[k] == 1) or (status[k] == 2)):
            leader = following[k]

            #print "k = %d, leader = %d, fc[k] = %d, fc[leader] = %d" %(k, leader, fc[k], fc[leader]);

            if ((fc[k]/fc[leader]) >= r):
                status[k] = 2
            else:
                status[k] = 1

    print "fc:"
    print fc
    print "following"
    print following
    print "status:"
    print status

    #Process II: Calculate cluster relations
    C = {}
    for v in nodes_list:
	cur = v
	while (status[cur] == 1):
            #print "Node %d follows %d" %(cur, following[cur])
	    cur = following[cur]
            if (cur == v):
		status[cur] = 2;
		break;
            
        #id_cluster = GetCluster(C, cur);
        if (C.has_key(cur)):
       	    C[cur].append(v)     
        else:
            cluster = []
            cluster.append(v)
            C[cur] = cluster
      

    print "%d clusters in total:" %(len(C))
    print C
        
    #Process IV: Draw Community Result
    DrawPlot(G, C)      
