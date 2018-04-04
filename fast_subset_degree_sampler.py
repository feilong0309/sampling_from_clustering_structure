from string import strip,split;
from math import ceil;
import time;
import operator;
import copy;
import collections;
import degree_copy;
import adjacency_copy;
from ordereddict import OrderedDict;
import networkx as nx
import os
import math
def fast_subset_degree_sampling(filename,rp,delimeter):
    start_time = time.time();
    fp=open(filename,'r');


    network = [];
    degree_nodes,adjacency_matrix = {},{};
    mod_degree_matrix, mod_adjacence_matrix = {},{};
    
    #Reading Input File to Obain Degree and Adjacency Matrix
    for line in fp:
        data = split(strip(line,'\r\n'),sep=delimeter);
        if (len(data)<3):
            weight = 1;
        else: 
            weight = float(data[2]);
        node1 = int(float(data[0]));
        node2 = int(float(data[1]));
        #Degree Calculation
        if (node1 not in degree_nodes):
            degree_nodes[node1] = weight;
        else:
            degree_nodes[node1] = degree_nodes[node1]+weight;
        if (node2 not in degree_nodes):
            degree_nodes[node2] = weight;
        else:
    	    degree_nodes[node2] = degree_nodes[node2]+weight;

        #Adjacency Matrix Calculation
        if (node1 not in adjacency_matrix):
            adjacency_matrix[node1] = [weight,node2];
        else:
            adjacency_matrix[node1][0] = adjacency_matrix[node1][0]+weight;
            adjacency_matrix[node1].append(node2);
        if (node2 not in adjacency_matrix):
            adjacency_matrix[node2] = [weight,node1];
        else:
            adjacency_matrix[node2][0] = adjacency_matrix[node2][0]+weight;
            adjacency_matrix[node2].append(node1);

    median_degree = 1.0*sorted(degree_nodes.values())[len(degree_nodes.values())//2];
    No_nodes = len(degree_nodes);
    #print No_nodes;
    represent_points = rp;

    #Obtain Degree Matrix and Adjacency_Matrix with degree > median
    #Time required for this operation is O(n)
    mod_degree_nodes = dict((k,v) for (k,v) in degree_nodes.iteritems() if v>median_degree);
    kout = open('DegreeDistribution.txt','w');
    for key,values in mod_degree_nodes.iteritems():
        kout.write(str(key)+"\t"+str(values)+"\n");
    kout.close();
    #sorted_degree_nodes = collections.OrderedDict(sorted(mod_degree_nodes.iteritems(),key=operator.itemgetter(1),reverse=True));
    sorted_degree_nodes = OrderedDict(sorted(mod_degree_nodes.iteritems(),key=operator.itemgetter(1),reverse=True)); 
    del degree_nodes;
    #Time required for this operation is O(n)
    mod_adjacency_matrix = dict((k,v) for (k,v) in adjacency_matrix.iteritems() if v[0]>median_degree);
    del adjacency_matrix;
    fp.close();

    #We have obtained the dictionary for degree and adjacency_matrix and perform the algorithm
    temp_degree_nodes,temp_adjacency_matrix = {},{};
    selected_points = [];
    iteration,count=0,0;
    #Time required for this operation is O(n)
    if (len(mod_degree_nodes)<represent_points):
        print "Tune the model by decreasing representative number of points";

    print("First Step of the Sampling Process Completed");
    #The main algorithm goes here
    while(iteration<represent_points):
        if (bool(mod_degree_nodes)==0):				#Time required for this operation is O(1)
            count=count+1;
            print "Degree Matrix Empty for the %d time" %(count);
            if (bool(temp_degree_nodes)==0):				    #Time required for this operation is O(1)
                print "Tune the model by decreasing representative number of points";
            else:
	        del mod_degree_nodes,mod_adjacency_matrix;
                mod_degree_nodes,mod_adjacency_matrix = {},{};
                #mod_degree_nodes = copy.deepcopy(temp_degree_nodes);
                mod_degree_nodes = degree_copy.degree_copy(temp_degree_nodes);
	        del temp_degree_nodes;
                #mod_adjacency_matrix = copy.deepcopy(temp_adjacency_matrix);
                mod_adjacency_matrix = adjacency_copy.adjacency_copy(temp_adjacency_matrix);
	        del temp_adjacency_matrix,sorted_degree_nodes;
                #sorted_degree_nodes=collections.OrderedDict(sorted(mod_degree_nodes.iteritems(),key=operator.itemgetter(1),reverse=True));
                sorted_degree_nodes=OrderedDict(sorted(mod_degree_nodes.iteritems(),key=operator.itemgetter(1),reverse=True));
                temp_degree_nodes,temp_adjacency_matrix = {},{};
        else:
            degree_tuple = sorted_degree_nodes.popitem(False);                       #Time required for this operation is O(1)
	    adjacency_value = mod_adjacency_matrix[degree_tuple[0]];			         #Time required for this operation is O(1)
            adjacency_list = adjacency_value[1:len(adjacency_value)]; 			         #Time required for this operation is 0(1)
            selected_points.append((degree_tuple[0],degree_tuple[1],len(adjacency_list))); #Time required for this operation is O(1)
            del mod_degree_nodes[degree_tuple[0]];
            del mod_adjacency_matrix[degree_tuple[0]];
            iteration = iteration + 1;
            #print iteration;
            for node_index in adjacency_list:                                              #Look at the adjacency_list to start de-activating in O(k)
 	    	if (node_index in mod_degree_nodes):					                    #If node is active in a dictionary in O(1)
            	    degree_value = mod_degree_nodes[node_index];
            	    adjacency_value = mod_adjacency_matrix[node_index];
            	    temp_degree_nodes[node_index]=degree_value;                             #Put it in a temporary degree matrix in O(1);
            	    temp_adjacency_matrix[node_index]=adjacency_value;                      #Put it in a temporary adjacency_matrix in O(1);
            	    del mod_adjacency_matrix[node_index];					#Removal in O(1)
	    	    del mod_degree_nodes[node_index];					    #Removal in O(1)
	            del sorted_degree_nodes[node_index];
    selected_points = sorted(selected_points, key = operator.itemgetter(1), reverse=True);
    mod_degree_nodes.clear();
    mod_adjacency_matrix.clear();
    del sorted_degree_nodes;
    
    
    fout=open(filename+"_out.txt",'w');    
    for data in selected_points:
        outputstring = str(data[0])+"\t"+str(data[1])+"\t"+str(data[2])+"\n";
        fout.write(outputstring);
    fout.close();
    print time.time()-start_time, "seconds";
    
    fout=open(filename+"_out.txt",'r');  
    S = nx.read_edgelist(fout, nodetype=int, data=(('weight',float),))
    fout.close();    
    print S.nodes()
    print "%d nodes" %S.number_of_nodes()
    print S.edges(data = False)
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
    rootdir = parent_path + s + "Networks_with_ground_truth_communities" + s+ 'football';

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
                sample_rate= 0.2
                delimeter = '\t'
                numNodes = len(nx.nodes(original_))
                sample_size = int(math.ceil(float(numNodes)*float(sample_rate)))  
                sample_ = fast_subset_degree_sampling(full_name,sample_size,delimeter);                    
                fh=open("test.edgelist",'wb')
#                nx.write_edgelist(S,fh,data=False)
                nx.write_edgelist(sample_, parent+os.sep+"network_sample_p"+str(int(100*float(sample_rate)))+"_v1.dat", data=False)   
                
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

