'''
Created on Aug 30, 2013

@author: Emrah Cem{emrah.cem@utdallas.edu}
'''
import argparse
import os
import shutil
import errno
import subprocess
import networkx as nx
import random
import time
import math
from sampling.sampling_algorithms import *
import top_k_snowball_sampling_algorithm_balanced 
import top_k_snowball_sampling_algorithm_internal_priority 
import top_k_snowball_sampling_algorithm_internal_priority_extra_condition_for_initialization
import snowball_expansion_sampler
import fast_subset_degree_sampler

import SamplingAlgorithms
from numpy import random
from collections import deque
#from top_k_snowball_sampling_algorithm_internal_priority import *
#from top_k_snowball_sampling_algorithm_balanced import *
#from top_k_snowball_sampling_algorithm_internal_priority_extra_condition_for_initialization.py import *

import analytics

__all__=['add_path_to_graph','add_node_to_graph','add_edge_to_graph','generate_edge']

def handleArgs():
    """Handle command-line input arguments."""

    parser = argparse.ArgumentParser(description="Sample graphs.")
    parser.add_argument("-n", "--nodes", type=int, required=True, help="the number of nodes", dest="N")
    parser.add_argument("-s", "--start", default=1, type=int, help="the file number at which to start, inclusive", dest="start")
    parser.add_argument("-e", "--end", default=10, type=int, help="the file number at which to end, inclusive", dest="end")
    parser.add_argument("-o", "--output", default="generated_benches/", help="the output path, defaults to 'generated_benches/'", dest="out_directory_stem")
    parser.add_argument("-percentages", "--sample_percentage", nargs="+", default=[0.1, 0.3, 0.5, 0.7], help="Sample percentage of the whole graph", dest="percentages")
    parser.add_argument("-sac","--sampling_condition", default="", help="choose the sampling algorithm", dest="sampling_condition")
    parser.add_argument("-nc", "--numcluster", default = 10, type=int, help="the numbers of nodes", dest="numcluster")  
    global args
    args = parser.parse_args()


def generate_edge(G, with_replacement):
    if with_replacement:
        while True:
            yield random.choice(G.edges())
    else:
        edge_list=random.sample(G.edges(),G.number_of_edges())#this will shuffle edges
        for e in edge_list:
            yield e
         
def add_path_to_graph(G,path):
    if len(path)==1:
        add_node_to_graph(G,path[0])
    else:
        G.add_path(path)
        for n in path:
            G.node[n]['times_selected']=G.node[n].get('times_selected',0)+1
        u=path[0]
        for v in path[1:]:
            G.edge[u][v]['times_selected']=G.edge[u][v].get('times_selected',0)+1
            u=v
        G.graph['number_of_nodes_repeated']=G.graph.get('number_of_nodes_repeated',0)+len(path)
        G.graph['number_of_edges_repeated']=G.graph.get('number_of_edges_repeated',0)+len(path)-1
        
def add_node_to_graph(G,n):
    G.add_node(n)
    G.node[n]['times_selected']=G.node[n].get('times_selected',0)+1
    G.graph['number_of_nodes_repeated']=G.graph.get('number_of_nodes_repeated',0)+1

def add_edge_to_graph(G,e, add_nodes=True):
    G.add_edge(*e)
    G.edge[e[0]][e[1]]['times_selected']=G.edge[e[0]][e[1]].get('times_selected',0)+1
    if add_nodes:
        add_node_to_graph(G, e[0])
        add_node_to_graph(G, e[1])
    G.graph['number_of_edges_repeated']=G.graph.get('number_of_edges_repeated',0)+1

def createPathIfNeeded(path):
    """Credits to user 'Heikki Toivonen' on SO: http://stackoverflow.com/questions/273192/check-if-a-directory-exists-and-create-it-if-necessary"""
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise    

def sampleCommunities(sample, clustering_file, write_file, separator):
    """Given a network file separated by separator, removes edges such that the final network file_name
    contains no two edges that connect the same pair of nodes.
    Assumes node ids and cluster ids are integers.
    If assume_one_max, the function will assume that there are at most two
    edges in the original file connecting the same pair of nodes."""
    sample_nodes=set(sample.nodes())
    read_file = clustering_file
    #write_file ="sample_"+read_file
    #assert not os.path.isfile(write_file)

    with open(read_file, 'r') as read_f:
        with open(write_file, 'wb') as write_f:
            for line in read_f:
                node_id, cluster_id = line[:-1].split(separator)
                node_id = int(node_id)
                if  node_id in sample_nodes:
                    write_f.write(str(node_id) + separator + str(cluster_id) + '\n')                 
    
    #shutil.move(write_file, read_file)  
    
#if __name__ == "__main__":
#    handleArgs()
#    createPathIfNeeded(args.out_directory_stem)
  
#    for i in xrange(args.start, args.end + 1):
        # Does seed file need to be handled here?
        ###zjp add sampling stragegy
#        sample=induced_random_vertex_sampler(G, sample_size, with_replacement=False)
#        fh=open(args.out_directory_stem + 'network_sample_v' + str(i) + '.dat','wb')
#        nx.write_edgelist(sample, fh)
#        G=nx.read_edgelist(args.out_directory_stem + 'network_v' + str(i) + '.dat', nodetype=int)
 
#    shutil.move(args.bench_directory_stem + flag_file_name, args.out_directory_stem + "flags.dat")
    
if __name__ == "__main__":

    handleArgs()
    createPathIfNeeded(args.out_directory_stem)
    #the number of cluster
    num_of_cluster=args.numcluster   
    for i in xrange(args.start, args.end + 1):
      
        ###zjp add sampling stragegy
        re=open(args.out_directory_stem + 'network_v' + str(i) + '.dat', 'rb')
        G=nx.read_edgelist(re, nodetype=int, create_using=nx.Graph())
        re.close() 
        
        #G = nx.read_edgelist(path=readfile, delimiter=",", nodetype=int,  create_using=nx.Graph())
        # start = 0
        # G_ = nx.convert_node_labels_to_integers(G, first_label=start)
        # numNodes = len(nx.nodes(G_))
        
        numNodes = nx.number_of_nodes(G)
#        percentages = [0.1, 0.3, 0.5, 0.7]
        sampling_conditions = ['induced_random_edge','induced_random_vertex','induced_weighted_random_vertex','kk_path','km_path','random_path','random_vertex','random_edge','random_walk','metropolis_subgraph','metropolis_subgraph_CC','metropolized_random_walk','weighted_vertex','top_k_leader_balanced','top_k_leader_internal_priority','top_k_leader_internal_priority_extra','snowball_expansion','fast_subset_degree','multiple_random_walk','forest_fire_sample','random_walk_seed','random_walk_jump','random_walk_restart']
        
        #sampling strategy
        #sampling_condition = str(sampling_conditions[10])
        sampling_condition = args.sampling_condition
        if not sampling_condition in sampling_conditions:
            raise ValueError('Invalid sampling algorithms')
        for val in args.percentages:       
            sample_size = int(math.ceil(float(numNodes)*float(val)))        
            t=time.time()
            if sampling_condition=='induced_random_vertex':
                sample = induced_random_vertex_sampler(G, sample_size, with_replacement=False)

            elif sampling_condition=='induced_random_edge':
                sample = induced_random_edge_sampler(G, sample_size, stopping_condition='UNIQUE_NODES', with_replacement=True)

            elif sampling_condition=='induced_weighted_random_vertex':#invalid
                sample = sampling.sampling_algorithms.induced_weighted_random_vertex_sampler(G, sample_size, weights=None, with_replacement=True)

            elif sampling_condition=='kk_path':#invalid                
                sample = kk_path_sampler(G, sample_size, K=None, vantage_points=None, stopping_condition='UNIQUE_NODES', fuzzy_select=True, include_last_path_when_exceeds=True)
            
            elif sampling_condition=='km_path':#invalid
                sample = km_path_sampler(G, sample_size, K=None, M=None, source_nodes=None, destination_nodes=None, source_destination_nodes_can_overlap=False, stopping_condition='UNIQUE_NODES', fuzzy_select=True, include_last_path_when_exceeds=True)
            
            elif sampling_condition=='random_path':
                sample = random_path_sampler(G, sample_size, stopping_condition='UNIQUE_NODES', include_last_path_when_exceeds=True)
            
            elif sampling_condition=='random_vertex': #just nodes 
                sample = random_vertex_sampler(G, sample_size, with_replacement=False)
           
            elif sampling_condition=='random_edge':
                sample = random_edge_sampler(G, sample_size, stopping_condition='UNIQUE_NODES', with_replacement=True, include_last_edge_when_exceeds=True)
            
            elif sampling_condition=='random_walk':
                sample = random_walk_sampler(G, sample_size, initial_node=None, stopping_condition='UNIQUE_NODES', metropolized=False, excluded_initial_steps=0)
            
            elif sampling_condition=='metropolis_subgraph':
                p=10*G.number_of_edges()*log10(G.number_of_nodes())/G.number_of_nodes() 
                best, div=metropolis_subgraph_sampler(G, sample_size, analytics.DivergenceMetrics.JensenShannonDivergence, smp.SimpleGraphDegree(), 1000, p, 10, 2)
                # best, div=metropolis_subgraph_sampler(G, sample_size, analytics.DivergenceMetrics.JensenShannonDivergence, smp.SimpleGraphClusteringCoefficient(), 1000, p, 10, 2)
                sample = best

            elif sampling_condition=='metropolis_subgraph_CC':
                p=10*G.number_of_edges()*log10(G.number_of_nodes())/G.number_of_nodes() 
                # best, div=metropolis_subgraph_sampler(G, sample_size, analytics.DivergenceMetrics.JensenShannonDivergence, smp.SimpleGraphDegree(), 1000, p, 10, 2)
                best, div=metropolis_subgraph_sampler(G, sample_size, analytics.DivergenceMetrics.JensenShannonDivergence, smp.SimpleGraphClusteringCoefficient(), 1000, p, 10, 2)
                sample = best
            elif sampling_condition=='metropolized_random_walk':
                sample = metropolized_random_walk_sampler(G, sample_size, stopping_condition='UNIQUE_NODES', excluded_initial_steps=0)

            elif sampling_condition=='weighted_vertex':
                sample = weighted_vertex_sampler(G, sample_size, weights, with_replacement=True)

            elif sampling_condition=='top_k_leader_balanced':
                option_conditions_in_initialization = []
#                num_of_cluster=5000
                #option_conditions_in_initialization = ['--not_direct_neighbors']
                sample = top_k_snowball_sampling_algorithm_balanced.top_k_snowball_sampling_algorithm_balanced(G, float(val), num_of_cluster, option_conditions_in_initialization)
            
            elif sampling_condition=='top_k_leader_internal_priority':
#                num_of_cluster=5000
                #option_conditions_in_initialization = ['--not_direct_neighbors']
                option_conditions_in_initialization = []
                sample = top_k_snowball_sampling_algorithm_internal_priority.top_k_snowball_sampling_algorithm_internal_priority(G, float(val), num_of_cluster, option_conditions_in_initialization)
            
            elif sampling_condition=='top_k_leader_internal_priority_extra':
#                num_of_cluster=12
                #option_conditions_in_initialization = ['--not_direct_neighbors']
                option_conditions_in_initialization = []
                sample = top_k_snowball_sampling_algorithm_internal_priority_extra_condition_for_initialization.top_k_snowball_sampling_algorithm_internal_priority_extra(G, float(val), num_of_cluster, option_conditions_in_initialization)
            
            elif sampling_condition=='snowball_expansion':
                print 'ttttttttttt'
                sample = snowball_expansion_sampler.snowball_expansion_sampling(G, float(val))  

            elif sampling_condition=='fast_subset_degree':
                delimeter = '\t'
                sample_size = int(math.ceil(float(numNodes)*float(val)))  
                sample = fast_subset_degree_sampler.fast_subset_degree_sampling(args.out_directory_stem + 'network_v' + str(i) + '.dat',sample_size,delimeter);  

            elif sampling_condition=='forest_fire_sample':
                sample_size = int(math.ceil(float(numNodes)*float(val)))  
                sample = SamplingAlgorithms.forest_fire_sampler(G, sample_size)
                print nx.info(sample)

            elif sampling_condition=='multiple_random_walk':
                sample_size = int(math.ceil(float(numNodes)*float(val)))  
                nbunch_mrw = SamplingAlgorithms.multiRandomWalk(G, sample_size)
                sample = G.subgraph(nbunch_mrw)
                print nx.info(sample) 

            elif sampling_condition=='random_walk_seed':
                sample_size = int(math.ceil(float(numNodes)*float(val)))  
                nbunch_rw = SamplingAlgorithms.randomWalk(G,sample_size)
                sample = G.subgraph(nbunch_rw)
                print nx.info(sample) 

            elif sampling_condition=='random_walk_jump':

                sample_size = int(math.ceil(float(numNodes)*float(val))) 
                print("numnodes:"+str(numNodes))
                print("numnodes:"+(val))                                 
                nbunch_rwj = SamplingAlgorithms.randomWalk_with_jump(G, sample_size)
                sample = G.subgraph(nbunch_rwj)
                print nx.info(sample) 

            elif sampling_condition=='random_walk_restart':

                sample_size = int(math.ceil(float(numNodes)*float(val)))                  
                nbunch_rwj = SamplingAlgorithms.random_walk_restart(G, sample_size)
                sample = G.subgraph(nbunch_rwj)
                print nx.info(sample) 
            else:
                print("pleace checke the name of sampling algorithm")

            print time.time()-t  
            print 'number of unique nodes:',sample.number_of_nodes()
            print 'number of unique edges:',sample.number_of_edges()
            print 'number of nodes:',sample.graph.get('number_of_nodes_repeated',0)
            print 'number of edges:',sample.graph.get('number_of_edges_repeated',0)
#            print 'nodes',sample.nodes()
        
        #def sampleCommunities(sample,clustering_file, separator):
            sampleCommunities(sample,args.out_directory_stem + 'community_v' + str(i) + '.dat', args.out_directory_stem + 'community_sample_p'+str(int(100*float(val)))+'_v' + str(i) + '.dat','\t')
            we=open(args.out_directory_stem + 'network_sample_p'+str(int(100*float(val)))+'_v' + str(i) + '.dat','wb')
            nx.write_edgelist(sample, we ,data=False)
            we.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

 
   
