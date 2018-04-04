'''
Created on Aug 30, 2013

@author: Emrah Cem{emrah.cem@utdallas.edu}
'''

import copy
import string
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
import analytics
from collections import Counter
from math import *
import matplotlib.pyplot as plt
import numpy as np
import DivergenceMetrics2 as dm
import community
import supervised_evaluation as se
import vbmod as vd
import sampling as smp
import pandas as pd
from analytics.divergence import *
import igraph
__all__=['add_path_to_graph','add_node_to_graph','add_edge_to_graph','generate_edge']

def handleArgs():
    """Handle command-line input arguments."""

    parser = argparse.ArgumentParser(description="Sample graphs.")
#    parser.add_argument("-n", "--nodes", type=int, required=True, help="the number of nodes", dest="N")
#    parser.add_argument("-s", "--start", default=1, type=int, help="the file number at which to start, inclusive", dest="start")
#    parser.add_argument("-e", "--end", default=10, type=int, help="the file number at which to end, inclusive", dest="end")
    parser.add_argument("-i", "--input", default="original_snapshot_merge/", help="the input path, defaults to 'generated_benches/'", dest="input_directory_stem")
    parser.add_argument("-o", "--output", default="powlaw_degree_benchmark_results/", help="the output path, defaults to powlaw_degree_benchmark_results", dest="output_directory_stem")
    parser.add_argument("-percentages", "--sample_percentage", nargs="+", default=[0.1, 0.3, 0.5, 0.7], help="Sample percentage of the whole graph", dest="percentages")
    parser.add_argument("-sa","--sampling_algorithm", default="", help="choose the sampling algorithm", dest="sampling_algorithm")
    parser.add_argument("-ca","--clustering_algorithm", default="", help="choose the sampling algorithm", dest="clustering_algorithm")
    parser.add_argument("-ds","--data_set",  default="", help="choose the data set", dest="ds")
    parser.add_argument("-ss", "--snapshot", default=1, type=int, help="the number of snopshots", dest="snapshot")
    global args
    args = parser.parse_args()

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def log_binning(counter_dict,bin_count=35):

    max_x = log10(max(counter_dict.keys()))
    max_y = log10(max(counter_dict.values()))
    max_base = max([max_x,max_y])

    min_x = log10(min(drop_zeros(counter_dict.keys())))

    bins = np.logspace(min_x,max_base,num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    bin_means_y = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.values())[0] / np.histogram(counter_dict.keys(),bins)[0])
    bin_means_x = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.keys())[0] / np.histogram(counter_dict.keys(),bins)[0])

    return bin_means_x,bin_means_y

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

def removeDuplicateEdges(filename, separator, assume_one_max = False):
    """Given a network file separated by separator, removes edges such that the final network file_name
    contains no two edges that connect the same pair of nodes.
    Assumes node ids and cluster ids are integers.
    If assume_one_max, the function will assume that there are at most two
    edges in the original file connecting the same pair of nodes."""

    read_file = filename
    os.remove("temporary_function_execution.dat")
    write_file = "temporary_function_execution.dat"
#    assert not os.path.isfile(write_file)

    with open(read_file, 'r') as read_f:
        with open(write_file, 'wb') as write_f:
            redundant_edges = {}
            empty_set = set() 
            for line in read_f:
                source, destination = line.split(separator)
                source = int(source)
                destination = int(destination.rstrip()) # remove newline character and trailing spaces
                if not destination in redundant_edges.get(source, empty_set):
                    write_f.write(str(source) + separator + str(destination) + '\n')
                    redundant_edges[destination] = redundant_edges.get(destination, empty_set)
                    redundant_edges[destination].add(source)
                    empty_set = set() # reverse mutation due to previous line
                elif assume_one_max:
                    redundant_edges[source].remove(destination)
    
    #shutil.move(read_file, read_file + 'a')
    shutil.move(write_file, read_file)
    
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

#./bigclam -i:'./as20graph.txt' -c:200  -o:'jianpeng'
# def runBigclam(input_file, number_of_clusters, output_prefix):
#     """Normalized Mutual Information as defined here: https://sites.google.com/site/andrealancichinetti/mutual"""

#     result_lines = []
#     bigclam_directory="./bigclam"
#     process = subprocess.Popen(['./bigclam', '-i:', input_file, '-c:', str(number_of_clusters), '-o', output_prefix], cwd=bigclam_directory, stdout=subprocess.PIPE)
# #    output = process.communicate()[0].split()
# #    value = float(output[1])

#     print input_file + " completed"

def parseBigclamResults(output_prefix, out_file_separator):
    output_result=output_prefix+'cmtyvv.txt'
    commsfile = output_prefix + '.comms'
    # clustering file in (f_path + 'results_1/tp')

    read_file = open(output_result, 'r')
    write_file = open(commsfile, 'wb')

    cluster_number_string = '1'
    for line in read_file:
        if line[0] == '#':
            continue
        nodes = line.split()
        for node in nodes:
            write_file.write(node + out_file_separator + cluster_number_string + '\n')
        cluster_number_string = str(int(cluster_number_string) + 1)

    print('\nSuccessfully ran clustering and wrote results to file ' + output_result + 'to' + commsfile + '\n')

    read_file.close()
    write_file.close()

def runLancichNormMutInf(gold_standard_vector, partition_vector):
    """Normalized Mutual Information as defined here: https://sites.google.com/site/andrealancichinetti/mutual"""

    result_lines = []

    gold_standard_cluster_name = "file1.dat"
    partition_cluster_name = "file2.dat"

    writeLineDefinedClusterFile(gold_standard_vector, args.lnmi_directory + gold_standard_cluster_name)
    writeLineDefinedClusterFile(partition_vector, args.lnmi_directory + partition_cluster_name)

    process = subprocess.Popen(['./mutual', gold_standard_cluster_name, partition_cluster_name], cwd=args.lnmi_directory, stdout=subprocess.PIPE)
    output = process.communicate()[0].split()
    assert output[0] == 'mutual3:'
    value = float(output[1])

    os.remove(args.lnmi_directory + gold_standard_cluster_name)
    os.remove(args.lnmi_directory + partition_cluster_name)

    result_lines.append(['Lancichinetti NMI', 'Entire Graph', value])

    return result_lines

def runGMapAnalysis(relative_dotfile_path):
    """"""

    result_lines = []

    modularity = None
    conductance = None
    coverage = None

    gmap_metric_directory = args.gmap_directory + 'external/eba'
    absolute_dotfile_path = os.getcwd() + '/' + relative_dotfile_path
    order_kmeans = ' '.join(['./kmeans', '-action=metrics', '-o=metric_results.txt', absolute_dotfile_path]) 
    subprocess.call(['./kmeans', '-action=metrics', '-o=metric_results.txt', absolute_dotfile_path], cwd = gmap_metric_directory)
    gmap_output_path = gmap_metric_directory + '/metric_results.txt'
    f = open(gmap_output_path, 'r')
    for line in f:
        pieces = line.split()
        metric = pieces[0]
        value = pieces[1]
        if metric[:10] == 'Modularity':
            result_lines.append(['Modularity', 'Entire Graph', value])
        elif metric[:11] == 'Conductance':
            try:
                float(value)
                result_lines.append(['Conductance', 'Entire Graph', value])
            except ValueError:
                assert value == 'undefined'
                logfile_lines.append('Undefined conductance for ' + relative_dotfile_path)
                result_lines.append(['Conductance', 'Entire Graph', 0.0])
        elif metric[:8] == 'Coverage':
            result_lines.append(['Coverage', 'Entire Graph', value])

    f.close()
    os.remove(gmap_output_path)

    return result_lines

#./bigclam -i:'./as20graph.txt' -c:200  -o:'jianpeng'
def runBigclam(input_file, number_of_clusters, outupt_prefix):
    """Normalized Mutual Information as defined here: https://sites.google.com/site/andrealancichinetti/mutual"""

    result_lines = []
#    bigclam_directory="./bigclam/"
    bigclam_directory="."
    aaa= '-i:'+input_file
    bbb= '-c:'+str(number_of_clusters)
    ccc= '-o:'+outupt_prefix
#    order_list= './bigclam -i'+':'+input_file+' -c'+':'+str(number_of_clusters), '-o'+':'+outupt_prefix
    order_list=' '.join(['./bigclam2', aaa, bbb, ccc])
    print order_list
#    process = subprocess.Popen(['./bigclam2', '-i:', input_file, '-c:', str(number_of_clusters), '-o', outupt_prefix], cwd=bigclam_directory, stdout=subprocess.PIPE)
    process = subprocess.Popen(['./bigclam2', aaa, bbb, ccc], cwd=bigclam_directory, stdout=subprocess.PIPE)
    output = process.communicate()
#    print output
#    print '\n\n\n'
    print input_file + " completed"

#    retcode=subprocess.call(['./bigclam2', '-i:', input_file, '-c:', str(number_of_clusters), '-o', outupt_prefix])
#    print retcode    
    #process = subprocess.Popen([order_list], cwd=bigclam_directory, stdout=subprocess.PIPE)
    # output = process.communicate()[0].split()
    # value = output[1]


###add by zhukaijie#####
def expand_partition_with_other_components(part, graph, main_components):
    max_cluster_id = -1
    for i in part:
        if (part[i] > max_cluster_id):
            max_cluster_id = part[i]
            
    connectedComponents = nx.number_connected_components(graph)
    #print "connectedComponents = %d" %connectedComponents
    
    current_cluster_id = max_cluster_id + 1
    if connectedComponents > 1:  
        for comp in nx.connected_component_subgraphs(graph):
            compNodesList = comp.nodes()
            
            maincompNodesList = main_components.nodes()
            if( compNodesList[0] in  maincompNodesList): #indicating this comp is the main comp
                continue;
                #print "this comp is the main one!"
            else:
                #print compNodesList
                for node in compNodesList:
                    part[node] = current_cluster_id
                    
            current_cluster_id = current_cluster_id + 1

def parseLancichinettiResults(input_file, out_file):

    # clustering file in (f_path + 'results_1/tp')

    read_file = open(input_file, 'r')
    write_file = open(out_file, 'wb')

    cluster_number_string = '1'
    for line in read_file:
        if line[0] == '#':
            continue
        nodes = line.split()
        for node in nodes:
            write_file.write(node + ' ' + cluster_number_string + '\n')
        cluster_number_string = str(int(cluster_number_string) + 1)

    #print('\nSuccessfully ran transfering and wrote results to file ' + out_file + '\n')

    read_file.close()
    write_file.close() 

def snapshotCommunities(full_partition_prefix, part):
    """write the partition of snapshot t into separate file."""
    comms = []
    for node in part.keys():
        comms.append(str(node) + ' ' + str(part[node]) + '\n')

    commsfile = full_partition_prefix + '.comms'
    fp_commsfile = open(commsfile, 'w')
    for line in comms:
        fp_commsfile.write(line)
    fp_commsfile.close() 

    print commsfile + " completed"
    
def writeFullPartitionToFile(complete_part, total_nodes, file_name):    

    fp_complete_commsfile = open(file_name, 'w')
    
    l = 0
    while ( l < total_nodes):
        fp_complete_commsfile.write( str(l) + ' ' + str(complete_part[l]) + '\n')
        l = l + 1
        
    fp_complete_commsfile.close() 


def writePartitionToFile(part, file_name):
    fileObject = open(file_name, 'w')  
    for node_affiliation in part:  
        fileObject.write(str(node_affiliation)+' '+ str(part[node_affiliation]))  
        fileObject.write('\n')  
    fileObject.close() 

def writePartitionToFile_t(part, file_name):
    fileObject = open(file_name, 'w')  
    for node_affiliation in part:  
        fileObject.write(str(node_affiliation)+'\t'+ str(part[node_affiliation]))  
        fileObject.write('\n')  
    fileObject.close() 

    # for i in part: 
    #     print "dict[%s]=" % i,part[i] 
    #     fileObject.write(i+' '+ part[i])  
    #     fileObject.write('\n') 

    # print "###########items#####################" 
    # for (k,v) in  part.items:  .items(): 
    #     print "dict[%s]=" % k,v 
    #     fileObject.write(k+' '+ v)  
    #     fileObject.write('\n') 
     
    # print "###########iteritems#################" 
    # for k,v in part.iteritems(): 
    #     print "dict[%s]=" % k,v 
    #     fileObject.write(k +' '+ v)  
    #     fileObject.write('\n')              
    # print "###########iterkeys,itervalues#######" 
    # for k,v in zip(part.iterkeys(),part.itervalues()): 
    #     print "dict[%s]=" % k,v 
    #     fileObject.write(k+' '+ v)  
    #     fileObject.write('\n') 

def distribution_save(H, K, file_name, cdf=False):
        H,K=Divergence.compute(H, K)
        try:
            if cdf:
                if not (analytics.isValidCDF(H) and analytics.isValidCDF(K)):
                    raise smp.InvalidProbabilityDistributionException('Invalid cumulative distribution')
            if not cdf:
                if not (analytics.isValidPDF(H) and analytics.isValidPDF(K)):
                    raise smp.InvalidProbabilityDensityException('Invalid probability density')
                #convert to cdf
                H=analytics.pdfTocdf(H)
                K=analytics.pdfTocdf(K)
            
            all_keys= dict(H.items()+K.items()).keys()
            analytics.fill_gaps(H, all_keys, cdf=True)
            analytics.fill_gaps(K, all_keys, cdf=True)
            fileObject = open(file_name, 'w')  
            
            for key in sorted(all_keys):                
                fileObject.write(str(key)+' '+ str(K[key])+'\n')  
#                print("dict[%s] =" % key,K[key])
            fileObject.close()            
#==============================================================================
#             plot_keys=[]
#             plot_values=[]        
#             fileObject = open(file_name, 'w')  
#             for key in sorted(all_keys):
#                 plot_keys.append(key)
#                 plot_values.append(K[key])
#                 
#                 fileObject.write(str(key)+' '+ str(H[key])+'\n')  
#                 print("dict[%s] =" % key,H[key])
#             fileObject.close()
#             
#             plt.figure(1)
#             plt.plot(plot_keys, plot_values, 'r-')
#             plt.xscale('log')
#             plt.yscale('log')
#             plt.ylim([0,1])
#             plt.ylabel('CDF')
#             plt.xlabel('Degree')
#             plt.savefig(file_name+ '_cdf.pdf')
#             plt.clf()
#==============================================================================
        
        except smp.InvalidProbabilityDistributionException:
            logging.exception('Invalid probability distribution')
        except smp.InvalidProbabilityDensityException:
            logging.exception('Invalid probability density')

def deleteFileIfNeeded(file):
    try:
        os.remove(file)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
def runKcore(input_file, minimal_of_k, output_prefix):
# Plot the number of nodes in a k-core of a graph as a function of k.
# A subgraph H = (C,E|C) induced by the set C subset of V is a k-core or a 
# core of order k if and only if the degree of every node v in C induced in 
# H is greater or equal than k, and H is the maximum subgraph with this 
# property.
# A k-core of G can be obtained by recursively removing all the vertices of 
# degree less than k, until all vertices in the remaining graph have degree
# at least k.
# Parameters:
#    -i:Input undirected graph file (single directed edge per line) (default:'../as20graph.txt')
#    -k:Minimal clique overlap size (default:3)
#    -o:Output file prefix (default:'')
# Usage:
# Enumerate the communities in the AS graph:
# ./kcores2 -i:../as20graph.txt -k:2 -o:as20

    result_lines = []
#    bigclam_directory="./bigclam/"
    Kcore_directory="."
    aaa= '-i:'+input_file
    bbb= '-k:'+str(minimal_of_k)
    ccc= '-o:'+output_prefix
#    order_list= './bigclam -i'+':'+input_file+' -c'+':'+str(number_of_clusters), '-o'+':'+outupt_prefix
    order_list=' '.join(['./kcores2', aaa, bbb, ccc])
    print order_list
#    process = subprocess.Popen(['./bigclam2', '-i:', input_file, '-c:', str(number_of_clusters), '-o', outupt_prefix], cwd=bigclam_directory, stdout=subprocess.PIPE)
    process = subprocess.Popen(['./kcores2', aaa, bbb, ccc], cwd=Kcore_directory, stdout=subprocess.PIPE)
    output = process.communicate()
#    print output
#    print '\n\n\n'
    print input_file + "kcores completed"

def compute_kcore_distribution(kcores_filename,partition_separator='\t',normalize=True):
    kcores={}
    # Write node id lines
    with open(kcores_filename, 'r') as partition_file:
        for line in partition_file:
            if line.startswith('#'):
                continue
            else:
                pieces = line.split(partition_separator)
                kcore_k = int(pieces[0])
                kcore_number_of_nodes = int(pieces[1].rstrip()) # remove newline character and trailing spaces
                kcores[kcore_k]=kcore_number_of_nodes  

    # print('kcores',kcores)             
    #dic = dict(Counter(kcores.values()))
    #print('dic',dic) 
    #iG=igraph.Graph(len(G), edges=G.edges())
    #dic = dict(Counter(iG.degree()))
    #if dic!=dic2:
    #    print 'dic:',dic
    #    print 'dic2:',dic2
    total = sum(kcores.values())
    return {k:float(v)/total for k,v in kcores.items()} if normalize else kcores

def createPathIfNeeded(path):
    """Credits to user 'Heikki Toivonen' on SO: http://stackoverflow.com/questions/273192/check-if-a-directory-exists-and-create-it-if-necessary"""
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

def deletePathIfNeeded(path):
    try:
        shutil.rmtree(path)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
def kCorePreProcess(comm_list, original_snapshot_graph, subgraph):

    unlabeled_nodes = 0
    unlabeled_list = []
    
    subgraph_nodes_list = subgraph.nodes()
    
    for i in range(1,len(comm_list)):

        if (comm_list[i] == -1):
            
            unlabeled_nodes = unlabeled_nodes + 1
            
            #calculate neighbor proportion
            neighbors_in_original_graph = original_snapshot_graph.neighbors(i)
            neighbors_in_subgraph = []
            for node in neighbors_in_original_graph:
                if node in subgraph_nodes_list:
                    neighbors_in_subgraph.append(node)
            proportion = len(neighbors_in_subgraph) / len(neighbors_in_original_graph)
        
            #generate triples
            element = (i, proportion,  neighbors_in_original_graph)
            unlabeled_list.append(element)
          
    sorted_unlabeled_list =  sorted(unlabeled_list, key=lambda item:item[1], reverse = True )       

    for element in sorted_unlabeled_list:    
#    for element in unlabeled_list:
        node = element[0]
        proportion = element[1]
        neighbors_in_original_graph = element[2]
                
        #generate_neighbors_list
        neighbors_in_subgraph = []
        for neighbor in neighbors_in_original_graph:
            if (comm_list[neighbor] != -1):
                neighbors_in_subgraph.append(neighbor)
        
        #count label of neighbors
        labels_dic = {}
        for neighbor in neighbors_in_subgraph:
            label = comm_list[neighbor]
            # label = comm_list[i]
            if label in labels_dic:
                labels_dic[label] = labels_dic[label] + 1
            else:
                labels_dic[label] = 1
        if len(labels_dic) !=0:
            #select the most popular one as label of unlabeled node 
            labels_list = sorted(labels_dic.iteritems(), key=lambda item:item[1], reverse = True)
            selected_element = labels_list[0]
            selected_label = selected_element[0]
            comm_list[node] = selected_label

def buildCommunitieslist(part, total_nodes):
    
    pos = 0
    comm_list = []
    
    i = 0
    while (i < total_nodes):
        comm_list.append(-1)
        i = i + 1
        
    for node in part: 
        # print('node:',node)
        # print('part[node]:',part[node])
        # print('comm_list[node]:',comm_list[node])                   
        comm_list[node] = part[node]
    
    return comm_list
def buildFixedlist(comm_list):
    
    fixed_list = []
    for i in comm_list:
        if (i == -1):
            fixed_list.append(0)
        else:
            fixed_list.append(1) 
       
    return fixed_list 


def transform_community(input_file, output_file):
# tranform the community file to the designent file.
    fp_read = open(input_file)
    fp_write = open(output_file, 'w')
    
    for line in fp_read:
        line = line.replace(" \n", "\n")
        test = line.split('\t');
#        print test
        newline = ''
        flag = False
        for element in test:
            if (flag == True):
                newline = newline + ','
            newline = newline + element
            flag = True
            
        fp_write.write(newline)
    fp_write.close()


def transform_community_no_duplicate(input_file, output_file):
# tranform the community file to the designent file.
    fp_read = open(input_file)
    fp_write = open(output_file, 'w')
    node_list=set()
    
   
    for line in fp_read:

        line_list = line.rstrip().split('\t');
        # print line_list
        newline = ''

        node_id=line_list[0]
        cluster_id=line_list[1]


        if (node_id not in node_list):
            newline = node_id + ',' + cluster_id +'\n'
            fp_write.write(newline)                      
            node_list.add(node_id)

        else:
            print("this is an overlapping node")
            continue;
            
    fp_write.close()

def transform_edges(input_file, output_file):
# tranform the edge file to the designent file.
    fp_read = open(input_file)
    fp_write = open(output_file, 'w')
    
    for line in fp_read:
        # line = line.replace("\n", "")
        # test = line.split('\t');
        test = line.rstrip().split('\t');
        # print('test test line:%s'%test)
        newline = ''
        flag = False
        for element in test:
            if (flag == True):
                newline = newline + ','
            newline = newline + element
            flag = True
            
#        print newline    
        newline = newline + ',1' + '\n'
        fp_write.write(newline)   
    fp_write.close()


def create_arff(output_folder, filename, clu_file, edge_file, nc, cluster_flag):
# write the arff file.
    fp_write = open(output_folder+ filename, 'w')

    fp_write.write('@nodetype Person\n')
    fp_write.write('@attribute Name KEY\n')
    fp_write.write('@attribute Behaviour {')
    if cluster_flag==0:
        num_list = np.linspace(0,nc-1,nc)
    else:
        num_list = np.linspace(1,nc,nc)
    # print('num_list:%s' %num_list)
    newline =''
    flag = False
    for element in num_list:  
        if (flag == True):
            newline = newline + ','
        newline = newline + str(int(element))
        flag = True
    newline = newline.rstrip(',') 
    fp_write.write(newline)  
    fp_write.write('}\n')

    # fp_write.write('@nodedata community_v1.csv')
    fp_write.write('@nodedata ')
    fp_write.write(clu_file + '\n')


    fp_write.write('@edgetype Linked Person Person\n')
    fp_write.write('@Reversible\n')
    fp_write.write('@edgedata ')

    # fp_write.write('@edgedata ')
    fp_write.write(edge_file + '\n')
    fp_write.close()


def clusterByPopulation(output_folder, output_file, node_attibute, sample_clustering_file, inference_method, attribute_file):

    # deleteFileIfNeeded(output_folder + '/' + output_file)
    netkit_directory="./NetKit/"
    #netkit_directory="."
    mylist=['java', '-jar', 'NetKit-1.4.0.jar', '-attribute', node_attibute, '-known', sample_clustering_file, '-inferencemethod', inference_method, '-output', 'record', attribute_file]
    myorder=' '.join(mylist)
    print(myorder)
    process = subprocess.Popen(['java', '-jar', 'NetKit-1.4.0.jar', '-attribute', node_attibute, '-known', sample_clustering_file, '-inferencemethod', inference_method, '-output', 'record', attribute_file], cwd=netkit_directory, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    out, err = process.communicate()
    print 'out:'+out+'\n'
    print 'err:'+err+'\n'
    print output_file + "population inference completed"  
    ###move the record.predict
    shutil.move(netkit_directory+'record.predict', output_folder+'record.predict')


def create_PopClusteFile(output_folder, pre_filename, cluster_file, file_known):
# convert the clustering file.

    fp_read = open(output_folder+ pre_filename)


    fp_write = open(cluster_file, 'w')
    
    for line in fp_read:
        if (line[0] == '#'):
            continue         
        line = line.replace("\n", "")
        analysis = line.split(' ');
        node = ''
        min_frequency = '0'
        current_label = '0'
            
        for element in analysis: 
            if (element != ''):
                if ':' not in element:
                    node = element
                else:
                    label, frequency = element.split(':')
                    if (float(frequency) > float(min_frequency)):
                        current_label = label
                        min_frequency = frequency                
        newline = node + ' ' + current_label + '\n'
        fp_write.write(newline)

    # file_known = 'community_sample_p20_v1.csv'
    fp_known = open(file_known)
    for line in fp_known:
        knownnodes, knownlabel = line.rstrip().split(',')
        newline = knownnodes + ' '+ knownlabel + '\n'
        fp_write.write(newline)
    fp_write.close()
    fp_read.close()

def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

if __name__ == "__main__":

    handleArgs()

    # sampling_algorithm=['old_pies','new_pies','new_isolated_pies','streamES','streamNS','original_pies']
    # data_set=['merge','grow','mixed']
    # snapshot= 9
    # percentages = [0.15,0.30,0.45,0.60]       
    sampling_algorithm= args.sampling_algorithm
    percentages=args.percentages
    snapshot=args.snapshot
    data_set=args.ds
    # clustering_algorithms = ['vbmod','blondel','blondel_isolated_nodes','bigclam','bigclam_isolated_nodes']
    clustering_algorithm = args.clustering_algorithm
    deltas=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    ground_truth = 1 #ground-truth
    # ground_truth = 0 #no ground-truth
    # prefix_list=["birthdeath","expand","hide","mergesplit","switch"]
    # prefix= prefix_list[3]
    output_directory_stem= args.output_directory_stem #"./powlaw_degree_benchmark_results/"
    #rootdir = "../test/"
    suffix=".graph"
    repeats =2
#    percentages = [0.15]
    for per in percentages: 
        for rep in xrange(1, repeats):       
            results=[]  
            title_str=str('Snapshot'+','+'JS_degree'+ ','+'KS_degree'+','+'JS_CC'+ ','+'KS_CC'+','
            +'JS_path'+ ','+'KS_path'+','
            +'JS_kcores'+ ','+'KS_kcores'+','                
            +'1.0-precision'+ ','+'1.0-recall'+ ','+'0.9-precision'+ ','+'0.9-recall' + ','
            +'0.8-precision'+ ','+'0.8-recall'+ ','+'0.7-precision'+ ','+'0.7-recall' + ','
            +'0.6-precision'+ ','+'0.6-recall'+ ','+'0.5-precision'+ ','+'0.5-recall' + ','
            +'0.4-precision'+ ','+'0.4-recall'+ ','+'0.3-precision'+ ','+'0.3-recall' + ','
            +'0.2-precision'+ ','+'0.2-recall'+ ','+'0.1-precision'+ ','+'0.1-recall' + ','
            +'0-precision'+ ','+'0-recall'+','
            +'ANC'+ ','+'NLS' + ','
            +'NMI'+ ','+'ARS' + ','+'VNMI'+','
            +'#S_edges' + ','
            +'#S_nodes' +','+ '#S_components' +','
            +'S_eccentricity'+ ','+'S_radius'+','
            +'S_modularity'+','
            +'S_density' + ','+ 'S_clusterCoeff' +','
            +'#S_clusters'+ ','
            +'S_mediate_size_clusters'+ ','
            +'#Orig_edges' + ','
            +'#Orig_nodes' +','+ '#Orig_components' +','
            +'Orig_eccentricity'+ ','+'Orig_radius'+','
            +'Orig_modularity'+','
            +'Orig_density' + ','+ 'Orig_clusterCoeff' +','
            +'#Orig_clusters'+ ','+ 'Time_elipse'
            +'\n')
            results.append(title_str) 
          
            num_nodes_snapshots=[]
            #for LP inference, only the last snapshot is caculated.
            # if clustering_algorithm == 'LP_complete' and data_set != 'facebookYY':
            #     start_snapshot =snapshot-1
            # elif clustering_algorithm == 'LP_complete' and data_set == 'facebookYY':
            #     start_snapshot =snapshot-15           
            # else:
            #     start_snapshot =0
            if clustering_algorithm == 'LP_complete':
                start_snapshot =snapshot-1    
            else:
                start_snapshot =0

            for i in xrange(start_snapshot, snapshot): 

                t=time.time()                    
                ge=open(args.input_directory_stem + 'network_v1.dat', 'rb')
                original_=nx.read_edgelist(ge, nodetype=int, create_using=nx.Graph())
                ge.close()
                igr = igraph.Graph(n = original_.number_of_nodes(), edges = nx.convert_node_labels_to_integers(original_).edges())     

                orig_components = nx.number_connected_components(original_)
                print "orig: num components: "+str(orig_components)

                orig_nodes = nx.number_of_nodes(original_)
                print 'orig: number of unique nodes:',orig_nodes 
                num_nodes_snapshots.append(orig_nodes)          
                orig_edges = nx.number_of_edges(original_)
                print 'orig: number of unique edges:',orig_edges
                
               # partition original             
               # G = nx.read_edgelist(path=readfile, delimiter=",", nodetype=int,  create_using=nx.Graph())
               # start = 0
               # original_ = nx.convert_node_labels_to_integers(original, first_label=start)
                original=maximum_connected_components(original_)

                # orig_eccentricity = nx.diameter(original)
                # orig_radius = nx.radius(original)
                orig_eccentricity = 0
                orig_radius = 0
                print "orig: Eccentricity - "+str(orig_eccentricity)
                print "orig: Radius - "+str(orig_radius)
                orig_density = nx.density(original)
                print('orig: graph_density: %f' % (orig_density))
                orig_clustering_coeff = nx.average_clustering(original)
                print('orig: graph_clustering_coefficient: %f' % (orig_clustering_coeff))
      
               #rewriteEdgelistFromZero(args.out_directory_stem + 'network_v' + str(i) + '.dat', '\t')
               #Rewrite clustering file such that node ids start from zero to maintain consistency with edgelist file node ids
               #rewriteClusteringFromZero(args.out_directory_stem + 'community_v' + str(i) + '.dat', '\t')

                if ground_truth==0:
                    if 0:
                        partition_ground_truth = community.best_partition(nx.Graph(original))
    #                    snapshotCommunities(args.input_directory_stem + 'output-prefix.t0000'+ str(i), partition_ground_truth)  
                        writePartitionToFile(partition_ground_truth, args.input_directory_stem + 'output-prefix.t0000'+ str(i) +'.comms')   
                    if 1:
                        partition_ground_truth = community.best_partition(nx.Graph(original))
                        #expand partition
                        expand_partition_with_other_components(partition_ground_truth, original_, original)
                        #write the partition
                        #snapshotCommunities(full_partition_prefix, part)# this is almost the same with writePartitionToFile(), the difference prefix and fullname.
                        writePartitionToFile(partition_ground_truth, args.input_directory_stem + 'output-prefix.t0000'+ str(i) +'.comms')            
                
                #tranverse the form of clustering
                #parseLancichinettiResults(args.input_directory_stem + prefix+'.t0'+ str(i) + '.comm', args.input_directory_stem + 'output-prefix.t0000'+ str(i) + '.comm')
                partition_set = se.ReadInData(args.input_directory_stem  + 'output-prefix.t0000'+ str(i) +'.comms')
                partition_ground_truth = se.partitionFromFile(args.input_directory_stem + 'output-prefix.t0000'+ str(i) +'.comms', ' ')


                number_of_gt_partition = len(partition_set) 
                print "number_of_gt_partition:" + str(number_of_gt_partition)            
                orig_modularity = community.modularity(partition_ground_truth, original)
                print "orig_modularity:" + str(orig_modularity)               
    #draw graph & degree distribution            
    #            count = 0
    #            #pos = nx.spring_layout(original)
    #            pos = nx.spectral_layout(original)
    #            colors = ['#660066' ,'#eeb111' ,'#4bec13' ,'#d1d1d1' ,'#a3a3a3' ,'#c39797' ,'#a35f0c' ,'#5f0ca3' ,'#140ca3' ,'#a30c50' ,'#a30c50' ,'#0ca35f' ,'#bad8eb' ,'#ffe5a9' ,'#f5821f' ,'#00c060' ,'#00c0c0' ,'#b0e0e6' ,'#999999' ,'#ffb6c1' ,'#6897bb']
    #            for com in set(partition_ground_truth.values()) :
    #                count = count + 1
    #                list_nodes = [nodes for nodes in partition_ground_truth.keys() if partition_ground_truth[nodes] == com]
    #                nx.draw_networkx_nodes(original, pos, list_nodes, node_color = colors[count-1])            
    #            nx.draw_networkx_edges(original,pos, alpha=0.5)
    #            nx.draw(original, pos, alpha=0.5)
    #            plt.show()
                          
    #original degree distribution
    #            ba_g = nx.degree_centrality(original)
    #            ba_g2 = dict(Counter(ba_g.values()))
    #            ba_gx,ba_gy = log_binning(ba_g2,50)

    #             plt.figure(1)
    #             plt.xscale('log')
    #             plt.yscale('log')
    # #            plt.scatter(ba_gx,ba_gy,c='r',marker='s',s=50)
    #             plt.scatter(ba_g2.keys(),ba_g2.values(),c='b',marker='x')
    #             plt.xlabel('Connections (normalized)')
    #             plt.ylabel('Frequency')
    # #            plt.xlim((1e-4,1e-1))
    # #            plt.ylim((.9,1e4))
    #             plt.show()

      
    #            float('%.2f'%per)  round(per,2)
    #            print round(per,2)
                per_format = ("%.6f" % float(per))
#                full_partition_prefix = output_directory_stem + sampling_algorithm+'_'+ data_set+'_result_'+str(rep)+'/'+str(per_format)+'/snapshot_t'+ str(i)            
                full_partition_prefix = output_directory_stem + 'network_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep)
                
                re=open(full_partition_prefix +'.dat', 'rb')
                sample_=nx.read_edgelist(re, nodetype=int)
                re.close() 
                
                sample_edges = nx.number_of_edges(sample_)
                sample_nodes = nx.number_of_nodes(sample_)
                print 'S: number of unique nodes:',sample_nodes
                print 'S: number of unique edges:',sample_edges

                sample_components = nx.number_connected_components(sample_)
                print "S: Num components: "+str(sample_components)

                sample = maximum_connected_components(sample_)
                eccentricity = 0
                radius = 0
                # eccentricity = nx.diameter(sample)
                # radius = nx.radius(sample)
                print "S: Eccentricity - "+str(eccentricity)
                print "S: Radius - "+str(radius)

                # add new stat
                density = nx.density(sample)
                print('S: graph_density: %f' % (density))
                clustering_coeff = nx.average_clustering(sample)
                print('S: graph_clustering_coefficient: %f' % (clustering_coeff))

                time_elipse = 0.0
                #partition sample            
                if clustering_algorithm == 'vbmod':            
                #run vbmod
                    nx.write_edgelist(sample, full_partition_prefix +"_test.graph",delimiter='\t',data=False)    
                    Sample_test=nx.read_edgelist(full_partition_prefix +"_test.graph")
                    
                    # convert networkx graph object to sparse matrix
                    node_list=Sample_test.nodes()
                    A=nx.to_scipy_sparse_matrix(Sample_test, nodelist=node_list)
                    print(A.todense())
                    
                    N=A.shape[0]        # number of nodes
                    Kvec=range(number_of_gt_partition,number_of_gt_partition+1) # range of K values over which to search
                    
                    # hyperparameters for priors
                    net0={}
                    net0['ap0']=N*1.
                    net0['bp0']=1.
                    net0['am0']=1.
                    net0['bm0']=N*1.
                    
                    # options
                    opts={}
                    opts['NUM_RESTARTS']=1
                    
                    # run vb
                    (net,net_K)=vd.learn_restart(A.tocsr(),Kvec,net0,opts)
                    
                    q = net['Q']
                    filename =full_partition_prefix +'.comms';
                    fp = open(filename, "w+");
                    u = 0
                    r = 0;
                    max_probability = 0.0;
                    max_index = 0;
                    K=number_of_gt_partition
                    
                    for u in range(len(q)):
                        fp.write(str(node_list[u]))
                    #initialization
                        max_probability = 0.0;
                        max_index = 0;
                        for r in range(K): 
                            if q[u,r] > max_probability:
                                max_probability = q[u,r]
                                max_index = r 
                    #printf(" %.6f",q[u][r])
                        fp.write(" "+ str(max_index))
                        fp.write("\n")
                    fp.close()

                    part = se.partitionFromFile(full_partition_prefix +'.comms', ' ') 

                    #without considering the isolated nodes
                    # writePartitionToFile(part, full_partition_prefix +'.comms')            
                    # commsfile = full_partition_prefix +'.comms' 

                    #considering the isolated nodes
                    expand_partition_with_other_components(part, sample_, sample)
                    writePartitionToFile(part, full_partition_prefix +'_plus_isolated.comms')            
                    commsfile = full_partition_prefix + '_plus_isolated.comms'   

                elif clustering_algorithm == 'blondel':
                #run blondel
                    part = community.best_partition(nx.Graph(sample))
                    #snapshotCommunities(full_partition_prefix, part)
                    writePartitionToFile(part, full_partition_prefix +'.comms')            
                    commsfile = full_partition_prefix +'.comms'

                elif clustering_algorithm == 'blondel_isolated_nodes':
                #run blondel
                    part = community.best_partition(nx.Graph(sample))
                    #expand partition
                    expand_partition_with_other_components(part, sample_, sample)
                    #write the partition
                    #snapshotCommunities(full_partition_prefix, part)# this is almost the same with writePartitionToFile(), the difference prefix and fullname.
                    writePartitionToFile(part, full_partition_prefix +'_plus_isolated.comms')            
                    commsfile = full_partition_prefix + '_plus_isolated.comms'
       
                elif clustering_algorithm == 'bigclam':           
                #run bigclam
                #case 1
                    runBigclam(full_partition_prefix +'.graph', number_of_gt_partition, full_partition_prefix)
                    parseBigclamResults(full_partition_prefix, ' ')   
                    part = se.partitionFromFile(full_partition_prefix +'.comms', ' ')
                    commsfile = full_partition_prefix +'.comms'

                elif clustering_algorithm == 'bigclam_isolated_nodes':
                #case 2
                    nx.write_edgelist(sample, full_partition_prefix +"_test.graph",delimiter='\t',data=False)
                    runBigclam(full_partition_prefix +"_test.graph", number_of_gt_partition, full_partition_prefix)
                    parseBigclamResults(full_partition_prefix, ' ')
                    part = se.partitionFromFile(full_partition_prefix +'.comms', ' ')

                    expand_partition_with_other_components(part, sample_, sample)
                    writePartitionToFile(part, full_partition_prefix +'_plus_isolated.comms')            
                    commsfile = full_partition_prefix + '_plus_isolated.comms'
                    
                elif clustering_algorithm == 'LP_complete':
                #run blondel + lp  complete ,note that the index should start at 0 and consistent
                    time_clustering_start = time.time()              
                    part_ = community.best_partition(nx.Graph(sample))
                    #expand partition
                    expand_partition_with_other_components(part_, sample_, sample)                    
                    #generate comm_list for label propagation algorithm
                   
                    total_nodes = orig_nodes
                    print('total_nodes:', total_nodes)
                    print('part_:', len(part_))
                    comm_list = buildCommunitieslist(part_, total_nodes)
                    
                    #complete comm_list    
                    fixed_list = buildFixedlist(comm_list)
                    
                    #pre-process here
                    kCorePreProcess(comm_list, original_, sample_)
                    #################
                    
                    #generate complete comm_lsit
                    complete_part = igr.community_label_propagation(initial = comm_list, fixed = fixed_list).membership
                    time_clustering_end = time.time()   
                    time_elipse= format(time_clustering_end - time_clustering_start, '0.2f')
                    print '\nTime to clustering bench_directory:'+ str(time_clustering_end - time_clustering_start) + '\n'
                      
                    #write the partition
                    writeFullPartitionToFile(complete_part,total_nodes ,full_partition_prefix +'_entire.comms') 
                    part = se.partitionFromFile(full_partition_prefix +'_entire.comms', ' ')
                    commsfile = full_partition_prefix + '_entire.comms'


                elif clustering_algorithm == 'TCI':
                #run blondel + TCI complete ,note that the index should start at 0 and consistent
                    time_clustering_start = time.time()  

                    #####sampling clustering file
                    part = community.best_partition(nx.Graph(sample))
                    #expand partition
                    expand_partition_with_other_components(part, sample_, sample)
                    #write the partition
                    writePartitionToFile_t(part, full_partition_prefix +'_tci.comms')            
                    commsfile = full_partition_prefix + '_tci.comms'
                    community_file2= commsfile
                    community_output2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.csv'
                    transform_community_no_duplicate(community_file2, community_output2)
                    # transform_community(community_file, community_output)

                    #####original clustering file
                    community_file= output_directory_stem + 'community'+'_v'+ str(rep)+'.dat'
                    community_output= output_directory_stem + 'community'+'_v'+ str(rep)+'.csv'
                    transform_community_no_duplicate(community_file, community_output)
                    # transform_community(community_file, community_output)

                    #####transform edge file
                    edges_file= output_directory_stem + 'network'+'_v'+ str(rep)+'.dat'
                    edges_output= output_directory_stem + 'network'+'_v'+ str(rep)+'.rn'
                    transform_edges(edges_file, edges_output)

                    #####use the larger cluster number in original and sample clustering 
                    cluster_set_original = se.ReadInData(community_file)
                    cluster_set_sample = se.ReadInData(community_file2)
                    if (len(cluster_set_original)> len(cluster_set_sample)):
                        number_of_gt_partition = len(cluster_set_original)
                    else:
                        number_of_gt_partition = len(cluster_set_sample)
                    nc= number_of_gt_partition

                    cluster_flag=0  #the clustering ID is ranked from 1, otherwises it is from 0
                    create_arff(output_directory_stem, 'network'+'_v' + str(rep) + '.arff', 'community'+'_v'+ str(rep)+'.csv', 'network'+'_v'+ str(rep)+'.rn', nc, cluster_flag)
                    clusterByPopulation(output_directory_stem,'record.predict', 'Person:Behaviour', os.path.abspath(community_output2), 'relaxlabel', os.path.abspath(output_directory_stem +'network_v1.arff'))
                    
                    if(is_non_zero_file(output_directory_stem + 'record.predict')):
                        create_PopClusteFile(output_directory_stem, 'record.predict', full_partition_prefix +'_tci_entire.comms', community_output2)
                        part = se.partitionFromFile(full_partition_prefix +'_tci_entire.comms', ' ')
                        commsfile = full_partition_prefix + '_tci_entire.comms'                   
                        time_clustering_end = time.time()   
                        time_elipse= format(time_clustering_end - time_clustering_start, '0.2f')
                        print '\nTime to clustering bench_directory:'+ str(time_clustering_end - time_clustering_start) + '\n'
                    else:
                        print '\nNo record.predict is generated.'    
                        raise Exception('No record.predict is generated.') # Don't! If you catch, likely to hide bugs   


                elif clustering_algorithm == 'iterative':
                #run blondel + iterative complete ,note that the index should start at 0 and consistent
                    time_clustering_start = time.time()  

                    #####sampling clustering file
                    part = community.best_partition(nx.Graph(sample))
                    #expand partition
                    expand_partition_with_other_components(part, sample_, sample)
                    #write the partition
                    writePartitionToFile_t(part, full_partition_prefix +'_iterative.comms')            
                    commsfile = full_partition_prefix + '_iterative.comms'
                    community_file2= commsfile
                    community_output2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.csv'
                    transform_community_no_duplicate(community_file2, community_output2)
                    # transform_community(community_file, community_output)

                    #####original clustering file
                    community_file= output_directory_stem + 'community'+'_v'+ str(rep)+'.dat'
                    community_output= output_directory_stem + 'community'+'_v'+ str(rep)+'.csv'
                    transform_community_no_duplicate(community_file, community_output)
                    # transform_community(community_file, community_output)

                    #####transform edge file
                    edges_file= output_directory_stem + 'network'+'_v'+ str(rep)+'.dat'
                    edges_output= output_directory_stem + 'network'+'_v'+ str(rep)+'.rn'
                    transform_edges(edges_file, edges_output)

                    #####use the larger cluster number in original and sample clustering 
                    cluster_set_original = se.ReadInData(community_file)
                    cluster_set_sample = se.ReadInData(community_file2)
                    if (len(cluster_set_original)> len(cluster_set_sample)):
                        number_of_gt_partition = len(cluster_set_original)
                    else:
                        number_of_gt_partition = len(cluster_set_sample)
                    nc= number_of_gt_partition

                    cluster_flag=0  #the clustering ID is ranked from 1, otherwises it is from 0
                    create_arff(output_directory_stem, 'network'+'_v' + str(rep) + '.arff', 'community'+'_v'+ str(rep)+'.csv', 'network'+'_v'+ str(rep)+'.rn', nc, cluster_flag)
                    clusterByPopulation(output_directory_stem,'record.predict', 'Person:Behaviour', os.path.abspath(community_output2), 'iterative', os.path.abspath(output_directory_stem +'network_v1.arff'))
                    
                    if(is_non_zero_file(output_directory_stem + 'record.predict')):
                        create_PopClusteFile(output_directory_stem, 'record.predict', full_partition_prefix +'_iterative_entire.comms', community_output2)
                        part = se.partitionFromFile(full_partition_prefix +'_iterative_entire.comms', ' ')
                        commsfile = full_partition_prefix + '_iterative_entire.comms'                   
                        time_clustering_end = time.time()   
                        time_elipse= format(time_clustering_end - time_clustering_start, '0.2f')
                        print '\nTime to clustering bench_directory:'+ str(time_clustering_end - time_clustering_start) + '\n'
                    else:
                        print '\nNo record.predict is generated.'    
                        raise Exception('No record.predict is generated.') # Don't! If you catch, likely to hide bugs                   

                elif clustering_algorithm == 'gibbs':
                #run blondel + gibbs complete ,note that the index should start at 0 and consistent
                    time_clustering_start = time.time()  

                    #####sampling clustering file
                    part = community.best_partition(nx.Graph(sample))
                    #expand partition
                    expand_partition_with_other_components(part, sample_, sample)
                    #write the partition
                    writePartitionToFile_t(part, full_partition_prefix +'_gibbs.comms')            
                    commsfile = full_partition_prefix + '_gibbs.comms'
                    community_file2= commsfile
                    community_output2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.csv'
                    transform_community_no_duplicate(community_file2, community_output2)
                    # transform_community(community_file, community_output)

                    #####original clustering file
                    community_file= output_directory_stem + 'community'+'_v'+ str(rep)+'.dat'
                    community_output= output_directory_stem + 'community'+'_v'+ str(rep)+'.csv'
                    transform_community_no_duplicate(community_file, community_output)
                    # transform_community(community_file, community_output)

                    #####transform edge file
                    edges_file= output_directory_stem + 'network'+'_v'+ str(rep)+'.dat'
                    edges_output= output_directory_stem + 'network'+'_v'+ str(rep)+'.rn'
                    transform_edges(edges_file, edges_output)

                    #####use the larger cluster number in original and sample clustering 
                    cluster_set_original = se.ReadInData(community_file)
                    cluster_set_sample = se.ReadInData(community_file2)
                    if (len(cluster_set_original)> len(cluster_set_sample)):
                        number_of_gt_partition = len(cluster_set_original)
                    else:
                        number_of_gt_partition = len(cluster_set_sample)
                    nc= number_of_gt_partition

                    cluster_flag=0  #the clustering ID is ranked from 1, otherwises it is from 0
                    create_arff(output_directory_stem, 'network'+'_v' + str(rep) + '.arff', 'community'+'_v'+ str(rep)+'.csv', 'network'+'_v'+ str(rep)+'.rn', nc, cluster_flag)
                    clusterByPopulation(output_directory_stem,'record.predict', 'Person:Behaviour', os.path.abspath(community_output2), 'gibbs', os.path.abspath(output_directory_stem +'network_v1.arff'))
                    
                    if(is_non_zero_file(output_directory_stem + 'record.predict')):
                        create_PopClusteFile(output_directory_stem, 'record.predict', full_partition_prefix +'_gibbs_entire.comms', community_output2)
                        part = se.partitionFromFile(full_partition_prefix +'_gibbs_entire.comms', ' ')
                        commsfile = full_partition_prefix + '_gibbs_entire.comms'                   
                        time_clustering_end = time.time()   
                        time_elipse= format(time_clustering_end - time_clustering_start, '0.2f')
                        print '\nTime to clustering bench_directory:'+ str(time_clustering_end - time_clustering_start) + '\n'
                    else:
                        print '\nNo record.predict is generated.'    
                        raise Exception('No record.predict is generated.') # Don't! If you catch, likely to hide bugs   



                elif clustering_algorithm == 'TCI_ground_truth':
                #run blondel + lp  complete ,note that the index should start at 0 and consistent
                    time_clustering_start = time.time()    
                    # full_partition_prefix = output_directory_stem + 'network_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep)
                    community_file= output_directory_stem + 'community'+'_v'+ str(rep)+'.dat'
                    community_output= output_directory_stem + 'community'+'_v'+ str(rep)+'.csv'
                    transform_community_no_duplicate(community_file, community_output)
                    # transform_community(community_file, community_output)
                    community_file2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.dat'
                    community_output2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.csv'
                    transform_community_no_duplicate(community_file2, community_output2)
                    # transform_community(community_file, community_output)

                    edges_file= output_directory_stem + 'network'+'_v'+ str(rep)+'.dat'
                    edges_output= output_directory_stem + 'network'+'_v'+ str(rep)+'.rn'
                    transform_edges(edges_file, edges_output)
                    
                    cluster_set = se.ReadInData(community_file)
                    number_of_gt_partition = len(cluster_set)
                    nc= number_of_gt_partition
                    cluster_flag=0  #the clustering ID is ranked from 1, otherwises it is from 0
                    create_arff(output_directory_stem, 'network'+'_v' + str(rep) + '.arff', 'community'+'_v'+ str(rep)+'.csv', 'network'+'_v'+ str(rep)+'.rn', nc, cluster_flag)
                    clusterByPopulation(output_directory_stem,'record.predict', 'Person:Behaviour', os.path.abspath(community_output2), 'relaxlabel', os.path.abspath(output_directory_stem +'network_v1.arff'))
                    
                    if(is_non_zero_file(output_directory_stem + 'record.predict')):
                        create_PopClusteFile(output_directory_stem, 'record.predict', full_partition_prefix +'_tci.comms', community_output2)
                        part = se.partitionFromFile(full_partition_prefix +'_tci.comms', ' ')
                        commsfile = full_partition_prefix + '_tci.comms'                   
                        time_clustering_end = time.time()   
                        time_elipse= format(time_clustering_end - time_clustering_start, '0.2f')
                        print '\nTime to clustering bench_directory:'+ str(time_clustering_end - time_clustering_start) + '\n'
                    else:
                        print '\nNo record.predict is generated.'    
                        raise Exception('No record.predict is generated.') # Don't! If you catch, likely to hide bugs                   

                elif clustering_algorithm == 'iterative_ground_truth':
                #run blondel + lp  complete ,note that the index should start at 0 and consistent
                    time_clustering_start = time.time()    
                    # full_partition_prefix = output_directory_stem + 'network_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep)
                    community_file= output_directory_stem + 'community'+'_v'+ str(rep)+'.dat'
                    community_output= output_directory_stem + 'community'+'_v'+ str(rep)+'.csv'
                    transform_community_no_duplicate(community_file, community_output)
                    # transform_community(community_file, community_output)
                    community_file2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.dat'
                    community_output2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.csv'
                    transform_community_no_duplicate(community_file2, community_output2)
                    # transform_community(community_file, community_output)

                    edges_file= output_directory_stem + 'network'+'_v'+ str(rep)+'.dat'
                    edges_output= output_directory_stem + 'network'+'_v'+ str(rep)+'.rn'
                    transform_edges(edges_file, edges_output)
                    
                    cluster_set = se.ReadInData(community_file)
                    number_of_gt_partition = len(cluster_set)
                    nc= number_of_gt_partition
                    cluster_flag=0  #the clustering ID is ranked from 1, otherwises it is from 0
                    create_arff(output_directory_stem, 'network'+'_v' + str(rep) + '.arff', 'community'+'_v'+ str(rep)+'.csv', 'network'+'_v'+ str(rep)+'.rn', nc, cluster_flag)
                    
                    clusterByPopulation(output_directory_stem,'record.predict', 'Person:Behaviour', os.path.abspath(community_output2), 'iterative', os.path.abspath(output_directory_stem +'network_v1.arff'))
                    
                    if(is_non_zero_file(output_directory_stem + 'record.predict')):
                        create_PopClusteFile(output_directory_stem, 'record.predict', full_partition_prefix +'_iterative.comms', community_output2)
                        part = se.partitionFromFile(full_partition_prefix +'_iterative.comms', ' ')
                        commsfile = full_partition_prefix + '_iterative.comms'                   
                        time_clustering_end = time.time()   
                        time_elipse= format(time_clustering_end - time_clustering_start, '0.2f')
                        print '\nTime to clustering bench_directory:'+ str(time_clustering_end - time_clustering_start) + '\n'
                    else:
                        print '\nNo record.predict is generated.'    
                        raise Exception('No record.predict is generated.') # Don't! If you catch, likely to hide bugs 

                elif clustering_algorithm == 'gibbs_ground_truth':
                #run blondel + lp  complete ,note that the index should start at 0 and consistent
                    time_clustering_start = time.time()    
                    # full_partition_prefix = output_directory_stem + 'network_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep)
                    community_file= output_directory_stem + 'community'+'_v'+ str(rep)+'.dat'
                    community_output= output_directory_stem + 'community'+'_v'+ str(rep)+'.csv'
                    transform_community_no_duplicate(community_file, community_output)
                    # transform_community(community_file, community_output)
                    community_file2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.dat'
                    community_output2= output_directory_stem + 'community_sample_p'+ str(int(100*float(per))) +'_v'+ str(rep) +'.csv'
                    transform_community_no_duplicate(community_file2, community_output2)
                    # transform_community(community_file, community_output)

                    edges_file= output_directory_stem + 'network'+'_v'+ str(rep)+'.dat'
                    edges_output= output_directory_stem + 'network'+'_v'+ str(rep)+'.rn'
                    transform_edges(edges_file, edges_output)
                    
                    cluster_set = se.ReadInData(community_file)
                    number_of_gt_partition = len(cluster_set)
                    nc= number_of_gt_partition

                    cluster_flag=0  #the clustering ID is ranked from 1, otherwises it is from 0
                    create_arff(output_directory_stem, 'network'+'_v' + str(rep) + '.arff', 'community'+'_v'+ str(rep)+'.csv', 'network'+'_v'+ str(rep)+'.rn', nc, cluster_flag)

                    clusterByPopulation(output_directory_stem,'record.predict', 'Person:Behaviour', os.path.abspath(community_output2), 'gibbs', os.path.abspath(output_directory_stem +'network_v1.arff'))
                    
                    if(is_non_zero_file(output_directory_stem + 'record.predict')):
                        create_PopClusteFile(output_directory_stem, 'record.predict', full_partition_prefix +'_gibbs.comms', community_output2)
                        part = se.partitionFromFile(full_partition_prefix +'_gibbs.comms', ' ')
                        commsfile = full_partition_prefix + '_gibbs.comms'                   
                        time_clustering_end = time.time()   
                        time_elipse= format(time_clustering_end - time_clustering_start, '0.2f')
                        print '\nTime to clustering bench_directory:'+ str(time_clustering_end - time_clustering_start) + '\n'
                    else:
                        print '\nNo record.predict is generated.'    
                        raise Exception('No record.predict is generated.') #No record.predict is generated.   
                else:
                    print('please double check the algorithm name') 
                    raise Exception('No this algorithm.')    

                      

                partition_sample = se.ReadInData(commsfile)
                first_result_lines = se.Compare3(copy.deepcopy(partition_set), partition_sample, deltas, float(per), commsfile)
                
                if clustering_algorithm == 'LP_complete':
                    normalized_rate=1.0
                else:
                    normalized_rate =  orig_nodes* float(per)/sample_nodes #in case of the initial snapshots have less nodes than expected numbers.            
                if normalized_rate > 1.0:
                    normalized_rate = 1.0

                print("normalized_rate:%f" %normalized_rate)
                precision_micro=first_result_lines[0][2]  
                recall_micro=first_result_lines[1][2]*normalized_rate
                p9_precision_micro = first_result_lines[2][2]
                r9_recall_micro=first_result_lines[3][2]*normalized_rate
                p8_precision_micro=first_result_lines[4][2]
                r8_recall_micro=first_result_lines[5][2]*normalized_rate
                p7_precision_micro=first_result_lines[6][2]
                r7_recall_micro=first_result_lines[7][2]*normalized_rate
                p6_precision_micro=first_result_lines[8][2]
                r6_recall_micro=first_result_lines[9][2]*normalized_rate
                p5_precision_micro=first_result_lines[10][2]
                r5_recall_micro=first_result_lines[11][2]*normalized_rate
                p4_precision_micro = first_result_lines[12][2]
                r4_recall_micro=first_result_lines[13][2]*normalized_rate
                p3_precision_micro=first_result_lines[14][2]
                r3_recall_micro=first_result_lines[15][2]*normalized_rate
                p2_precision_micro=first_result_lines[16][2]
                r2_recall_micro=first_result_lines[17][2]*normalized_rate
                p1_precision_micro=first_result_lines[18][2]
                r1_recall_micro=first_result_lines[19][2]*normalized_rate
                p0_precision_micro=first_result_lines[20][2]
                r0_recall_micro=first_result_lines[21][2]*normalized_rate
                           
                mediate_size_micro=first_result_lines[22][2]                   
                sample_cluster_num_micro=first_result_lines[23][2]
                original_cluster_num_micro=first_result_lines[24][2]
                ANC=first_result_lines[25][2]
                NLS=first_result_lines[26][2] 
                
                print "precision_micro - "+str(precision_micro)
                print "recall_micro - "+str(recall_micro)
                print "p8_precision_micro - "+str(p8_precision_micro)
                print "r8_recall_micro - "+str(r8_recall_micro)
                print "p5_precision_micro - "+str(p5_precision_micro)
                print "r5_recall_micro - "+str(r5_recall_micro)
                print "mediate_size_micro - "+str(mediate_size_micro)
                print "sample_cluster_num_micro - "+str(sample_cluster_num_micro)
                print "ANC - "+str(ANC)
                print "NLS - "+str(NLS)

                result_lines_reference = se.runComparisonGraphMetrics(part, partition_ground_truth, False)                
                NMI = result_lines_reference[0][2]
                ARS = result_lines_reference[1][2]
                VNMI = result_lines_reference[2][2]  
                print "VNMI: - "+str(VNMI)           
                                            
                #calculate the modularity        
                new_nodes=[]
                for node_index in part:
                    new_nodes.append(node_index)
                H = sample.subgraph(new_nodes)
                try:
                    modularity = community.modularity(part, H)
                except ValueError, e:
                    modularity = 0
                print "S: Modularity - "+str(modularity)

    #another way to calculate the modularity, not complete
                # igr = igraph.Graph(n = H.number_of_nodes(), edges = nx.convert_node_labels_to_integers(H).edges())     
                # complete_part = igr.community_label_propagation(initial = comm_list, fixed = fixed_list).membership
                # modularity= igr.modularity(complete_part)
                     
    #==============================================================================
    #             plt.figure(2)
    #             ba_c = nx.degree_centrality(sample)
    #             ba_c2 = dict(Counter(ba_c.values()))
    #             ba_x,ba_y = log_binning(ba_c2,50)
    #     
    #             plt.xscale('log')
    #             plt.yscale('log')
    #             plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)
    #             plt.scatter(ba_c2.keys(),ba_c2.values(),c='b',marker='x')
    #             plt.xlim((1e-4,1e-1))
    #             plt.ylim((.9,1e4))
    #             plt.xlabel('Connections (normalized)')
    #             plt.ylabel('Frequency')
    #             plt.show()
    #==============================================================================            
                print time.time()-t


                ###################degree distribution and distance
                original_degree=smp.SimpleGraphDegree()
                original_degree_dis =original_degree.compute_frontend_distribution(original_)
                #print('original_degree_dis',original_degree_dis)
                print('\n')
                sample_degree=smp.SimpleGraphDegree()
                sample_degree_dis =sample_degree.compute_frontend_distribution(sample_)
                print('sample_degree_dis',sample_degree_dis)
                print('\n')            
                js_result_pies =dm.JensenShannonDivergence.compute(original_degree_dis, sample_degree_dis)
                ks_result_pies =dm.KolmogorovSmirnovDistance.compute(original_degree_dis, sample_degree_dis)
                print js_result_pies
                print ks_result_pies   

                ###################clusteringcoefficient distribution and distance    
                original_clusteringcoefficient=smp.SimpleGraphClusteringCoefficient()
                original_clusteringcoefficient_dis = original_clusteringcoefficient.compute_frontend_distribution(original_)
    #            print('original_clusteringcoefficient_dis',original_clusteringcoefficient_dis)
                print('\n')            
                sample_clusteringcoefficient=smp.SimpleGraphClusteringCoefficient()
                sample_clusteringcoefficient_dis = sample_clusteringcoefficient.compute_frontend_distribution(sample_)
    #            print('sample_clusteringcoefficient_dis',sample_clusteringcoefficient_dis)
                print('\n')
                cc_js_result_pies =dm.JensenShannonDivergence.compute(original_clusteringcoefficient_dis, sample_clusteringcoefficient_dis)
                cc_ks_result_pies =dm.KolmogorovSmirnovDistance.compute(original_clusteringcoefficient_dis, sample_clusteringcoefficient_dis)
                print cc_js_result_pies
                print cc_ks_result_pies

                ###################shortest path distribution and distance
                original_pathlength=smp.SimpleGraphPathLength()
                original_pathlength_dis = original_pathlength.compute_frontend_distribution(original)
    #            print('original_pathlength_dis',original_pathlength_dis)
                print('\n')
                sample_pathlength=smp.SimpleGraphPathLength()
                sample_pathlength_dis = sample_pathlength.compute_frontend_distribution(sample)
    #            print('sample_pathlength_dis',sample_clusteringcoefficient_dis)
                print('\n')   
                path_js_result_pies =dm.JensenShannonDivergence.compute(original_pathlength_dis, sample_pathlength_dis)
                path_ks_result_pies =dm.KolmogorovSmirnovDistance.compute(original_pathlength_dis, sample_pathlength_dis)
                print path_js_result_pies
                print path_ks_result_pies

                ###################kcores distribution and distance
                minimal_of_k=1
    #            kcores_path = os.getcwd() + os.sep + 'tem_kcores_dir'+os.sep 
                orig_prefix ='orig_core'+str(i)
                runKcore(args.input_directory_stem + 'network_v1.dat', minimal_of_k, orig_prefix)
                original_kcores_dis = compute_kcore_distribution('kcore-'+orig_prefix+'.tab', '\t', normalize=True)
                print('original_kcores_dis',original_kcores_dis)
                print('\n') 

                sample_prefix = 'sample_core'+str(i)
                runKcore(full_partition_prefix +'.dat', minimal_of_k, sample_prefix)
                sample_kcores_dis = compute_kcore_distribution('kcore-'+sample_prefix+'.tab', '\t', normalize=True)
                print('sample_kcores_dis',sample_kcores_dis)

                kcores_js_result_pies =dm.JensenShannonDivergence.compute(original_kcores_dis, sample_kcores_dis)
                kcores_ks_result_pies =dm.KolmogorovSmirnovDistance.compute(original_kcores_dis, sample_kcores_dis)
                print kcores_js_result_pies
                print kcores_ks_result_pies 

    ############summraize  the distribution
                #save_pdf_cdf_plot_for_a_single_graph(res)   
                Result_folder = 'finial_results'+ os.sep + data_set + os.sep + clustering_algorithm + os.sep           
                if per == percentages[0] and rep==1 and i== snapshot-1:  #only draw the distribution of the first trial and first percentage and the last snapshot. nice.
                    # orig_eccentricity = nx.diameter(original)
                    # orig_radius = nx.radius(original)
                    # eccentricity = nx.diameter(sample)
                    # radius = nx.radius(sample)


                    # if not os.path.exists(Result_folder):
                    #     os.makedirs(Result_folder)
                    # # It means when the ./order_ start again and the first percenage, we should clean the folder. Only suitable for our case jianpeng
                    if per == percentages[0] and rep==1 and sampling_algorithm =='top_k_leader_internal_priority_extra' :
                        deletePathIfNeeded(Result_folder)    
                        createPathIfNeeded(Result_folder)


                    #static of original graph
                    distribution_save(original_degree_dis, original_degree_dis, Result_folder + data_set +'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '_degree_distribution_original.dat')
                    distribution_save(original_clusteringcoefficient_dis, original_clusteringcoefficient_dis, Result_folder + data_set +'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '_clusercoefficient_distribution_original.dat')
                    distribution_save(original_pathlength_dis, original_pathlength_dis, Result_folder + data_set +'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '_shortpath_distribution_original.dat')
                    distribution_save(original_kcores_dis, original_kcores_dis, Result_folder + data_set +'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '_kcores_distribution_original.dat')

      
                    #static of sample graph
                    distribution_save(original_degree_dis, sample_degree_dis, Result_folder + data_set+'_'+clustering_algorithm+'_'+sampling_algorithm+'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '_degree_distribution.dat')
                    distribution_save(original_clusteringcoefficient_dis, sample_clusteringcoefficient_dis, Result_folder + data_set+'_'+clustering_algorithm+'_'+sampling_algorithm+'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '_clusercoefficient_distribution.dat')
                    distribution_save(original_pathlength_dis, sample_pathlength_dis, Result_folder + data_set+'_'+clustering_algorithm+'_'+sampling_algorithm+'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '_shortpath_distribution.dat')                                            
                    distribution_save(original_kcores_dis, sample_kcores_dis, Result_folder + data_set+'_'+clustering_algorithm+'_'+sampling_algorithm+'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '_kcores_distribution.dat')                                            
                                
                results.append(str(i)+ ','+str(js_result_pies)+ ','+str(ks_result_pies)+ ','+str(cc_js_result_pies)+ ','+str(cc_ks_result_pies)+ ','
                        +str(path_js_result_pies)+ ','+str(path_ks_result_pies)+ ','  
                        +str(kcores_js_result_pies)+ ','+str(kcores_ks_result_pies)+ ','                                
                        +str(precision_micro)+ ','+ str(recall_micro)+ ','+ str(p9_precision_micro)+ ','+str(r9_recall_micro)+ ','
                        +str(p8_precision_micro)+ ','+str(r8_recall_micro)+ ','+str(p7_precision_micro)+ ','+str(r7_recall_micro)+ ','
                        +str(p6_precision_micro)+ ','+str(r6_recall_micro)+ ','+str(p5_precision_micro)+ ','+str(r5_recall_micro)+ ','
                        +str(p4_precision_micro)+ ','+str(r4_recall_micro)+ ','+str(p3_precision_micro)+ ','+str(r3_recall_micro)+ ','
                        +str(p2_precision_micro)+ ','+str(r2_recall_micro)+ ','+str(p1_precision_micro)+ ','+str(r1_recall_micro)+ ','
                        +str(p0_precision_micro)+ ','+str(r0_recall_micro)+ ','
                        +str(ANC)+ ','+str(NLS)+ ','
                        +str(NMI)+ ','+str(ARS)+ ','+ str(VNMI)+','
                        +str(sample_edges) + ','
                        +str(sample_nodes)+','+ str(sample_components)+','
                        +str(eccentricity)+ ','+str(radius)+ ','
                        +str(modularity)+ ','
                        +str(density)+','+ str(clustering_coeff) +','
                        +str(sample_cluster_num_micro)+ ','
                        +str(mediate_size_micro)+ ','
                        +str(orig_edges) + ','
                        +str(orig_nodes)+','+ str(orig_components)+','
                        +str(orig_eccentricity)+ ','+str(orig_radius)+ ','
                        +str(orig_modularity)+ ','
                        +str(orig_density)+','+ str(orig_clustering_coeff) +','
                        +str(original_cluster_num_micro)+','+ str(time_elipse)
                        +'\n') 
     
            we=open(Result_folder + data_set+'_'+clustering_algorithm+'_'+sampling_algorithm+'_summary_Divergence_sample_p_'+str(int(100*float(per))) +'_trial_' + str(rep)+'.csv','wb')
            for line in results:
                we.write(line)
            we.close() 

        ####Repeat 5 time for the average values stat############
        ###################################
        ###################################
        average_results=[]
        result_repeats=[]
        if sampling_algorithm =='top_k_leader_internal_priority_extra':       
            sample_title_str ='Sample_algorithm'+','+title_str
            average_results.append(sample_title_str) 
        for rep in xrange(1, repeats):
            average_result_single = list(pd.read_csv(Result_folder + data_set+'_'+clustering_algorithm+'_'+sampling_algorithm+'_summary_Divergence_sample_p_'+str(int(100*float(per))) +'_trial_' + str(rep)+'.csv', sep=",", dtype="float64").mean())           
            result_repeats.append(average_result_single)
        print('result_repeats:',result_repeats) 
        print('\n') 
        print('\n') 
        average_result_repeats = np.mean(np.array(result_repeats),axis=0).tolist()
        print('average_result_repeats:',average_result_repeats)        
        print('\n') 
        print('\n')    
      
        newitem=[]
        newitem.append(sampling_algorithm)

        for e in average_result_repeats:
            str_e = str(e)
            newitem.append(str_e)  
       
        newstr= ','.join(newitem)
        newstr= newstr+'\n'
        print('newstr:', newstr)
        print('\n')
        average_results.append(newstr)
        we=open(Result_folder +'All_repeats_'+data_set+'_'+clustering_algorithm+'_summary_Divergence_sample_p_'+str(int(100*float(per))) + '.csv','ab')
        for line in average_results:
            we.write(line)
        we.close()

    ####Average size of different snapshots############
    average_nodes=sum(num_nodes_snapshots)/len(num_nodes_snapshots)
    print "number of nodes of each snapshot: %s" %average_nodes  
    print "writing completed"