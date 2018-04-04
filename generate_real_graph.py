## Author: Scott Emmons (scott@scottemmons.com)
## Purpose: A script to generate multiple LFR benchmark graphs based on the given parameters
## Date: January 2, 2014

import argparse
import os
import shutil
import errno
import subprocess
import networkx as nx

####################
# Global Variables #
####################

flag_file_name = "myflags.dat"

##################
# Main Functions #
##################

def handleArgs():
    """Handle command-line input arguments."""

    parser = argparse.ArgumentParser(description="Generate LFR benchmark graphs.")
    parser.add_argument("-n", "--nodes", type=int, required=True, help="the number of nodes", dest="N")
    parser.add_argument("-k", "--avgdegree", default=25, type=int, help="the average degree of the nodes, defaults to 25", dest="k")
    parser.add_argument("--maxk", "--maxdegree", type=int, required=True, help="the maximum degree of the nodes", dest="maxk")
    parser.add_argument("--mu", type=float, required=True, help="the mixing parameter", dest="mu")
    parser.add_argument("--minc", default=50, type=int, help="the minimum community size, defaults to 50", dest="minc")
    parser.add_argument("--maxc", type=int, required=True, help="the maximum community size", dest="maxc")
    parser.add_argument("-s", "--start", default=1, type=int, help="the file number at which to start, inclusive", dest="start")
    parser.add_argument("-e", "--end", default=10, type=int, help="the file number at which to end, inclusive", dest="end")
    parser.add_argument("-b", "--benchmark", default="binary_networks/", help="the path to the installed LFR generation software", dest="bench_directory_stem")
    parser.add_argument("-o", "--output", default="generated_benches/", help="the output path, defaults to 'generated_benches/'", dest="out_directory_stem")

    global args
    args = parser.parse_args()

def deletePathIfNeeded(path):
    try:
        shutil.rmtree(path)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
            
def createPathIfNeeded(path):
    """Credits to user 'Heikki Toivonen' on SO: http://stackoverflow.com/questions/273192/check-if-a-directory-exists-and-create-it-if-necessary"""
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

def getMinEdgelistId(edgelist_file, separator):
    """"""

    with open(edgelist_file, 'r') as f:
        source_id, destination_id = f.readline().split(separator)
        destination_id = destination_id[:-1] #remove newline character
        min_id = min(int(source_id), int(destination_id))
        for line in f:
            source_id, destination_id = line[:-1].split(separator) #line[:-1] removes newline character from destination_id 
            min_id = min(int(source_id), int(destination_id), min_id)

    return min_id

def rewriteEdgelistFromZero(graph_file, separator):
    """"""

    temporary_file = 'temporary_program_file_s_' + str(args.start) + '_e_' + str(args.end) + '.dat'
#    assert not os.path.isfile(temporary_file)

    min_id = getMinEdgelistId(graph_file, separator)
    source = open(graph_file, 'r')
    destination = open(temporary_file, 'wb')

    for line in source:
        source_id, destination_id = line[:-1].split(separator) #line[:-1] removes newline character from destination_id
        source_id = str(int(source_id) - min_id)
        destination_id = str(int(destination_id) - min_id)
        destination.write(source_id + separator + destination_id + '\n')

    source.close()
    destination.close()

    shutil.move(temporary_file, graph_file)

def getMinClusteringId(clustering_file, separator):
    """"""
    
    with open(clustering_file, 'r') as f:
        min_id = int(f.readline().split(separator)[0])
        for line in f:
            node_id = int(line.split(separator)[0])
            min_id = min(node_id, min_id)

    return min_id

def rewriteClusteringFromZero(clustering_file, separator):
    """"""

    temporary_file = 'temporary_program_file_s_' + str(args.start) + '_e_' + str(args.end) + '.dat'
#    assert not os.path.isfile(temporary_file)

    min_id = getMinClusteringId(clustering_file, separator)
    source = open(clustering_file, 'r')
    destination = open(temporary_file, 'wb')

    for line in source:
        node_id, cluster_id = line[:-1].split(separator) #line[:-1] removes newline character from cluster_id
        node_id = str(int(node_id) - min_id)
        destination.write(node_id + separator + cluster_id + '\n')

    source.close()
    destination.close()

    shutil.move(temporary_file, clustering_file)

def generateFlagFile(file_name, out_directory_stem, N, k, maxk, mu, minc, maxc):
    """file_name: String
    out_directory_stem: String
    N: int
    mu: float"""

    to_write = ""

    to_write += "-N " + str(N) + "\n"
    to_write += "-k " + str(k) + "\n"
    to_write += "-maxk " + str(maxk) + "\n"
    to_write += "-mu " + str(mu) + "\n"
    to_write += "-t1 2\n"
    to_write += "-t2 1\n"
    to_write += "-minc " + str(minc) + "\n"
    to_write += "-maxc " + str(maxc) + "\n"
    to_write += "-on 0\n"
    to_write += "-om 0\n"

    f = open(out_directory_stem + file_name, 'w')
    f.write(to_write)

def removeDuplicateEdges(filename, separator, assume_one_max = False):
    """Given a network file separated by separator, removes edges such that the final network file_name
    contains no two edges that connect the same pair of nodes.
    Assumes node ids and cluster ids are integers.
    If assume_one_max, the function will assume that there are at most two
    edges in the original file connecting the same pair of nodes."""

    read_file = filename
    write_file = "temporary_function_execution_s_" + str(args.start) + "_e_" + str(args.end) + ".dat"
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
            write_file.write(node + '\t' + cluster_number_string + '\n')
        cluster_number_string = str(int(cluster_number_string) + 1)

    #print('\nSuccessfully ran transfering and wrote results to file ' + out_file + '\n')

    read_file.close()
    write_file.close() 
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

if __name__ == "__main__":

    handleArgs()
    generateFlagFile(flag_file_name, args.bench_directory_stem, args.N, args.k, args.maxk, args.mu, args.minc, args.maxc)
    deletePathIfNeeded(args.out_directory_stem)  
    createPathIfNeeded(args.out_directory_stem)
    
    for i in xrange(args.start, args.end + 1):
        if args.mu == 0.95:
            shutil.copy(args.bench_directory_stem + 'polbooks/network_v1.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'polbooks/community_v1.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')
        
        elif args.mu == 0.90:
            shutil.copy(args.bench_directory_stem + 'dolphins/network_v1.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'dolphins/community_v1.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')

        elif args.mu == 0.85:
            shutil.copy(args.bench_directory_stem + 'polblogs/network_v1_normalized.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'polblogs/community_v1_normalized.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')

        elif args.mu == 0.75:
            shutil.copy(args.bench_directory_stem + 'football/network_v1.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'football/community_v1.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')

        elif args.mu == 0.65:
            shutil.copy(args.bench_directory_stem + 'karate/network_v1_normalized.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'karate/community_v1_normalized.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')

        elif args.mu == 0.6:
            shutil.copy(args.bench_directory_stem + 'friendster/network_v1_subgraph_speed_up_normalized.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'friendster/community_v1_normalized.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')
            #com-friendster.top5000.cmty.txt
        elif args.mu == 0.5:
            shutil.copy(args.bench_directory_stem + 'dblp/network_v1_subgraph_speed_up_normalized.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'dblp/community_v1_normalized.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')
            #com-dblp.top5000.cmty.txt
        
        elif args.mu == 0.4:
            shutil.copy(args.bench_directory_stem + 'amazon/network_v1_subgraph_speed_up_normalized.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'amazon/community_v1_normalized.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')
            #com-amazon.top5000.cmty.txt
            
        elif args.mu == 0.3:
            # shutil.copy(args.bench_directory_stem + 'com_LiveJournal/network_v1_subgraph_speed_up.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            # shutil.copy(args.bench_directory_stem + 'com_LiveJournal/community_v1.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')             #com-lj.top5000.cmty.txt
            shutil.copy(args.bench_directory_stem + 'com_LiveJournal/network_v1_subgraph_speed_up_normalized.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'com_LiveJournal/community_v1_normalized.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')             #com-lj.top5000.cmty.txt
                        
        elif args.mu == 0.2:
            shutil.copy(args.bench_directory_stem + 'youtube/network_v1_subgraph_speed_up_normalized.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'youtube/community_v1_normalized.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')                #com-youtube.top5000.cmty.txt        
        
        elif args.mu == 0.1: 
            shutil.copy(args.bench_directory_stem + 'orkut/network_v1_subgraph_speed_up_normalized.dat', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'orkut/community_v1_normalized.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')

        elif args.mu == 0.8: 
            shutil.copy(args.bench_directory_stem + 'email-Eu-core/email-Eu-core.txt', args.out_directory_stem + 'network_v' + str(i) + '.dat')
            shutil.copy(args.bench_directory_stem + 'email-Eu-core/email-Eu-core-department-labels.txt', args.out_directory_stem + 'community_v' + str(i) + '.dat')
        else:
            print("no dataset need to copy")

        # Does seed file need to be handled here?
#        subprocess.call(['./benchmark', '-f', flag_file_name], cwd = args.bench_directory_stem)
    

#        Statistics of the graphs. 
        if 0:
#            parseLancichinettiResults(args.out_directory_stem + 'intermediate_community_v' + str(i) + '.dat', args.out_directory_stem + 'community_v' + str(i) + '.dat')
#            G = nx.read_edgelist(path=readfile, delimiter=",", nodetype=int,  create_using=nx.Graph())
#            start = 0
#            G_ = nx.convert_node_labels_to_integers(G, first_label=start)
#            print 'number of unique nodes:',G_.number_of_nodes()
#            print 'number of unique edges:',G_.number_of_edges()
#            print 'number of nodes:',sample.graph.get('number_of_nodes_repeated',0)
#            print 'number of edges:',sample.graph.get('number_of_edges_repeated',0)
#            print 'nodes',sample.nodes()
             
            original=open(args.out_directory_stem + 'network_v' + str(i) + '.dat', 'rb')
#            G_multi=nx.read_edgelist(original, nodetype=int,data=False, create_using=nx.MultiGraph())
#            num_nodes_multi = nx.number_of_nodes(G_multi)
#            num_edges_multi = nx.number_of_edges(G_multi)
#            print('number of multiple nodes: %d' % (num_nodes_multi))
#            print('number of multiple edges: %d' % (num_edges_multi))            

            G=nx.read_edgelist(original, nodetype=int, create_using=nx.Graph())            
            original_=maximum_connected_components(G)
            num_nodes = nx.number_of_nodes(original_)
            num_edges = nx.number_of_edges(original_)
            print('number of nodes: %d' % (num_nodes))
            print('number of edges: %d' % (num_edges))
            graph_density = nx.density(original_)
            print('graph_density: %f' % (graph_density))
            graph_clustering_coefficient = nx.average_clustering(original_)
        #    graph_clustering_coefficient = nx.average_clustering(original_, trials=1000)
            print('graph_clustering_coefficient: %f' % (graph_clustering_coefficient))
            diameter = nx.diameter(original_)
            print('graph_diameter: %f, %s' % (diameter, args.out_directory_stem + 'network_v' + str(i) + '.dat'))
            radius = nx.radius(original_)
            print('graph_radius: %f, %s' % (radius, args.out_directory_stem + 'network_v' + str(i) + '.dat'))   
            original.close()           
            
            

       # Remove duplicate edges from edgelist file and rewrite edgelist file such that node ids start from zero for compatibility with clustering program input formats
        
        #removeDuplicateEdges(args.out_directory_stem + 'network_v' + str(i) + '.dat', '\t', assume_one_max = False)
        #rewriteEdgelistFromZero(args.out_directory_stem + 'network_v' + str(i) + '.dat', '\t')
        
        # Rewrite clustering file such that node ids start from zero to maintain consistency with edgelist file node ids
        #rewriteClusteringFromZero(args.out_directory_stem + 'community_v' + str(i) + '.dat', '\t')
 
    shutil.move(args.bench_directory_stem + flag_file_name, args.out_directory_stem + "flags.dat")