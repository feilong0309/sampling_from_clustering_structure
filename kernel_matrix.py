import sys;
#import logging;
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
try:
    trainfile = sys.argv[1];
    N = int(sys.argv[2]);
    testfile = sys.argv[3];
    if (testfile=="null"):
        fp=open(trainfile+".txt",'r');
        fp1=0;
        fout = open("Omega.txt",'w');
        fpdict = open("Dictionary.txt","w");
	fpmaxkey = open("Maxkey.txt","w");
    else:
        fp = open(trainfile+".txt","r");
        fp1 = open(testfile+".txt","r");
        fpdict = open("Dictionary.txt","r");
	fpmaxkey = open("Maxkey.txt","r");
        fout1 = open("Omega_test.txt",'w');
except:
    sys.exit("File not exists. Make sure of the right filename");
#Checked whether we are generating Omega only for training or for testing as well
from string import *;
from math import *;
import numpy;
from gensim import corpora,models
from gensim import similarities;
from decimal import Decimal;
if (fp1==0):
    graph_connection,dictionary_features = {},{}
    corpus = [];
    count=1;
    for line in fp:
        data = split(strip(line,"\r\n"),sep=",");
        node1 = int(Decimal(data[0]));
        node2 = int(Decimal(data[1]));
        weight = float(data[2]);
        if (node1 not in graph_connection):
            graph_connection[node1]=[(node2,weight)];
        else:
            graph_connection[node1].append((node2,weight));
        if (node2 not in dictionary_features):
            dictionary_features[node2]=count;
            count=count+1;
    for key,value in dictionary_features.iteritems():
        fpdict.write(str(key)+" "+str(value)+"\n");
    #Obtain the features of the graph
    keylist = graph_connection.keys();
    keylist.sort();
    maxkey = max(keylist);
    fpmaxkey.write(str(maxkey)+"\n");
    for key in keylist:
        new_vec = [];
        for data in graph_connection[key]:
            if (data[0] in dictionary_features):
                new_vec.append((dictionary_features[data[0]],data[1]));
        #if new_vec!=[]:
        corpus.append(new_vec);
    tfidf = models.TfidfModel(corpus);
    tfidf.save('/users/sista/rmall/MatlabWork/KSC_LargeScale/Python_Code/temp.tfidf_model');
    #index = similarities.MatrixSimilarity(tfidf[corpus],N);
    index = similarities.Similarity('/users/sista/rmall/MatlabWork/KSC_LargeScale/Python_Code/temp',tfidf[corpus], num_features=len(dictionary_features.keys())+1, num_best=N, chunksize=1024, shardsize=32768);
    index.save('/users/sista/rmall/MatlabWork/KSC_LargeScale/Python_Code/temp.index');
    for key in keylist:
        vec=[];
        for data in graph_connection[key]:
            if (data[0] in dictionary_features):
                vec.append((dictionary_features[data[0]],data[1]));
        sims = index[tfidf[vec]];
        for data in sims:
            node=data[0]+1;
    	    if (data[1]>=1.0):
                simvalue=1;    
            else:  
                simvalue = data[1];     
            fout.write(str(key)+" "+str(node)+" "+str(simvalue)+"\n");
else:
    graph_connection,dictionary_features,test_connection = {},{},{};
    corpus = [];
    count=1;
    maxkey=0;
    for line in fpmaxkey:
	data = strip(line,"\r\n");
	maxkey = int(data);
    for line in fpdict:
        data = split(strip(line,"\r\n"),sep=" ");
        ident = int(Decimal(data[0]));
        value = int(Decimal(data[1]));
        dictionary_features[ident]=value;
    for line in fp1:
        data = split(strip(line,"\r\n"),sep=",");
        node1 = int(Decimal(data[0]));
        node2 = int(Decimal(data[1]));
        weight = float(data[2]);
        if (node1 not in test_connection):
            test_connection[node1] = [(node2,weight)];
        else:
            test_connection[node1].append((node2,weight));
    tfidf = models.TfidfModel.load('/users/sista/rmall/MatlabWork/KSC_LargeScale/Python_Code/temp.tfidf_model');
    #index = similarities.MatrixSimilarity(tfidf[corpus],N);
    #index = similarities.Similarity('/users/sista/rmall/MatlabWork/KSC_LargeScale/Python_Code/temp',tfidf[corpus], num_features=len(dictionary_features.keys())+1, num_best=N, chunksize=1024, shardsize=32768);
    index = similarities.Similarity.load('/users/sista/rmall/MatlabWork/KSC_LargeScale/Python_Code/temp.index');
    for key in test_connection.keys():
        flag=0;
        vec=[];
        for data in test_connection[key]:
            if (data[0] in dictionary_features):
                vec.append((dictionary_features[data[0]],data[1]));
        sims = index[tfidf[vec]];
        if not sims:
            flag=1;
            fout1.write(str(key)+" "+str(maxkey)+" "+str(0)+"\n");
        for data in sims:
            node = data[0]+1;
            if (node==maxkey):
                flag=1;
            if (data[1]>=1.0):
                simvalue=1;
            else:
             	simvalue = data[1];     
            fout1.write(str(key)+" "+str(node)+" "+str(simvalue)+"\n");
        if flag==0:
            fout1.write(str(key)+" "+str(maxkey)+" "+str(0.0)+"\n");

#Initialization of dictionary of dictionary
#class AutoVivification(dict):
#    """Implementation of perl's autovivification feature."""
#    def __getitem__(self, item):
#        try:
#            return dict.__getitem__(self, item)
#        except KeyError:
#            value = self[item] = type(self)()
#            return value
#Case when we construct Omega only for training data
#if (fp1==0):
#    graph_connection = AutoVivification();
#    for line in fp:
#        data = split(strip(line,"\r\n"),sep=",");
#        node1 = data[0];
#        node2 = data[1];
#        weight = float(data[2]);
#        graph_connection[node1][node2]=weight;
#    for key1,values1 in graph_connection.iteritems():
#        for key2,values2 in graph_connection.iteritems():
#            list1 = values1.values();
#            norm1 = numpy.linalg.norm(numpy.array(list1));
#            list2 = values2.values();
#            norm2 = numpy.linalg.norm(numpy.array(list2));
#            cosineval = 0.0;
#            if (len(values1.keys())<len(values2.keys())):
#                intersect = [values1[item]*values2[item] for item in values1.keys() if values2.has_key(item)];
#                cosineval = sum(intersect);
#            else:
#                intersect = [values1[item]*values2[item] for item in values2.keys() if values1.has_key(item)];
#                cosineval = sum(intersect);
#            cosineval = (1.0*cosineval)/(norm1*norm2);
#            fout.write(key1+" "+key2+" "+str(cosineval)+"\n");
#    fp.close();
#    fout.close();
#else:
#    train_connection = AutoVivification();
#    test_connection = AutoVivification();
#    for line in fp:
#        data = split(strip(line,"\r\n"),sep=",");
#        node1 = data[0];
#        node2 = data[1];
#        weight = float(data[2]);
#        train_connection[node1][node2]=weight;
#    for line in fp1:
#        data = split(strip(line,"\r\n"),sep=",");
#        node1 = data[0];
#        node2 = data[1];
#        weight = float(data[2]);
#        test_connection[node1][node2]=weight;
#        for key1,values1 in train_connection.iteritems():
#            for key2,values2 in test_connection.iteritems():
#                list1 = values1.values();
#                norm1 = numpy.linalg.norm(numpy.array(list1));
#                list2 = values2.values();
#                norm2 = numpy.linalg.norm(numpy.array(list2));
#                cosineval = 0.0;
#                if (len(values1.keys())<len(values2.keys())):
#                    intersect = [values1[item]*values2[item] for item in values1.keys() if values2.has_key(item)];
#                    cosineval = sum(intersect);
#                else:
#                    intersect = [values1[item]*values2[item] for item in values2.keys() if values1.has_key(item)];
#                    cosineval = sum(intersect);
#                    cosineval = (1.0*cosineval)/(norm1*norm2);
#                    fout1.write(key1+" "+key2+" "+str(cosineval)+"\n");
#    fp.close();
#    fp1.close();
#    fout.close();
#    fout1.close();
