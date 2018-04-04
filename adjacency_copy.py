def adjacency_copy(adjacency_matrix):
    import copy;
    import gc;
    mod_adjacency_matrix = copy.deepcopy(adjacency_matrix);
    del adjacency_matrix;
    gc.collect();
    return mod_adjacency_matrix;
