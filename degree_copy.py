def degree_copy(nodes):
    import copy;
    import gc;
    mod_nodes = copy.deepcopy(nodes);
    del nodes;
    gc.collect();
    return mod_nodes;
