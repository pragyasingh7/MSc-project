import community
import networkx as nx


def calc_nodal_partition(G):
    # Make sure the edges are binarized
    for u, v, d in G.edges(data=True):
        if d.get('weight', 1) != 1:
            raise ValueError("G should be a binary graph")        
            
    # Now calculate the best partition
    nodal_partition = community.best_partition(G)

    # Reverse the dictionary to record a list of nodes per module, rather than
    # module per node
    module_partition = {}
    for n, m in nodal_partition.items():
        try:
            module_partition[m].append(n)
        except KeyError:
            module_partition[m] = [n]

    return nodal_partition, module_partition

def participation_coefficient(G):
    ## Should run on degree mode, not strength
    copy_G = G.copy()
    for (u, v) in copy_G.edges():
        copy_G.edges[u,v]['weight'] = 1
        
    module_partition = calc_nodal_partition(copy_G)[1]
    
    # Initialise dictionary for the participation coefficients
    pc_dict = {}

    # Loop over modules to calculate participation coefficient for each node
    for m in module_partition.keys():
        # Create module subgraph
        M = set(module_partition[m])
        for v in M:
            # Calculate the degree of v in G
            degree = float(nx.degree(G=copy_G, nbunch=v))
            
            # Calculate the number of intramodule degree of v
            wm_degree = float(sum([1 for u in M if (u, v) in copy_G.edges()]))

            # The participation coeficient is 1 - the square of
            # the ratio of the within module degree and the total degree
            pc_dict[v] = (1 - ((float(wm_degree) / float(degree))**2)) if degree != 0 else 0

    pc_dict = dict(sorted(pc_dict.items()))
    return pc_dict


def small_worldness(G):
    # Ensure the graph is connected by taking the largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # Calculate the clustering coefficient for the original graph
    clustering_coeff = nx.average_clustering(G)
    
    # Calculate the average shortest path length for the original graph
    avg_shortest_path_length = nx.average_shortest_path_length(G)
    
    # Generate a random graph with the same number of nodes and edges
    random_graph = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
    
    # Ensure the random graph is connected
    while not nx.is_connected(random_graph):
        random_graph = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
    
    # Calculate the clustering coefficient for the random graph
    random_clustering_coeff = nx.average_clustering(random_graph)
    
    # Calculate the average shortest path length for the random graph
    random_avg_shortest_path_length = nx.average_shortest_path_length(random_graph)
    
    # Calculate the small-worldness metric
    small_worldness = (clustering_coeff / random_clustering_coeff) / (avg_shortest_path_length / random_avg_shortest_path_length)
    
    return small_worldness


def node_degree_centrality(G):
    return dict(nx.degree_centrality(G))

def betweenness_centrality(G):
    return dict(nx.betweenness_centrality(G))

def closeness_centrality(G):
    return dict(nx.closeness_centrality(G))

def eigenvector_centrality(G):
    return dict(nx.eigenvector_centrality(G))

def clustering_coefficient(G):
    return dict(nx.clustering(G))


def topK_subgraph(G, metric, k):
    if metric == 'degree':
        metric_dict = node_degree_centrality(G)
    elif metric == 'betweenness':
        metric_dict = betweenness_centrality(G)
    elif metric == 'closeness':
        metric_dict = closeness_centrality(G)
    elif metric == 'eigenvector':
        metric_dict = eigenvector_centrality(G)
    elif metric == 'clustering':
        metric_dict = clustering_coefficient(G)
    elif metric == 'participation':
        metric_dict = participation_coefficient(G)
    else:
        raise ValueError('Invalid metric')
    
    top_k_nodes = sorted(metric_dict, key=metric_dict.get, reverse=True)[:k]
    return G.subgraph(top_k_nodes)