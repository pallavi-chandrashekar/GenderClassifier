import json
from collections import Counter, defaultdict, deque
import copy
import networkx as nx
import re
import itertools
import pickle
import twitterCalls as f1

def getFriends():
    Twitter = f1.getTwitter()
    friends = defaultdict(list)
    tweet_list = []
    data= open('tweets.txt','rb')
    d = pickle.load(data)
    for k,v in d.items():
        tweet_list.append(v)
    count = 0
    for tweet in tweet_list[:10]:
        request = f1.getRequest(Twitter, 'friends/list', {'screen_name': tweet['user']['screen_name'], 'count': 200})
        for r in request:
            friends[tweet['user']['screen_name']].append(r['screen_name'])
        print(count)
        count += 1
    print(friends)
    f = open('Friends.txt', 'w')
    for k,v in friends.items():
        for i in v:
            f.write(str(k)+"\t"+str(i)+"\n")
    f.close()
    

def readGraph():
    graph=  nx.read_edgelist('Friends.txt', delimiter='\t')
    return graph



def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.
    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque
    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.
    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node to this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    """
    ###TODO
    pass
    N2P = defaultdict()
    N2D = defaultdict()
    paths = defaultdict()
    deq = deque()
    nodes = graph.nodes()
    travesed={x:"No" for x in nodes}
    dis= {x:float("inf") for x in nodes}
    parents = {x:[] for x in nodes}

    dis[root] = 0
    parents[root] = root
    deq.append(root)

    while(len(deq) != 0):
        u = deq.popleft()
        u_children = graph.neighbors(u)
        if(dis[u] > max_depth):
            break
        for child in u_children:
            if(travesed[child] == "No"):
                if(dis[child] >= dis[u] + 1 ):
                    dis[child] = dis[u] + 1
                    parents[child].append(u)
                    deq.append(child)

        travesed[u] = "Yes"

    for node in travesed:
        if(travesed[node] == "Yes"):
            N2D[node] = dis[node]
            paths[node] = len(parents[node])
            if(node != root):
                N2P[node] = parents[node]

    return N2D, paths, N2P

def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...
    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).
    """
    ###TODO
    pass
    N2D = node2distances
    N2P = node2parents
    max_distance = max(N2D.values())
    nodes_maxdist = [x for x,v in N2D.items() if v==max_distance]
   
    NW = {x:1 for x in N2D.keys()}
    E = []
   
    for key in sorted(N2D, key=N2D.get,reverse=True):
        if(key != root):
            for parent in N2P[key]:
                NW[parent] += (NW[key]/len(N2P[key]))
                if(key > parent):
                    E.append((parent, key))
                else:
                    E.append((key, parent))

    final_val = defaultdict()

   
    for e in E:
        if( root in e):
            index_of_root = e.index(root)
            if(index_of_root > 0):
                final_val[e] = (NW[e[0]]/len(N2P[e[0]]))
            else:
                final_val[e] = (NW[e[1]]/len(N2P[e[1]]))
        elif(e[0] in N2P[e[1]]):
            final_val[e] = (NW[e[1]]/len(N2P[e[1]]))
        else:
            final_val[e] = (NW[e[0]]/len(N2P[e[0]]))

    return final_val

def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.
    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.
    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).
    """
    ###TODO
    pass
    dict_edge = dict()
    for i in graph.nodes():
        N2D, paths, N2P = bfs(graph, i, max_depth)
        edge_dict = bottom_up(i, N2D, paths, N2P)
        for key, val in edge_dict.items():
            if (key in dict_edge.keys()):
                dict_edge[key] = dict_edge[key] + val
            else:
                dict_edge[key] = val

    for key, val in dict_edge.items():
        dict_edge[key] = float(val / 2)
    return dict_edge


def partition_girvan_newman(graph, max_depth):
	copy_graph = graph.copy()
	app_val = approximate_betweenness(copy_graph, max_depth)
	app_val = sorted(app_val.items(), key=lambda x: (-x[1], x[0]))
	for i in app_val:
		copy_graph.remove_edge(i[0][0], i[0][1])
		components = list(nx.connected_component_subgraphs(copy_graph))
		if len(components) > 1:
			break
    
	return components



    #graph1 = graph.copy()
    #E2B = approximate_betweenness(graph1, max_depth)
    #E2B = sorted(E2B.items(), key=lambda x : (-x[1], x[0]))

    #edgeList = iter(E2B)
    #while nx.is_connected(graph1):
    #    [k, v] = next(edgeList)
    #    graph1.remove_edge(*k)

    #return list(nx.connected_component_subgraphs(graph1))

def main():
    getFriends()
    graph = readGraph()
    clusters = partition_girvan_newman(graph, 5)
    commuityCluster = []
    for i in range(len(clusters)):
        commuityCluster.append(clusters[i].nodes())
    data = open('Cluster.txt', 'wb')
    pickle.dump(commuityCluster, data)
    print("Number of Clusters:",len(commuityCluster))
    print("Cluster data stored in Cluster.txt")


if __name__ == '__main__':
    main()
