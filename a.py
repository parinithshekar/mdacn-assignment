from itertools import count
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def question1(g):
    print("1)")

    print(f'Number of nodes: {nx.number_of_nodes(g)}')
    print(f'Number of links: {nx.number_of_edges(g)}')
    degrees = np.array(g.degree)[:, 1]
    print(f'Mean: {degrees.mean()}')
    print(f'Standard deviation: {degrees.std()}')


def question2(g):
    print("2)")

    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
    degree, counts = np.unique(degree_sequence, return_counts=True)
    pdegree = counts / counts.sum()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.set_title('Histogram')
    ax1.bar(degree, counts)
    ax1.set_xlabel("Degree (k)")
    ax1.set_ylabel("Counts")

    ax2.set_title('Linear Linear scale')
    ax2.plot(degree, pdegree)
    ax2.set_xlabel("Degree (k)")
    ax2.set_ylabel("Pr[deg=k]")

    ax3.set_title('Log Log scale')
    ax3.loglog(degree, pdegree)
    ax3.set_xlabel("Degree (k)")
    ax3.set_ylabel("Pr[deg=k]")

    # fig = plt.figure("Degree distribution")
    # chart = fig.add_subplot()

    # chart.plot(degree, pdegree)
    # chart.loglog(degree, pdegree)
    # # chart.bar(*np.unique(degree_sequence, return_counts=True))
    # chart.set_xlabel("degree (k)")
    # chart.set_ylabel("Pr[degree=k]")
    plt.show()


def question3(g):
    print("3)")
    print(f'Degree correlation: {nx.degree_assortativity_coefficient(g)}')


def question4(g):
    print("4)")
    print(f'Average clustering coefficient: {nx.average_clustering(g)}')


def question5(g):
    print("5)")
    print(f'Average hopcount: {nx.average_shortest_path_length(g)}')
    print(f'Diameter: {nx.diameter(g)}')


def question6(g):
    # Have to fix this
    print("6)")
    print(f'Number of edges g: {g.number_of_edges()}')

    nb_nodes = g.number_of_nodes()
    # make cycle graphs
    for d in range(1, 11):
        greg = nx.Graph()
        for node_index in range(nb_nodes):
            for i in range(1, d + 1):
                node_index_target = (node_index + i) % nb_nodes
                if node_index_target == 0:
                    node_index_target = 167
                greg.add_edge(node_index, node_index_target)

        print(d)
        print(f'Average hopcount: {nx.average_shortest_path_length(greg)}')
        print(f'Diameter: {nx.diameter(greg)}')

    # make erdos renyi graphs
    for i in range(10):
        connected = False
        g_er = nx.Graph()
        while not connected:
            for edge in g.edges:
                node1 = rnd.randint(1, 167)
                node2 = rnd.randint(1, 167)
                g_er.add_edge(node1, node2)
            connected = nx.is_connected(g_er)
        print(f"Random graph with {g_er.number_of_edges()} edges")
        print(f'Average hopcount: {nx.average_shortest_path_length(g_er)}')
        print(f'Diameter: {nx.diameter(g_er)}')

    #is_small_world = False
    #if nx.average_shortest_path_length(g) < g.number_of_nodes() / 20:
    #    is_small_world = True
    #print(f'Small-world property? {is_small_world}')


def question7(g):
    print("7)")
    # print(f'Largest eigenvalue: {nx.laplacian_spectrum(g)}')
    print(f'Largest eigenvalue: {max(list(nx.adjacency_spectrum(g)))}')

def question8(weighted_edges):
    d = {}
    # Can modify this parameter to get different results
    dx = 500
    weights = np.array([edge[2]['weight'][0] for edge in weighted_edges])
    max_weight = np.max(weights)
    weights, counts = np.unique(weights, return_counts=True)
    total_edges = np.sum(counts)
    for weight, count in zip(weights, counts):
        d[weight] = count
    
    fwx = []
    for x in range(1, max_weight+1):
        res = 0
        for w in range(x, min(max_weight+1, x+dx+1)):
            if w in d:
                res += d[w]
        fwx.append((res/total_edges)/dx)

    plt.figure("Weight distribution")
    plt.plot(range(1, max_weight+1), fwx)
    plt.xlabel("Weight (w)")
    plt.ylabel("Probability density")
    plt.show()
