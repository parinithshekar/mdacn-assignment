import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd

import a
import b
import c

FILE = "./network.xlsx"
NODES = 0
T = 0

def make_simple_edges(excel_file):
    linked_list = []
    for row_index in range(excel_file.index.size):
        linked_list.append((excel_file.at[row_index, 'node1'], excel_file.at[row_index, 'node2']))
    return list(set(linked_list))


def make_timestamped_edges(excel_file):
    linked_list_with_timestamp = []
    for row_index in range(excel_file.index.size):
        linked_list_with_timestamp.append((excel_file.at[row_index, 'node1'], excel_file.at[row_index, 'node2'],
                                           excel_file.at[row_index, 'timestamp']))
    return linked_list_with_timestamp


def make_g2_reshuffled_timestamped_edges(excel_file):
    g2_file = excel_file.copy(deep=True)

    # shuffle timestamps
    timestamps = np.array(g2_file["timestamp"])
    np.random.shuffle(timestamps)
    g2_file["timestamp"] = timestamps

    linked_list = []
    for row_index in range(excel_file.index.size):
        linked_list.append(
            (g2_file.at[row_index, 'node1'], g2_file.at[row_index, 'node2'],
            g2_file.at[row_index, 'timestamp']))
    return sorted(linked_list, key=lambda edge: edge[2])

def make_g3_reassigned_timestamp_edges(excel_file):
    timestamps = np.array(excel_file["timestamp"])
    unique_edges = make_simple_edges(excel_file)
    n_edges = len(unique_edges)
    linked_list = []
    for ts in timestamps:
        random_index = np.random.randint(0, n_edges)
        linked_list.append((excel_file.at[random_index, 'node1'], excel_file.at[random_index, 'node2'], ts))
    return linked_list


def update_step(known_edges, node1, node2):
    edge_index = -1
    for i in range(len(known_edges)):
        node1_edge, node2_edge, weight_edge = known_edges[i]
        if (node1_edge == node1 and node2_edge == node2) or (node2_edge == node2 and node1_edge == node1):
            edge_index = i
            known_edges[i] = (node1_edge, node2_edge, weight_edge + 1)
    if edge_index == -1:
        known_edges.append((node1, node2, 1))
    return known_edges


def make_weighted_edges(excel_file):
    linked_list_weights = []
    for row_index in range(excel_file.index.size):
        node1 = excel_file.at[row_index, 'node1']
        node2 = excel_file.at[row_index, 'node2']
        linked_list_weights = update_step(linked_list_weights, node1, node2)
        # print(excel_file.at[row_index, 'timestamp'])
    final_edges = [(edge[0], edge[1], {'weight': [edge[2]]}) for edge in linked_list_weights]
    return final_edges


def part_a(excel_file):
    print("A")
    simple_edges = make_simple_edges(excel_file)
    weighted_edges = make_weighted_edges(excel_file)

    g = nx.Graph()
    g.add_edges_from(simple_edges)

    gw = nx.Graph()
    gw.add_edges_from(weighted_edges)

    a.question1(g)
    a.question2(g)
    a.question3(g)
    a.question4(g)
    a.question5(g)
    a.question6(g)
    a.question7(g)
    a.question8(weighted_edges)


def part_b(excel_file):
    print("B")
    simple_edges = make_simple_edges(excel_file)
    weighted_edges = make_weighted_edges(excel_file)
    timestamped_edges = make_timestamped_edges(excel_file)

    g = nx.Graph()
    g.add_edges_from(simple_edges)

    (all_infected_nodes,
        node_efficiency_temp,
        all_nodes_first_contact,
        majority_nodes_first_contact
        ) = b.get_infection_metrics(NODES, T, timestamped_edges)

    # 9
    infection_mean = np.array(all_infected_nodes).mean(axis=0)
    infection_std = np.array(all_infected_nodes).std(axis=0)
    b.plot_infection_stats(infection_mean, infection_std)
    
    # 10
    node_efficiency = b.get_node_efficiency(node_efficiency_temp)

    # 11
    # get degree list sorted
    degrees = b.get_degrees(NODES, g)
    strengths = b.get_strengths(NODES, weighted_edges)

    # 12
    first_contacts = b.get_first_contacts(all_nodes_first_contact)

    # 13
    rprime = b.get_rprime(majority_nodes_first_contact)

    # Recognition rate for degrees (rrd), strengths (rrs) and z (rrz) to predict efficiency (R)
    ((rrd, rrs, rrz),
    # Recognition rate for efficiency (rpr), degrees (rpd), strengths (rps), z (rpz) tp predict rprime (R')
    (rpr, rpd, rps, rpz)) = b.get_f_recognition(
        NODES, node_efficiency, degrees, strengths, first_contacts, rprime)

    f = [0.05*i for i in range(1, 11)]
    b.plot_f_recognition_efficiency(f, rrd, rrs, rrz)
    b.plot_f_recognition_rprime(f, rpr, rpd, rps, rpz)


def part_c(excel_file):
    print("C")
    timestamped_edges = make_timestamped_edges(excel_file)
    g2_edges = make_g2_reshuffled_timestamped_edges(excel_file)
    g3_edges = make_g3_reassigned_timestamp_edges(excel_file)

    # 14
    gdata_inter_arrivals = c.get_inter_arrival_times(timestamped_edges)
    g2_inter_arrivals = c.get_inter_arrival_times(g2_edges)
    g3_inter_arrivals = c.get_inter_arrival_times(g3_edges)

    gdata_all_infected_nodes, _, _, _ = b.get_infection_metrics(NODES, T, timestamped_edges)
    g2_all_infected_nodes, _, _, _ = b.get_infection_metrics(NODES, T, g2_edges)
    g3_all_infected_nodes, _, _, _ = b.get_infection_metrics(NODES, T, g3_edges)

    # 15
    gdata_infection_mean = np.array(gdata_all_infected_nodes).mean(axis=0)
    gdata_infection_std = np.array(gdata_all_infected_nodes).std(axis=0)
    g2_infection_mean = np.array(g2_all_infected_nodes).mean(axis=0)
    g2_infection_std = np.array(g2_all_infected_nodes).std(axis=0)
    g3_infection_mean = np.array(g3_all_infected_nodes).mean(axis=0)
    g3_infection_std = np.array(g3_all_infected_nodes).std(axis=0)

    # 14
    c.plot_inter_arrival_density(gdata_inter_arrivals, g2_inter_arrivals, g3_inter_arrivals)
    
    # 15
    c.plot_infection_stats(
        gdata_infection_mean, gdata_infection_std,
        g2_infection_mean, g2_infection_std,
        g3_infection_mean, g3_infection_std
    )
    # How to quantify information spreading performance?

def start_part(user_input, excel_file):
    if user_input == "a":
        part_a(excel_file)
    elif user_input == "b":
        part_b(excel_file)
    elif user_input == "c":
        part_c(excel_file)
    else:
        print("nice monkey")

def main():
    global NODES
    global T
    # print("Assignment")
    # print("Part to do?")
    # part_chosen = input()
    file = pd.read_excel(FILE)

    # Set global constants
    node1 = file["node1"]
    node2 = file["node2"]
    timestamp = file["timestamp"]
    node_min = min(node1.min(), node2.min())
    node_max = max(node1.max(), node2.max())
    NODES = node_max - node_min + 1
    T = timestamp.max()

    start_part("b", file)

if __name__ == '__main__':
    main()
