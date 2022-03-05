import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_infection_metrics(number_of_nodes, time_steps, timestamped_edges):
    majority_threshold = math.ceil(number_of_nodes * 0.75)

    all_infected_nodes = []
    node_efficiency_temp = []
    # Denotes when a node first gets infected (averaged over different initializations)
    all_nodes_first_contact = []
    # First contact time for the first 126 (167 * 0.75) nodes
    majority_nodes_first_contact = []
    

    for loop_index in range(number_of_nodes):
        # The infected nodes
        infected_nodes_list = []
        # initialize the graph with only nodes and no links
        gdata = nx.Graph()
        for i in range(number_of_nodes):
            gdata.add_node(i + 1, status="susceptible")

        # initialize other stuff
        infected_nodes_over_time = [0 for i in range(time_steps)]
        nodes_first_contact = [time_steps for i in range(number_of_nodes)]
        node_index = loop_index + 1
        gdata.nodes[node_index]["status"] = "infected"
        infected_nodes_over_time[0] = 1
        infected_nodes = 1
        # print(gdata.nodes(data=True))
        nodes_to_be_updated = []
        majority_nodes_avg_contact_time = 0
        previous_timestamp = 0
        infected_nodes_list.append(node_index)

        # go through all edges
        for edge in timestamped_edges:
            node1, node2, current_timestamp = edge

            nodes_first_contact[node1-1] = min(previous_timestamp, nodes_first_contact[node1-1])
            nodes_first_contact[node2-1] = min(previous_timestamp, nodes_first_contact[node2-1])

            # check whether we are still at the same timestamp
            if current_timestamp != previous_timestamp:
                # if not, then update to infected all nodes that got infected during the previous timestamp
                previous_timestamp = current_timestamp
                for node in nodes_to_be_updated:
                    gdata.nodes[node]["status"] = "infected"
                    # nodes_first_contact[node-1] = previous_timestamp
                nodes_to_be_updated = []

            # check if the link is transmissive for the infection, if so update
            if gdata.nodes[node1]["status"] == "susceptible" and gdata.nodes[node2]["status"] == "infected" and node1 not in infected_nodes_list:
                nodes_to_be_updated.append(node1)
                majority_nodes_avg_contact_time += current_timestamp if (infected_nodes < majority_threshold) else 0
                infected_nodes_over_time[current_timestamp-1] += 1
                infected_nodes += 1
                infected_nodes_list.append(node1)
                
            elif gdata.nodes[node2]["status"] == "susceptible" and gdata.nodes[node1]["status"] == "infected" and node2 not in infected_nodes_list:
                nodes_to_be_updated.append(node2)
                majority_nodes_avg_contact_time += current_timestamp if (infected_nodes < majority_threshold) else 0
                infected_nodes_over_time[current_timestamp-1] += 1
                infected_nodes += 1
                infected_nodes_list.append(node2)
                

        # this is just to sum them all (cumulative number of infected nodes)
        for i in range(1, len(infected_nodes_over_time)):
            infected_nodes_over_time[i] += infected_nodes_over_time[i - 1]

        # efficiency is the first time step where there is more than 126 (167*0.75) nodes infected
        reach_majority = [i for i in range(len(infected_nodes_over_time)) if infected_nodes_over_time[i] >= majority_threshold]
        node_efficiency_temp.append(reach_majority[0] if len(reach_majority)>0 else time_steps)

        all_infected_nodes.append(infected_nodes_over_time)
        all_nodes_first_contact.append(nodes_first_contact)
        majority_nodes_first_contact.append(majority_nodes_avg_contact_time/majority_threshold)

    return all_infected_nodes, node_efficiency_temp, all_nodes_first_contact, majority_nodes_first_contact

def plot_infection_stats(infection_mean, infection_std):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title("Infected nodes mean")
    ax1.plot(infection_mean, label='Mean')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Mean infected nodes')
    ax1.legend()

    ax2.set_title("Infected nodes STD")
    ax2.plot(infection_std, label='STD')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Infected nodes STD')
    ax2.legend()

    plt.show()

def get_node_efficiency(node_efficiency_values):
    node_efficiency = []
    for node_index in range(len(node_efficiency_values)):
        node_efficiency.append((node_index+1, node_efficiency_values[node_index]))
    node_efficiency.sort(key=lambda gnode: gnode[1])
    return node_efficiency

def get_degrees(number_of_nodes, graph):
    degrees = []
    for node_index in range(1, number_of_nodes):
        degrees.append((node_index, graph.degree[node_index]))
    degrees.sort(key=lambda gnode: gnode[1])
    degrees.reverse()
    return degrees

def get_strengths(number_of_nodes, weighted_edges):
    # print(number_of_nodes)
    strengths_temp = [0 for i in range(number_of_nodes)]
    for edge in weighted_edges:
        node1, node2, weight = edge
        # print(len(strengths_temp))
        # print(node1, node2, weight.get("weight"))
        strengths_temp[node1-1] += weight.get("weight")[0]
        strengths_temp[node2-1] += weight.get("weight")[0]
    strengths = []
    for node_index in range(len(strengths_temp)):
        strengths.append((node_index+1, strengths_temp[node_index]))
    strengths.sort(key=lambda gnode: gnode[1])
    strengths.reverse()
    return strengths

def get_first_contacts(all_nodes_first_contact):
    all_nodes_first_contact = np.array(all_nodes_first_contact).sum(axis=0)
    first_contacts = []
    for i, fc in enumerate(all_nodes_first_contact):
        first_contacts.append((i+1, fc))
    first_contacts.sort(key=lambda gnode: gnode[1])
    return first_contacts

def get_rprime(majority_nodes_first_contact):
    rprime = []
    for i, rp in enumerate(majority_nodes_first_contact):
        rprime.append((i+1, rp))
    rprime.sort(key=lambda gnode: gnode[1])
    return rprime

def get_f_recognition(number_of_nodes, node_efficiency, degrees, strengths, first_contacts, rprime):

    # 11
    rrd = []
    rrs = []
    # 12
    rrz = []

    # 13
    rpr = []
    rpd = []
    rps = []
    rpz = []

    f_values = [0.05*i for i in range(1, 11)]

    for f in f_values:
        nb_d, nb_s, nb_z = 0, 0, 0
        nrp_r, nrp_d, nrp_s, nrp_z = 0, 0, 0, 0

        size = math.ceil(f*number_of_nodes)

        r = set(node[0] for node in node_efficiency[:size])
        d = set(node[0] for node in degrees[:size])
        s = set(node[0] for node in strengths[:size])
        z = set(node[0] for node in first_contacts[:size])
        rp = set(node[0] for node in rprime[:size])

        # 11
        nb_d = len(r.intersection(d))
        nb_s = len(r.intersection(s))
        # 12
        nb_z = len(r.intersection(z))
        # 13
        nrp_r = len(rp.intersection(r))
        nrp_d = len(rp.intersection(d))
        nrp_s = len(rp.intersection(s))
        nrp_z = len(rp.intersection(z))
        
        # 11
        rrd.append(nb_d/size)
        rrs.append(nb_s/size)
        # 12
        rrz.append(nb_z/size)
        # 13
        rpr.append(nrp_r/size)
        rpd.append(nrp_d/size)
        rps.append(nrp_s/size)
        rpz.append(nrp_z/size)
    
    return (rrd, rrs, rrz), (rpr, rpd, rps, rpz)

def plot_f_recognition_efficiency(f, degrees, strengths, first_contacts):
    plt.figure('F recognition for estimating efficiency')
    plt.plot(f, degrees, label='Degree')
    plt.plot(f, strengths, label='Strength')
    plt.plot(f, first_contacts, label='Z')
    plt.xlabel('F values')
    plt.ylabel('Recognition score')
    plt.legend()
    plt.show()

def plot_f_recognition_rprime(f, efficiency, degrees, strengths, first_contacts):
    plt.figure('F recognition for estimating R\'')
    plt.plot(f, efficiency, label='Efficiency')
    plt.plot(f, degrees, label='Degree')
    plt.plot(f, strengths, label='Strength')
    plt.plot(f, first_contacts, label='Z')
    plt.xlabel('F values')
    plt.ylabel('Recognition score')
    plt.legend()
    plt.show()