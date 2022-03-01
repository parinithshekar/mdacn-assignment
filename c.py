import numpy as np
import matplotlib.pyplot as plt

def get_inter_arrival_times(edges):
    d = {}
    inter_arrivals = []
    for timestamped_edge in edges:
        node1, node2, timestamp = timestamped_edge
        edge = frozenset((node1, node2))
        if edge in d:
            inter_arrivals.append(timestamp - d[edge])
        d[edge] = timestamp
    return inter_arrivals

def plot_infection_stats(gd_mean, gd_std, g2_mean, g2_std, g3_mean, g3_std):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title("Infected nodes mean")
    ax1.plot(gd_mean, label='Gdata')
    ax1.plot(g2_mean, label='G2')
    ax1.plot(g3_mean, label='G3')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Mean infected nodes')
    ax1.legend()

    ax2.set_title("Infected nodes STD")
    ax2.plot(gd_std, label='Gdata')
    ax2.plot(g2_std, label='G2')
    ax2.plot(g3_std, label='G3')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Infected nodes STD')
    ax2.legend()

    plt.show()

def plot_inter_arrival_density(gd, g2, g3):
    dgd, dg2, dg3 = {}, {}, {}
    dx = 20
    gdmax, g2max, g3max = np.max(gd), np.max(g2), np.max(g3)
    gdvalues, gdcounts = np.unique(gd, return_counts=True)
    g2values, g2counts = np.unique(gd, return_counts=True)
    g3values, g3counts = np.unique(gd, return_counts=True)
    gdtotal, g2total, g3total = np.sum(gdcounts), np.sum(g2counts), np.sum(g3counts)
    
    for value, count in zip(gdvalues, gdcounts):
        dgd[value] = count
    for value, count in zip(g2values, g2counts):
        dg2[value] = count
    for value, count in zip(g3values, g3counts):
        dg3[value] = count
    
    fgd = []
    for x in range(1, gdmax+1):
        res = 0
        for w in range(x, min(gdmax+1, x+dx+1)):
            if w in dgd:
                res += dgd[w]
        fgd.append((res/gdtotal)/dx)
    
    fg2 = []
    for x in range(1, g2max+1):
        res = 0
        for w in range(x, min(g2max+1, x+dx+1)):
            if w in dg2:
                res += dg2[w]
        fg2.append((res/g2total)/dx)
    
    fg3 = []
    for x in range(1, g3max+1):
        res = 0
        for w in range(x, min(g3max+1, x+dx+1)):
            if w in dg3:
                res += dg3[w]
        fg3.append((res/g3total)/dx)
    

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.set_title("Gdata interarrival distribution")
    ax1.loglog(fgd, label='Gdata', color='tab:blue')
    ax1.set_xlabel("Inter-arrival time")
    ax1.set_ylabel("Probability density")
    ax1.legend()

    ax2.set_title("G2 interarrival distribution")
    ax2.loglog(fg2, label='G2', color='tab:orange')
    ax2.set_xlabel("Inter-arrival time")
    ax2.set_ylabel("Probability density")
    ax2.legend()

    ax3.set_title("G3 interarrival distribution")
    ax3.loglog(fg3, label='G3', color='tab:green')
    ax3.set_xlabel("Inter-arrival time")
    ax3.set_ylabel("Probability density")
    ax3.legend()

    plt.show()