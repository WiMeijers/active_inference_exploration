import numpy as np
from pymdp import control, inference, utils
from ai_functions import update_posterior_policies
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.sparse import diags

## Basic example on the use of active inference with frontiers

def main():
    ## Creating the Generative Model
    sf = 2  # number of state factors
    obs = 1  # number of observations
    max_nodes = 7

    A = utils.obj_array(obs)
    B = utils.obj_array(sf)
    C = utils.obj_array(obs)
    D = utils.obj_array(3)

    D[0] = np.zeros(max_nodes)
    D[0][0] = 1
    D = generate_D_graph_size_normal(D, layer = 2, mu = 4.5, sigma = 1, 
                                        min = 2, max = max_nodes)
    D = generate_D_victim(D, layer=1, graph_size_layer = 2, mu = 4.5)

    D = D[0:2]

    AG = generate_line_graph(max_nodes)
    A = generate_A(A, AG)
    B = generate_agent_B(B, AG, layer=0)
    B = generate_independent_B(B, nodes = max_nodes, layers=[1, 2])
    C[0] = np.array([1, 0, 0])

    policy = [np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])]
    G = np.zeros(6)
    UT = np.zeros(6)
    SIG = np.zeros(6)
    for i in range(len(policy[0])):
        policies = [policy[0][0:i+1]]
        G[i], UT[i], SIG[i], _ = update_posterior_policies(D, A, B, C, policies)
    for i in range(1, len(G)):
        G[i] -= G[0:i].sum()
        SIG[i] -= SIG[0:i].sum()
        UT[i] -= UT[0:i].sum()
    x = np.linspace(1, 6, 6)
    dx = 1/len(G)
    w = dx * .8
    plt.figure(figsize = [6,4])
    plt.bar(x - dx, -G, width = w)
    plt.bar(x, -UT, width = w)
    plt.bar(x + dx, SIG, width = w)
    plt.legend(['neg. EFE', 'neg. Utility', 'Info Gain'])
    plt.ylim([0, 2])
    plt.show()
    

def generate_D_graph_size_normal(D, layer, mu, sigma, min, max):
    ## finds the belief on the total graph size, with:
    # min: the number the graph will at least consists off
    # max: the number the graph will at most consists off
    # mu: our initial guess on the graph size
    # sigma: our certainty on the graph size

    D[layer] = np.zeros(max)
    nd = norm(mu - 1, sigma)
    for i in range(min - 1, max):
        D[layer][i] = nd.pdf(i)
    D[layer] *= (1 / D[layer].sum())
    return D

def generate_D_victim(D, layer, graph_size_layer = None, mu = None, nodes = None):
    if graph_size_layer is not None:
        n = graph_size_layer
        D[layer] = np.zeros(len(D[n]))
        D[layer][0] = 1
        for i in range(1, len(D[n])):
            D[layer][i] = D[layer][i-1] - D[n][i-1]
        D[layer] /= mu
    else:
        D[layer] = np.ones(nodes)/nodes
    return D

def generate_line_graph(nodes):
    d = [1] * (nodes - 1)
    A = diags([d, d], [1, -1]).toarray()
    return A

def generate_agent_B(B, AG, layer=0):
    nodes = len(AG)
    B[layer] = np.zeros((nodes, nodes, nodes))
    for a in range(nodes):
        beta = np.zeros((nodes, 1))
        beta[a] = 1
        I = np.eye(nodes)
        B[layer][:, :, a] = AG*beta + I - I*AG[a,:]
    return B

def generate_independent_B(B, nodes, layers=[1]):
    for layer in layers:
        B[layer] = np.zeros((nodes, nodes, nodes))
        for a in range(nodes):
            B[layer][:, :, a] = np.eye(nodes)
        return B

def generate_A(A, AG):
    nodes = len(AG)
    A[0] = np.zeros((3, nodes, nodes))
    for v in range(nodes):
        for a in range(nodes):
            if a == v:
                A[0][0, a, v] = 1
            neighbours = np.where(AG[:, v] == 1)
            if a in neighbours[0]:
                A[0][1, a, v] = .5
            A[0][2, :] = np.ones(nodes) - A[0][0, :] - A[0][1, :]
    return A

if __name__ == "__main__":
    main()