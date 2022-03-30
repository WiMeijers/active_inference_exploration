import numpy as np
from pymdp import control, inference, utils
from ai_functions import update_posterior_policies
import matplotlib.pyplot as plt

## Basic example on the use of active inference over graphs


def main():
    ## Creating the Generative Model
    sf = 2  # number of state factors
    obs = 1  # number of observations

    A = utils.obj_array(obs)
    B = utils.obj_array(sf)
    C = utils.obj_array(obs)
    D = utils.obj_array(sf)

    AG = np.array(([0, 1, 0, 0, 0],  # graph adjecancy matrix
                [1, 0, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 0, 1],
                [0, 0, 0, 1, 0]))

    nodes = len(AG)

    A = generate_A(A, AG)
    B = generate_agent_B(B, AG, layer=0)
    B = generate_victim_B(B, layer=1)
    C[0] = np.array([1, 0])
    D[0] = np.array([1, 0, 0, 0, 0])
    D[1] = np.ones(nodes) / nodes

    ## Inference over first action 
    victim_loc = 4
    agent_loc = np.where(D[0] == 1)
    obs = A[0][:, agent_loc, victim_loc]
    D = inference.update_posterior_states(A, obs, D)
    D[0] = np.array([0, 1, 0, 0, 0])
    agent_loc = np.where(D[0] == 1)
    obs = A[0][:, agent_loc, victim_loc]
    D = inference.update_posterior_states(A, obs, D)
    
    ## EFE calculations for actions 2 and 3
    policies = []
    policies += [np.array([[2, 3]])]
    policies += [np.array([[3, 2]])]
    policies += [np.array([[3, 4]])]
    G, UT, SIG, _ = update_posterior_policies(D, A, B, C, policies)

def generate_agent_B(B, AG, layer=0):
    nodes = len(AG)
    B[layer] = np.zeros((nodes, nodes, nodes))
    for a in range(nodes):
        beta = np.zeros((nodes, 1))
        beta[a] = 1
        I = np.eye(nodes)
        B[layer][:, :, a] = AG*beta + I - I*AG[a,:]
    return B

def generate_victim_B(B,  layer=1):
    B[layer] = np.zeros((nodes, nodes, nodes))
    for a in range(nodes):
        B[layer][:, :, a] = np.eye(nodes)
    return B

def generate_A(A, AG):
    nodes = len(AG)
    A[0] = np.zeros((2, nodes, nodes))
    for v in range(nodes):
        for a in range(nodes):
            if a == v:
                A[0][0, a, v] = 1
            neighbours = np.where(AG[:, v] == 1)
            if a in neighbours[0]:
                A[0][0, a, v] = .5
            A[0][1, :] = np.ones(nodes) - A[0][0, :]
    return A

if __name__ == "__main__":
    main()