import numpy as np
from pymdp import control, inference, utils
from ai_functions import update_posterior_policies, generate_agent_B_short
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
    B = generate_agent_B_short(B, AG)  # only as an example of the shorter formulation

    A = generate_A(A, AG)
    B = generate_agent_B(B, AG, layer=0)
    B = generate_victim_B(B, nodes, layer=1)
    C[0] = np.array([1, 0, 0])
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
    
    ## EFE calculations for policies of length 1
    policies = []
    policies += [np.array([[0, 0]])]
    policies += [np.array([[2, 2]])]
    policies += [np.array([[3, 3]])]
    G, UT, SIG, _ = update_posterior_policies(D, A, B, C, policies)
    print('The results for policies of length 1:')
    print('G: \n', G)
    print('UT: \n', UT)
    print('SIG: \n', SIG, '\n')
    plot_expectations(D, A, B, C, policies)

    ## EFE calculations for policies of length 2
    policies = []
    policies += [np.array([[2, 2], [3, 3]])]
    policies += [np.array([[3, 3], [2, 2]])]
    policies += [np.array([[3, 3], [4, 4]])]
    G, UT, SIG, _ = update_posterior_policies(D, A, B, C, policies)
    print('The results for policies of length 2:')
    print('G: \n', G)
    print('UT: \n', UT)
    print('SIG: \n', SIG, '\n')

    ## EFE calculations for policies of length 2
    policies = []
    policies += [np.array([[2, 2], [3, 3], [4, 4]])]
    policies += [np.array([[3, 3], [2, 2], [3, 3]])]
    policies += [np.array([[3, 3], [2, 2], [1, 1]])]
    policies += [np.array([[3, 3], [4, 4], [3, 3]])]
    G, UT, SIG, _ = update_posterior_policies(D, A, B, C, policies)
    print('The results for policies of length 3:')
    print('G: \n', G)
    print('UT: \n', UT)
    print('SIG: \n', SIG, '\n')

def generate_agent_B(B, AG, layer=0):
    nodes = len(AG)
    B[layer] = np.zeros((nodes, nodes, nodes))
    for a in range(nodes):
        beta = np.zeros((nodes, 1))
        beta[a] = 1
        I = np.eye(nodes)
        B[layer][:, :, a] = AG*beta + I - I*AG[a,:]
    return B

def generate_victim_B(B, nodes, layer=1):
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

def plot_expectations(D, A, B, C, policies):
    s_agent = []
    s_victim = []
    o_victim = []
    for policy in policies:
        qs_pi = control.get_expected_states(D, B, policy)
        obs = control.get_expected_obs(qs_pi, A)
        o_victim += [obs[0][0]]
    xo = np.linspace(0, 2, 3)
    dx = .25/len(o_victim)
    w = dx * .8
    plt.bar(xo - dx, o_victim[0], width = w)
    plt.bar(xo, o_victim[1], width = w)
    plt.bar(xo + dx, o_victim[2], width = w)
    plt.legend(['action 0', 'action 2', 'action 3'])
    plt.show()


if __name__ == "__main__":
    main()