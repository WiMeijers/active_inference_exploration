## Some (adjusted) functions from the PYMDP package
from pymdp.maths import softmax, spm_log_single
from pymdp import control
import numpy as np
import copy

def update_posterior_policies(qs, A, B, C, policies, use_utility=True, 
                                use_states_info_gain=True, gamma=16.0):
    n_policies = len(policies)
    G = np.zeros(n_policies)
    SIG = np.zeros(n_policies)
    UT = np.zeros(n_policies)
    q_pi = np.zeros((n_policies, 1))

    for idx, policy in enumerate(policies):
        qs_pi = control.get_expected_states(qs, B, policy)
        qo_pi = control.get_expected_obs(qs_pi, A)
        if use_utility:
            UT_c = control.calc_expected_utility(qo_pi, C)
            UT[idx] = UT_c
            G[idx] += UT_c
        if use_states_info_gain:
            SIG_c = control.calc_states_info_gain(A, qs_pi)
            SIG[idx] += SIG_c
            G[idx] += SIG_c
    q_pi = softmax(G * gamma)    
    return G, UT, SIG, q_pi

def generate_agent_B_short(B, AG, layer=0):
    nodes = len(AG)
    degrees = np.sum(AG, axis=0)
    actions = np.max(degrees)
    B[layer] = np.zeros((nodes, nodes, actions))
    for a in range(actions):
        for v in range(nodes):
            vector = AG[:, v]
            entries = np.where(vector == 1)[0]
            if len(entries) > a:
                B[layer][entries[a], v, a] = 1
            else:
                B[layer][v, v, a] = 1
    return B