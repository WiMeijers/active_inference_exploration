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