__author__ = "Cristiana L. Lara"
# Stochastic Dual Dynamic Integer Programming (SDDiP) description at:
# https://link.springer.com/content/pdf/10.1007%2Fs10107-018-1249-5.pdf

# This algorithm scenario tree satisfies stage-wise independence

import time
import math
import random
from copy import deepcopy
import os.path
from pyomo.environ import *
import pymp
import csv

from scenarioTree import create_scenario_tree
import readData
import optBlocks as b
from forward import forward_pass
from backward_SDDiP import backward_pass

# import pyomo.common.timing as timing
# timing.report_timing()

# ######################################################################################################################
# USER-DEFINED PARAMS

# Define case-study
curPath = os.path.abspath(os.path.curdir)
# filepath = os.path.join(curPath, 'data/GTEP_data_10years.db')
filepath = os.path.join(curPath, 'data/GTEPdata_5years.db')
n_stages = 5  # number od stages in the scenario tree
stages = range(1, n_stages + 1)
scenarios = ['L', 'M', 'H']
single_prob = {'L': 1 / 3, 'M': 1 / 3, 'H': 1 / 3}

# time_periods = 10
time_periods = 5
set_time_periods = range(1, time_periods + 1)
# t_per_stage = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8], 5: [9, 10]}
t_per_stage = {1: [1], 2: [2], 3: [3], 4: [4], 5: [5]}

# Define parameters of the decomposition
max_iter = 10
opt_tol = 5  # %
ns = 10  # Number of scenarios solved per Forward/Backward Pass per process
# NOTE: ns should be between 1 and len(n_stage[n_stages])
z_alpha_2 = 1.96  # 95% confidence level

# Parallel parameters
NumProcesses = 3

# ######################################################################################################################

# create scenarios and input data
nodes, n_stage, parent_node, children_node, prob, sc_nodes = create_scenario_tree(stages, scenarios, single_prob)
readData.read_data(filepath, stages, n_stage, t_per_stage)
sc_headers = list(sc_nodes.keys())

# separate nodes by processes
scenarios_by_processid = {}
for i in range(NumProcesses):
    start = int(len(sc_nodes) * i / float(NumProcesses))
    stop = int(len(sc_nodes) * (i + 1) / float(NumProcesses))
    scenarios_by_processid[i] = sc_headers[start:stop]
# print(scenarios_by_processid)

nodes_by_processid = {}
for pid, scenarios in scenarios_by_processid.items():
    nodes_by_processid[pid] = set()
    for scenario in scenarios:
        nodes_by_processid[pid].update(sc_nodes[scenario])
# print(nodes_by_processid)

n_stage_processid = {}
for pid in scenarios_by_processid.keys():
    n_stage_processid[pid] = {stage: [] for stage in stages}
    for stage in stages:
        for node in n_stage[stage]:
            if node in nodes_by_processid[pid]:
                n_stage_processid[pid][stage].append(node)
# print(n_stage_processid.keys())

# Split uncertain parameter by processes
L_max_scenario_pid = {}
for pid in scenarios_by_processid.keys():
    L_max_scenario_pid[pid] = {(t, stage, node): readData.L_max_s[t, stage, node] for stage in stages
                               for node in n_stage[stage] if node in nodes_by_processid[pid]
                               for t in t_per_stage[stage]}
# print(L_max_scenario_pid)

# Shared data among processes
ngo_rn_par_k = pymp.shared.dict()
ngo_th_par_k = pymp.shared.dict()
cost_scenario_forward = pymp.shared.dict()
mltp_o_rn = pymp.shared.dict()
mltp_o_th = pymp.shared.dict()
cost_backward = pymp.shared.dict()
barrier_flag = pymp.shared.list([0] * (NumProcesses + 1))
if_converged = pymp.shared.dict()

# Map stage by time_period
stage_per_t = {t: k for k, v in t_per_stage.items() for t in v}


# print(stage_per_t)


class Barrier:
    def __init__(self, n):
        self.n = n
        self.count = pymp.shared.list([0])
        self.mutex = pymp.shared._MANAGER.Semaphore(1)
        self.barrier = pymp.shared._MANAGER.Semaphore(0)

    def wait(self):
        self.mutex.acquire()
        self.count[0] = self.count[0] + 1
        if self.count[0] == self.n:
            self.barrier.release()
        self.mutex.release()

        self.barrier.acquire()
        self.mutex.acquire()
        self.count[0] = self.count[0] - 1
        if self.count[0]:
            self.barrier.release()
        self.mutex.release()


solve_barrier = Barrier(NumProcesses)
calc_barrier = Barrier(NumProcesses)
break_barrier = Barrier(NumProcesses)

with pymp.Parallel(NumProcesses) as p:
    pid = p.thread_num

    # create blocks
    m = b.create_model(n_stages, time_periods, t_per_stage, max_iter, n_stage_processid[pid], nodes_by_processid[pid],
                       L_max_scenario_pid[pid], pid)
    start_time = time.time()

    # Decomposition Parameters
    m.ngo_rn_par = Param(m.rn_r, m.t_node, default=0, initialize=0, mutable=True)
    m.ngo_th_par = Param(m.th_r, m.t_node, default=0, initialize=0, mutable=True)
    m.cost = Param(m.n_stage, m.iter, default=0, initialize=0, mutable=True)

    # Parameters to compute upper and lower bounds
    cost_forward = {}
    mean = {}
    std_dev = {}
    cost_tot_forward = {}
    cost_UB = {}
    cost_LB = {}
    gap = {}
    scenarios_iter = {}

    # converting sets to lists:
    rn_r = list(m.rn_r)
    th_r = list(m.th_r)

    # request the dual variables for all (locally) defined blocks
    for b in m.Bl.values():
        b.dual = Suffix(direction=Suffix.IMPORT)

    # Add equality constraints
    for stage in m.stages:
        for n in n_stage_processid[pid][stage]:
            if stage != 1:
                t_prev = t_per_stage[stage - 1][-1]
                # print('stage', stage, 't_prev', t_prev)
                for (rn, r) in m.rn_r:
                    for pn in n_stage_processid[pid][stage - 1]:
                        if pn in parent_node[n]:
                            m.Bl[stage, n].link_equal1.add(expr=(m.Bl[stage, n].ngo_rn_prev[rn, r] ==
                                                                 m.ngo_rn_par[rn, r, t_prev, pn]))
                for (th, r) in m.th_r:
                    for pn in n_stage_processid[pid][stage - 1]:
                        if pn in parent_node[n]:
                            m.Bl[stage, n].link_equal2.add(expr=(m.Bl[stage, n].ngo_th_prev[th, r] ==
                                                                 m.ngo_th_par[th, r, t_prev, pn]))

    # Stochastic Dual Dynamic integer Programming Algorithm (SDDiP)
    for iter_ in m.iter:

        sample_index = deepcopy(random.sample(range(0, len(scenarios_by_processid[pid])), ns))
        sampled_scenarios = {}
        sampled_nodes = []
        sampled_nodes_stage = {}
        scenarios_iter[iter_] = {}
        for i in sample_index:
            n = scenarios_by_processid[pid][i]
            sampled_scenarios[n] = sc_nodes[n]
            scenarios_iter[iter_][n] = list(sampled_scenarios[n])
        for stage in stages:
            for n in list(sampled_scenarios.keys()):
                s_sc = sampled_scenarios[n][n_stages - stage]
                if s_sc not in sampled_nodes:
                    sampled_nodes.append(s_sc)
                sampled_nodes_stage[stage] = sampled_nodes
            sampled_nodes = []

        # Forward Pass
        for stage in m.stages:
            for n in sampled_nodes_stage[stage]:
                print("Forward Pass: ", "Stage", stage, "Current Node", n)

                ngo_rn, ngo_th, cost = forward_pass(m.Bl[stage, n], rn_r, th_r, t_per_stage[stage])

                for t in t_per_stage[stage]:
                    for (rn, r) in rn_r:
                        ngo_rn_par_k[rn, r, t, n, iter_] = ngo_rn[rn, r, t]
                        m.ngo_rn_par[rn, r, t, n] = ngo_rn[rn, r, t]
                    for (th, r) in th_r:
                        ngo_th_par_k[th, r, t, n, iter_] = ngo_th[th, r, t]
                        m.ngo_th_par[th, r, t, n] = ngo_th[th, r, t]
                cost_forward[stage, n, iter_] = cost
                print('cost', cost_forward[stage, n, iter_])

        # Compute cost per scenario solved inside of a process
        for s_sc in list(sampled_scenarios.keys()):
            cost_scenario_forward[s_sc, iter_] = 0
            for stage in m.stages:
                cost_scenario_forward[s_sc, iter_] += \
                    cost_forward[stage, sampled_scenarios[s_sc][len(m.stages) - stage], iter_]
        # print(cost_scenario_forward)

        print("thread %s completed forward pass" % p.thread_num)
        solve_barrier.wait()
        print("thread %s starting calculation phase" % p.thread_num)

        if p.thread_num == 0:
            # Compute statistical upper bound
            mean[iter_] = (1 / (ns * NumProcesses)) * sum(cost_scenario_forward[s_sc, iteration]
                                                          for (s_sc, iteration) in cost_scenario_forward.keys()
                                                          if iteration == iter_)
            print('mean', mean[iter_])
            std_dev[iter_] = math.sqrt((1 / (ns * NumProcesses - 1))
                                       * sum((cost_scenario_forward[s_sc, iteration] - mean[iter_]) ** 2
                                             for (s_sc, iteration) in cost_scenario_forward.keys() if
                                             iteration == iter_))
            print('std dev', std_dev[iter_])
            cost_tot_forward[iter_] = mean[iter_] + z_alpha_2 * (std_dev[iter_] / math.sqrt(ns * NumProcesses))
            if iter_ == 1:
                cost_UB[iter_] = cost_tot_forward[iter_]
            else:
                cost_UB[iter_] = min(cost_tot_forward[kk] for kk in m.iter if kk <= iter_
                                     if cost_tot_forward[kk] >= cost_LB[iter_ - 1])
            print(cost_UB)
            elapsed_time = time.time() - start_time
            print("CPU Time (s)", elapsed_time)

        print("thread %s completed calculation phase" % p.thread_num)
        calc_barrier.wait()

        # Backward Pass
        m.k.add(iter_)

        sampled_nodes_globally = [(t, n) for (rn, r, t, n, it) in ngo_rn_par_k.keys() if it == iter_]
        sampled_nodes_globally_stage = {stage: [] for stage in stages}
        for key in sampled_nodes_globally:
            stage_ = stage_per_t[key[0]]
            n_ = key[1]
            if n_ not in sampled_nodes_globally_stage[stage_]:
                sampled_nodes_globally_stage[stage_].append(n_)
        # print(sampled_nodes_globally_stage)

        for stage in reversed(list(m.stages)):
            for n in sampled_nodes_stage[stage]:
                print("Backward Pass:", "Stage", stage, "Current Node", n)
                # print("Current thread", pp.thread_num)

                mltp_rn, mltp_th, cost = backward_pass(stage, m.Bl[stage, n], n_stages, rn_r, th_r)

                cost_backward[stage, n, iter_] = cost
                print('cost', stage, n, cost_backward[stage, n, iter_])

                if stage != 1:
                    for (rn, r) in rn_r:
                        mltp_o_rn[rn, r, stage, n, iter_] = mltp_rn[rn, r]
                    for (th, r) in th_r:
                        mltp_o_th[th, r, stage, n, iter_] = mltp_th[th, r]

            solve_barrier.wait()

            if stage != 1:
                for pn in sampled_nodes_stage[stage - 1]:
                    t = t_per_stage[stage][0]
                    t_prev = t_per_stage[stage - 1][-1]
                    # add Benders cut for current iteration
                    m.Bl[stage - 1, pn].fut_cost.add(expr=(m.Bl[stage - 1, pn].alphafut >=
                                                           (1 / len(sampled_nodes_globally_stage[stage])) *
                                                           sum(cost_backward[stage, n_, iter_]
                                                               + sum(mltp_o_rn[rn, r, stage, n_, iter_] *
                                                                     (ngo_rn_par_k[rn, r, t_prev, pn, iter_] -
                                                                      m.Bl[stage - 1, pn].ngo_rn[rn, r, t_prev])
                                                                     for rn, r in m.rn_r)
                                                               + sum(mltp_o_th[th, r, stage, n_, iter_] *
                                                                     (ngo_th_par_k[th, r, t_prev, pn, iter_] -
                                                                      m.Bl[stage - 1, pn].ngo_th[th, r, t_prev])
                                                                     for th, r in m.th_r)
                                                               for n_ in sampled_nodes_globally_stage[stage])))

                # m.Bl[t, n].fut_cost.pprint()

            calc_barrier.wait()
            print("thread %s completed calculation phase for stage" % p.thread_num, stage)

        solve_barrier.wait()
        print("thread %s completed backward pass" % p.thread_num)

        if p.thread_num == 0:
            # Compute lower bound
            cost_LB[iter_] = cost_backward[1, 'O', iter_]
            print(cost_LB)
            # Compute optimality gap
            gap[iter_] = (cost_UB[iter_] - cost_LB[iter_]) / cost_UB[iter_] * 100
            print("gap: ", gap[iter_])

            if gap[iter_] <= opt_tol:
                with open('results.csv', mode='w') as results_file:
                    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for stage in stages:
                        for n in sampled_nodes_globally_stage[stage]:
                            for t in t_per_stage[stage]:
                                for (rn, r) in rn_r:
                                    if ngo_rn_par_k[rn, r, t, n, iter_] != 0:
                                        results_writer.writerow([rn, r, t, n, ngo_rn_par_k[rn, r, t, n, iter_]])
                                for (th, r) in th_r:
                                    if ngo_th_par_k[th, r, t, n, iter_] != 0:
                                        results_writer.writerow([th, r, t, n, ngo_th_par_k[th, r, t, n, iter_]])
                if_converged[iter_] = True
            else:
                if_converged[iter_] = False

            elapsed_time = time.time() - start_time
            print("CPU Time (s)", elapsed_time)

        print("thread %s completed calculation phase" % p.thread_num)
        calc_barrier.wait()

        if if_converged[iter_]:
            last_iter = iter_
            break
    print("thread %s completed algorithm" % p.thread_num)
    break_barrier.wait()

    if p.thread_num == 0:
        elapsed_time = time.time() - start_time
        print("Upper Bound", cost_UB[last_iter])
        print("Lower Bound", cost_LB[last_iter])
        print("Optimality gap (%)", gap[last_iter])
        print("CPU Time (s)", elapsed_time)

    print("thread %s completed calculation phase" % p.thread_num)
    calc_barrier.wait()
