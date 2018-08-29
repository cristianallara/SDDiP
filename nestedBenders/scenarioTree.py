import networkx as nx
import collections

# Build scenario tree
G = nx.DiGraph()

node_origin = ['O']
scenarios = ['L', 'R', 'H']
stages = range(1, 6)

# Create a set all the nodes in the scenario tree
N = ['F']
NP = ['F']
N1 = list(node_origin)
N2 = []
N3 = []
N4 = []
N5 = []
# N6 = []
# N7 = []
# N8 = []
# N9 = []
# N10 = []

PN = collections.OrderedDict()  # parental nodes
G.add_edge('F', 'O')
PN['O'] = ['F']
for s2 in scenarios:
    n2 = 'O' + str(s2)
    N2.append(n2)
    G.add_edge('O', n2)
    PN[n2] = ['O']
    for s3 in scenarios:
        n3 = 'O' + str(s2) + str(s3)
        N3.append(n3)
        G.add_edge(n2, n3)
        PN[n3] = [n2]
        for s4 in scenarios:
            n4 = 'O' + str(s2) + str(s3) + str(s4)
            N4.append(n4)
            G.add_edge(n3, n4)
            PN[n4] = [n3]
            for s5 in scenarios:
                n5 = 'O' + str(s2) + str(s3) + str(s4) + str(s5)
                N5.append(n5)
                G.add_edge(n4, n5)
                PN[n5] = [n4]
                # for s6 in scenarios:
                #     n6 = 'O' + str(s2) + str(s3) + str(s4) + str(s5) + str(s6)
                #     N6.append(n6)
                #     G.add_edge(n5, n6)
                #     PN[n6] = [n5]
                #     for s7 in scenarios:
                #         n7 = 'O' + str(s2) + str(s3) + str(s4) + str(s5) + str(s6) + str(s7)
                #         N7.append(n7)
                #         G.add_edge(n6, n7)
                #         PN[n7] = [n6]
                #         for s8 in scenarios:
                #             n8 = 'O' + str(s2) + str(s3) + str(s4) + str(s5) + str(s6) \
                #                  + str(s7) + str(s8)
                #             N8.append(n8)
                #             G.add_edge(n7, n8)
                #             PN[n8] = [n7]
                #             for s9 in scenarios:
                #                 n9 = 'O' + str(s2) + str(s3) + str(s4) + \
                #                      str(s5) + str(s6) + str(s7) + str(s8) + str(s9)
                #                 N9.append(n9)
                #                 G.add_edge(n8, n9)
                #                 PN[n9] = [n8]
                #                 for s10 in scenarios:
                #                     n10 = 'O' + str(s2) + str(s3) + str(s4) \
                #                           + str(s5) + str(s6) + str(s7) + str(s8) + str(s9) + str(s10)
                #                     N10.append(n10)
                #                     G.add_edge(n9, n10)
                #                     PN[n10] = [n9]

N.extend(N1)
N.extend(N2)
N.extend(N3)
N.extend(N4)
N.extend(N5)
# N.extend(N6)
# N.extend(N7)
# N.extend(N8)
# N.extend(N9)
# N.extend(N10)

# add nodes to the tree
G.add_nodes_from(N)
E = G.edges()
# print(E)
# print(PN)

N_stage = collections.OrderedDict()
N_stage[1] = N1
N_stage[2] = N2
N_stage[3] = N3
N_stage[4] = N4
N_stage[5] = N5
# N_stage[6] = N6
# N_stage[7] = N7
# N_stage[8] = N8
# N_stage[9] = N9
# N_stage[10] = N10

print('number of scenarios:', len(N_stage[10]))
# get weights based on children nodes
single_prob = {}
single_prob['L'] = 0.25
single_prob['R'] = 0.5
single_prob['H'] = 0.25

prob = collections.OrderedDict()
for n in N1:
    prob[n] = 1
for n in N2:
    m = n[1:2]
    prob[n] = single_prob[m]
for n in N3:
    m = n[1:2]
    o = n[2:3]
    prob[n] = single_prob[m] * single_prob[o]
for n in N4:
    m = n[1:2]
    o = n[2:3]
    l = n[3:4]
    prob[n] = single_prob[m] * single_prob[o] * single_prob[l]
for n in N5:
    m = n[1:2]
    o = n[2:3]
    l = n[3:4]
    q = n[4:5]
    prob[n] = single_prob[m] * single_prob[o] * single_prob[l] * single_prob[q]
# for n in N6:
#     m = n[1:2]
#     o = n[2:3]
#     l = n[3:4]
#     q = n[4:5]
#     p = n[5:6]
#     prob[n] = single_prob[m] * single_prob[o] * single_prob[l] * single_prob[q] * single_prob[p]
# for n in N7:
#     m = n[1:2]
#     o = n[2:3]
#     l = n[3:4]
#     q = n[4:5]
#     p = n[5:6]
#     r = n[6:7]
#     prob[n] = single_prob[m] * single_prob[o] * single_prob[l] * single_prob[q] \
#               * single_prob[p] * single_prob[r]
# for n in N8:
#     m = n[1:2]
#     o = n[2:3]
#     l = n[3:4]
#     q = n[4:5]
#     p = n[5:6]
#     r = n[6:7]
#     s = n[7:8]
#     prob[n] = single_prob[m] * single_prob[o] * single_prob[l] * single_prob[q] \
#               * single_prob[p] * single_prob[r] * single_prob[s]
# for n in N9:
#     m = n[1:2]
#     o = n[2:3]
#     l = n[3:4]
#     q = n[4:5]
#     p = n[5:6]
#     r = n[6:7]
#     s = n[7:8]
#     u = n[8:9]
#     prob[n] = single_prob[m] * single_prob[o] * single_prob[l] * single_prob[q] \
#               * single_prob[p] * single_prob[r] * single_prob[s] * single_prob[u]
# for n in N10:
#     m = n[1:2]
#     o = n[2:3]
#     l = n[3:4]
#     q = n[4:5]
#     p = n[5:6]
#     r = n[6:7]
#     s = n[7:8]
#     u = n[8:9]
#     v = n[9:10]
#     prob[n] = single_prob[m] * single_prob[o] * single_prob[l] * single_prob[q] \
#               * single_prob[p] * single_prob[r] * single_prob[s] * single_prob[u] \
#               * single_prob[v]

# Define scenarios:
sc_nodes = {}  # list(N_stage[len(stages)])
for n in N_stage[len(stages)]:  # 15
    for pn in PN[n]:  # 14
        for ppn in PN[pn]:  # 13
            for pppn in PN[ppn]:  # 12
                for ppppn in PN[pppn]:  # 11
                    # for pn1 in PN[ppppn]:  # 10
                    #     for ppn1 in PN[pn1]:  # 9
                    #         for pppn1 in PN[ppn1]:  # 8
                    #             for ppppn1 in PN[pppn1]:  # 7
                    #                 for pn2 in PN[ppppn1]:  # 6
                                        sc_nodes[n] = [
                                            # pn2, ppppn1, pppn1, ppn1, pn1,
                                            ppppn, pppn, ppn, pn, n]

# print(sc_nodes)
sc = list(sc_nodes.keys())
print('finished creating scenario tree')

# print(sc)
# CN = collections.OrderedDict() #children nodes
# for n_ in N:
#     if n_ not in N_stage[5]:
#         CN_list = []
#         for n in G.adj[n_]:
#             CN_list.extend([n])
#         CN[n_] = CN_list
