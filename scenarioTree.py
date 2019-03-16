import networkx as nx
import collections


def create_scenario_tree(stages, scenarios, single_prob):

    # Build scenario tree
    g = nx.DiGraph()

    # create nodes by stage
    nodes = ['F']  # fake node such that origin also has a parent
    node_origin = ['O']
    n_stage = collections.OrderedDict()
    n_stage[1] = list(node_origin)
    for i in range(2, len(stages)+1):
        n_stage[i] = []

    # initialize parent nodes
    parent_node = collections.OrderedDict()
    parent_node['O'] = 'F'
    g.add_edge('F', 'O')

    # initialize children nodes
    children_node = collections.OrderedDict()

    # initialize transition probability
    prob = collections.OrderedDict()
    prob['O'] = 1
    for i in stages[1:]:
        for j in n_stage[i-1]:
            children_node[j] = []
            for s in scenarios:
                node = j + str(s)
                n_stage[i].append(node)
                parent_node[node] = j
                children_node[j].append(node)
                prob[node] = prob[j] * single_prob[s]
                g.add_edge(j, node)
    for i in stages:
        nodes.extend(n_stage[i])
    g.add_nodes_from(nodes)

    sc_nodes = collections.OrderedDict()
    for n in n_stage[stages[-1]]:
        sc_nodes[n] = [n]
        for s in reversed(stages):
            pn = parent_node[sc_nodes[n][-1]]
            sc_nodes[n].append(pn)

    # print(sc_nodes)
    print('finished creating scenario tree')
    print('number of scenarios:', len(n_stage[stages[-1]]))

    return nodes, n_stage, parent_node, children_node, prob, sc_nodes

