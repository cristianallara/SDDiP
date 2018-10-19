# Generation and Transmission Expansion
# IDAES project
# author: Cristiana L. Lara
# date: 02/07/2017

from pyomo.environ import *
import os.path
import math

from scenarioTree import create_scenario_tree
import readData

# Define case-study
curPath = os.path.abspath(os.path.curdir)
filepath = os.path.join(curPath, 'GTEPdata_5years.db')
time_periods = 5
stages = range(1, time_periods + 1)
scenarios = ['L', 'R', 'H']
single_prob = {'L': 1 / 3, 'R': 1 / 3, 'H': 1 / 3}

# create scenarios and input data
nodes, n_stage, parent_node, children_node, prob, sc_nodes = create_scenario_tree(stages, scenarios, single_prob)
sc_headers = list(sc_nodes.keys())
readData.read_data(filepath, stages, n_stage)

det = ConcreteModel()

# Declaration of sets
det.r = Set(initialize=['Northeast', 'West', 'Coastal', 'South', 'Panhandle'], ordered=True)
det.i = Set(initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'nuc-st-old', 'pv-old', 'wind-old',
                        'nuc-st-new', 'wind-new', 'pv-new', 'csp-new', 'coal-igcc-new', 'coal-igcc-ccs-new',
                        'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
det.th = Set(within=det.i, initialize=['nuc-st-old', 'nuc-st-new', 'coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new',
                                       'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new', 'ng-cc-ccs-new',
                                       'ng-ct-new'], ordered=True)
det.rn = Set(within=det.i, initialize=['pv-old', 'pv-new', 'csp-new', 'wind-old', 'wind-new'], ordered=True)
det.co = Set(within=det.th, initialize=['coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new'], ordered=True)
det.ng = Set(within=det.th, initialize=['ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new', 'ng-cc-ccs-new',
                                        'ng-ct-new'], ordered=True)
det.nu = Set(within=det.th, initialize=['nuc-st-old', 'nuc-st-new'], ordered=True)
det.pv = Set(within=det.rn, initialize=['pv-old', 'pv-new'], ordered=True)
det.csp = Set(within=det.rn, initialize=['csp-new'], ordered=True)
det.wi = Set(within=det.rn, initialize=['wind-old', 'wind-new'], ordered=True)
det.old = Set(within=det.i, initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'nuc-st-old', 'pv-old',
                                        'wind-old'], ordered=True)
det.new = Set(within=det.i, initialize=['nuc-st-new', 'wind-new', 'pv-new', 'csp-new', 'coal-igcc-new',
                                        'coal-igcc-ccs-new', 'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
det.rold = Set(within=det.old, initialize=['pv-old', 'wind-old'], ordered=True)
det.rnew = Set(within=det.new, initialize=['wind-new', 'pv-new', 'csp-new'], ordered=True)
det.told = Set(within=det.old, initialize=['nuc-st-old', 'coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old'],
               ordered=True)
det.tnew = Set(within=det.new, initialize=['nuc-st-new', 'coal-igcc-new', 'coal-igcc-ccs-new', 'ng-cc-new',
                                           'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)

det.d = Set(initialize=['spring', 'summer', 'fall', 'winter'], ordered=True)
det.s = Set(initialize=['1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00',
                        '22:00', '23:00', '24:00'], ordered=True)

det.t = RangeSet(time_periods)

t_stage = [(t, n) for t in det.t for n in n_stage[t]]
det.n_stage = Set(initialize=t_stage)

# Import parameters
det.Ng_old = Param(det.i, det.r, default=0, initialize=readData.Ng_old)


def i_r_filter(m, i, r):
    return i in m.new or (i in m.old and m.Ng_old[i, r] != 0)


det.i_r = Set(initialize=det.i * det.r, filter=i_r_filter, ordered=True)


def rn_r_filter(m, rn, r):
    return rn in m.rnew or (rn in m.rold and m.Ng_old[rn, r] != 0)


det.rn_r = Set(initialize=det.rn * det.r, filter=rn_r_filter, ordered=True)


def th_r_filter(m, th, r):
    return th in m.tnew or (th in m.told and m.Ng_old[th, r] != 0)


det.th_r = Set(initialize=det.th * det.r, filter=th_r_filter, ordered=True)

det.L = Param(det.r, det.t, det.d, det.s, default=0, initialize=readData.L)
det.n_d = Param(det.d, default=0, initialize=readData.n_ss)
det.L_max = Param(det.n_stage, default=0, initialize=readData.L_max_s)
# m.L_max = Param(m.t, default=0, initialize=L_max)
det.cf = Param(det.i, det.r, det.t, det.d, det.s, default=0, initialize=readData.cf)
det.Qg_np = Param(det.i_r, default=0, initialize=readData.Qg_np)
det.Ng_max = Param(det.i_r, default=0, initialize=readData.Ng_max)
det.Qinst_UB = Param(det.i, det.t, default=0, initialize=readData.Qinst_UB)
det.LT = Param(det.i, initialize=readData.LT, default=0)
det.Tremain = Param(det.t, default=0, initialize=readData.Tremain)
det.Ng_r = Param(det.old, det.r, det.t, default=0, initialize=readData.Ng_r)
det.q_v = Param(det.i, default=0, initialize=readData.q_v)
det.Pg_min = Param(det.i, default=0, initialize=readData.Pg_min)
det.Ru_max = Param(det.i, default=0, initialize=readData.Ru_max)
det.Rd_max = Param(det.i, default=0, initialize=readData.Rd_max)
det.f_start = Param(det.i, default=0, initialize=readData.f_start)
det.C_start = Param(det.i, default=0, initialize=readData.C_start)
det.frac_spin = Param(det.i, default=0, initialize=readData.frac_spin)
det.frac_Qstart = Param(det.i, default=0, initialize=readData.frac_Qstart)
det.t_loss = Param(det.r, det.r, default=0, initialize=readData.t_loss)
det.t_up = Param(det.r, det.r, default=0, initialize=readData.t_up)
det.dist = Param(det.r, det.r, default=0, initialize=readData.dist)
det.if_ = Param(det.t, default=0, initialize=readData.if_)
det.ED = Param(det.t, default=0, initialize=readData.ED)
det.Rmin = Param(det.t, default=0, initialize=readData.Rmin)
det.hr = Param(det.i_r, default=0, initialize=readData.hr)
det.P_fuel = Param(det.i, det.t, default=0, initialize=readData.P_fuel)
# m.P_fuel = Param(m.i, m.n_stage, default=0, initialize=P_fuel)
det.EF_CO2 = Param(det.i, default=0, initialize=readData.EF_CO2)
det.FOC = Param(det.i, det.t, default=0, initialize=readData.FOC)
det.VOC = Param(det.i, det.t, default=0, initialize=readData.VOC)
det.CCm = Param(det.i, default=0, initialize=readData.CCm)
det.DIC = Param(det.i, det.t, default=0, initialize=readData.DIC)
det.LEC = Param(det.i, default=0, initialize=readData.LEC)
det.PEN = Param(det.t, default=0, initialize=readData.PEN)
det.PENc = Param(default=0, initialize=readData.PENc)
det.tx_CO2 = Param(det.t, default=0, initialize=readData.tx_CO2)
det.RES_min = Param(det.t, default=0, initialize=readData.RES_min)
det.hs = Param(initialize=readData.hs, default=0)
det.ir = Param(initialize=readData.ir, default=0)
det.prob = Param(nodes, initialize=prob)


# Block of Equations per time period
def planning_block_rule(b, t, n):
    def bound_P(_b, i, r, d, s):
        if i in det.old:
            return 0, det.Qg_np[i, r] * det.Ng_old[i, r]
        else:
            return 0, det.Qg_np[i, r] * det.Ng_max[i, r]

    def bound_Pflow(_b, r, r_, d, s):
        if r_ != r:
            return 0, readData.t_up[r, r_]
        else:
            return 0, 0

    def bound_o_rn(_b, rn, r):
        if rn in det.rold:
            return 0, det.Ng_old[rn, r]
        else:
            return 0, det.Ng_max[rn, r]

    def bound_b_rn(_b, rn, r):
        if rn in det.rold:
            return 0, 0
        else:
            return 0, math.floor(det.Qinst_UB[rn, t] / det.Qg_np[rn, r])

    def bound_r_rn(_b, rn, r):
        if rn in det.rold:
            return 0, det.Ng_old[rn, r]
        else:
            return 0, det.Ng_max[rn, r]

    def bound_e_rn(_b, rn, r):
        if rn in det.rold:
            return 0, det.Ng_r[rn, r, t]
        else:
            return 0, det.Ng_max[rn, r]

    def bound_o_th(_b, th, r):
        if th in det.told:
            return 0, det.Ng_old[th, r]
        else:
            return 0, det.Ng_max[th, r]

    def bound_b_th(_b, th, r):
        if th in det.told:
            return 0, 0
        else:
            return 0, math.floor(det.Qinst_UB[th, t] / det.Qg_np[th, r])

    def bound_r_th(_b, th, r):
        if th in det.told:
            return 0, det.Ng_old[th, r]
        else:
            return 0, det.Ng_max[th, r]

    def bound_e_th(_b, th, r):
        if th in det.told:
            return 0, det.Ng_r[th, r, t]
        else:
            return 0, det.Ng_max[th, r]

    def bound_UC(_b, th, r, d, s):
        if th in det.told:
            return 0, det.Ng_old[th, r]
        else:
            return 0, det.Ng_max[th, r]

    b.P = Var(det.i_r, det.d, det.s, within=NonNegativeReals, bounds=bound_P)
    b.cu = Var(det.r, det.d, det.s, within=NonNegativeReals)
    b.RES_def = Var(within=NonNegativeReals)
    b.P_flow = Var(det.r, det.r, det.d, det.s, bounds=bound_Pflow)
    b.Q_spin = Var(det.th_r, det.d, det.s, within=NonNegativeReals)
    b.Q_Qstart = Var(det.th_r, det.d, det.s, within=NonNegativeReals)
    b.ngr_rn = Var(det.rn_r, bounds=bound_r_rn, domain=NonNegativeReals)
    for rnew, r in det.rn_r:
        if rnew in det.rnew:
            if t == 1:
                b.ngr_rn[rnew, r].fix(0.0)
            else:
                b.ngr_rn[rnew, r].unfix()

    b.nge_rn = Var(det.rn_r, bounds=bound_e_rn, domain=NonNegativeReals)
    for rnew, r in det.rn_r:
        if rnew in det.rnew:
            if t == 1:
                b.nge_rn[rnew, r].fix(0.0)
            else:
                b.nge_rn[rnew, r].unfix()

    b.ngr_th = Var(det.th_r, bounds=bound_r_th, domain=NonNegativeIntegers)
    for tnew, r in det.th_r:
        if tnew in det.tnew:
            if t == 1:
                b.ngr_th[tnew, r].fix(0.0)
            else:
                b.ngr_th[tnew, r].unfix()

    b.nge_th = Var(det.th_r, bounds=bound_e_th, domain=NonNegativeIntegers)
    for tnew, r in det.th_r:
        if tnew in det.tnew:
            if t == 1:
                b.nge_th[tnew, r].fix(0.0)
            else:
                b.nge_th[tnew, r].unfix()
    b.u = Var(det.th_r, det.d, det.s, bounds=bound_UC, domain=NonNegativeIntegers)
    b.su = Var(det.th_r, det.d, det.s, bounds=bound_UC, domain=NonNegativeIntegers)
    b.sd = Var(det.th_r, det.d, det.s, bounds=bound_UC, domain=NonNegativeIntegers)
    for th, r, d, s in det.th_r * det.d * det.s:
        if det.s.ord(s) == 1:
            b.sd[th, r, d, s].fix(0.0)
        else:
            b.sd[th, r, d, s].unfix()
    b.alphafut = Var(within=Reals, domain=NonNegativeReals)

    b.ngo_rn = Var(det.rn_r, bounds=bound_o_rn, domain=NonNegativeReals)
    b.ngb_rn = Var(det.rn_r, bounds=bound_b_rn, domain=NonNegativeReals)
    for rold, r in det.rn_r:
        if rold in det.rold:
            b.ngb_rn[rold, r].fix(0.0)

    b.ngo_th = Var(det.th_r, bounds=bound_o_th, domain=NonNegativeIntegers)
    b.ngb_th = Var(det.th_r, bounds=bound_b_th, domain=NonNegativeIntegers)
    for told, r in det.th_r:
        if told in det.told:
            b.ngb_th[told, r].fix(0.0)

    b.ngo_rn_prev = Var(det.rn_r, bounds=bound_o_rn, domain=NonNegativeReals)

    b.ngo_th_prev = Var(det.th_r, bounds=bound_o_th, domain=NonNegativeReals)

    def min_RN_req(_b):
        return sum(det.n_d[d] * det.hs * sum(_b.P[rn, r, d, s] - _b.cu[r, d, s] for rn, r in det.i_r if rn in det.rn) \
                   for d in det.d for s in det.s) \
               + _b.RES_def >= det.RES_min[t] * det.ED[t]

    b.min_RN_req = Constraint(rule=min_RN_req)

    def min_reserve(_b):
        return sum(det.Qg_np[rn, r] * _b.ngo_rn[rn, r] * det.q_v[rn] for rn, r in det.i_r if rn in det.rn) \
               + sum(det.Qg_np[th, r] * _b.ngo_th[th, r] for th, r in det.i_r if th in det.th) \
               >= (1 + det.Rmin[t]) * det.L_max[t, n]

    b.min_reserve = Constraint(rule=min_reserve)

    def inst_RN_UB(_b, rnew):
        return sum(_b.ngb_rn[rnew, r] for r in det.r) \
               <= det.Qinst_UB[rnew, t] / sum(det.Qg_np[rnew, r] / len(det.r) for r in det.r)

    b.inst_RN_UB = Constraint(det.rnew, rule=inst_RN_UB)

    def inst_TH_UB(_b, tnew):
        return sum(_b.ngb_th[tnew, r] for r in det.r) \
               <= det.Qinst_UB[tnew, t] / sum(det.Qg_np[tnew, r] / len(det.r) for r in det.r)

    b.inst_TH_UB = Constraint(det.tnew, rule=inst_TH_UB)

    def en_bal(_b, r, d, s):
        return sum(_b.P[i, r, d, s] for i in det.i if (i, r) in det.i_r) \
               + sum(_b.P_flow[r_, r, d, s] * (1 - det.t_loss[r, r_] * det.dist[r, r_]) - \
                     _b.P_flow[r, r_, d, s] for r_ in det.r if r_ != r) \
               == det.L[r, t, d, s] + _b.cu[r, d, s]

    b.en_bal = Constraint(det.r, det.d, det.s, rule=en_bal)

    def capfactor(_b, rn, r, d, s):
        return _b.P[rn, r, d, s] == det.Qg_np[rn, r] * det.cf[rn, r, t, d, s] \
               * _b.ngo_rn[rn, r]

    b.capfactor = Constraint(det.rn_r, det.d, det.s, rule=capfactor)

    def min_output(_b, th, r, d, s):
        return _b.u[th, r, d, s] * det.Pg_min[th] * det.Qg_np[th, r] <= _b.P[th, r, d, s]

    b.min_output = Constraint(det.th_r, det.d, det.s, rule=min_output)

    def max_output(_b, th, r, d, s):
        return _b.u[th, r, d, s] * det.Qg_np[th, r] >= _b.P[th, r, d, s] \
               + _b.Q_spin[th, r, d, s]

    b.max_output = Constraint(det.th_r, det.d, det.s, rule=max_output)

    def unit_commit1(_b, th, r, d, s, s_):
        if det.s.ord(s_) == det.s.ord(s) - 1:
            return _b.u[th, r, d, s] == _b.u[th, r, d, s_] + _b.su[th, r, d, s] \
                   - _b.sd[th, r, d, s]
        return Constraint.Skip

    b.unit_commit1 = Constraint(det.th_r, det.d, det.s, det.s, rule=unit_commit1)

    def ramp_up(_b, th, r, d, s, s_):
        if (th, r) in det.i_r and th in det.th and (det.s.ord(s_) == det.s.ord(s) - 1):
            return _b.P[th, r, d, s] - _b.P[th, r, d, s_] <= det.Ru_max[th] * det.hs * \
                   det.Qg_np[th, r] * (_b.u[th, r, d, s] - _b.su[th, r, d, s]) + \
                   max(det.Pg_min[th], det.Ru_max[th] * det.hs) * det.Qg_np[th, r] * _b.su[th, r, d, s]
        return Constraint.Skip

    b.ramp_up = Constraint(det.th, det.r, det.d, det.s, det.s, rule=ramp_up)

    def ramp_down(_b, th, r, d, s, s_):
        if det.s.ord(s_) == det.s.ord(s) - 1:
            return _b.P[th, r, d, s_] - _b.P[th, r, d, s] <= det.Rd_max[th] * det.hs * \
                   det.Qg_np[th, r] * (_b.u[th, r, d, s] - _b.su[th, r, d, s]) + \
                   max(det.Pg_min[th], det.Rd_max[th] * det.hs) * det.Qg_np[th, r] * _b.sd[th, r, d, s]
        return Constraint.Skip

    b.ramp_down = Constraint(det.th_r, det.d, det.s, det.s, rule=ramp_down)

    def total_op_reserve(_b, r, d, s):
        return sum(_b.Q_spin[th, r, d, s] + _b.Q_Qstart[th, r, d, s] for th in det.th if (th, r) in det.th_r) \
               >= 0.075 * det.L[r, t, d, s]
    b.total_op_reserve = Constraint(det.r, det.d, det.s, rule=total_op_reserve)

    def total_spin_reserve(_b, r, d, s):
        return sum(_b.Q_spin[th, r, d, s] for th in det.th if (th, r) in det.th_r) >= 0.015 * det.L[r, t, d, s]

    b.total_spin_reserve = Constraint(det.r, det.d, det.s, rule=total_spin_reserve)

    def reserve_cap_1(_b, th, r, d, s):
        return _b.Q_spin[th, r, d, s] <= _b.u[th, r, d, s] * det.Qg_np[th, r] * \
               det.frac_spin[th]

    b.reserve_cap_1 = Constraint(det.th_r, det.d, det.s, rule=reserve_cap_1)

    def reserve_cap_2(_b, th, r, d, s):
        return _b.Q_Qstart[th, r, d, s] <= (_b.ngo_th[th, r] - _b.u[th, r, d, s]) \
               * det.Qg_np[th, r] * det.frac_Qstart[th]

    b.reserve_cap_2 = Constraint(det.th_r, det.d, det.s, rule=reserve_cap_2)

    def logic_RN_1(_b, rn, r):
        if t == 1:
            return _b.ngb_rn[rn, r] - _b.ngr_rn[rn, r] == _b.ngo_rn[rn, r] - \
                   det.Ng_old[rn, r]
        else:
            return _b.ngb_rn[rn, r] - _b.ngr_rn[rn, r] == _b.ngo_rn[rn, r] - \
                   _b.ngo_rn_prev[rn, r]

    b.logic_RN_1 = Constraint(det.rn_r, rule=logic_RN_1)

    def logic_TH_1(_b, th, r):
        if t == 1:
            return _b.ngb_th[th, r] - _b.ngr_th[th, r] == _b.ngo_th[th, r] - \
                   det.Ng_old[th, r]
        else:
            return _b.ngb_th[th, r] - _b.ngr_th[th, r] == _b.ngo_th[th, r] - \
                   _b.ngo_th_prev[th, r]

    b.logic_TH_1 = Constraint(det.th_r, rule=logic_TH_1)

    def logic_ROld_2(_b, rold, r):
        if rold in det.rold:
            return _b.ngr_rn[rold, r] + _b.nge_rn[rold, r] == det.Ng_r[rold, r, t]
        return Constraint.Skip

    b.logic_ROld_2 = Constraint(det.rn_r, rule=logic_ROld_2)

    def logic_TOld_2(_b, told, r):
        if told in det.told:
            return _b.ngr_th[told, r] + _b.nge_th[told, r] == det.Ng_r[told, r, t]
        return Constraint.Skip

    b.logic_TOld_2 = Constraint(det.th_r, rule=logic_TOld_2)

    def logic_3(_b, th, r, d, s):
        return _b.u[th, r, d, s] <= _b.ngo_th[th, r]

    b.logic_3 = Constraint(det.th_r, det.d, det.s, rule=logic_3)


det.Bl = Block(det.n_stage, rule=planning_block_rule)


# link blocks within same scenario


# link nodes of different scenarios:


# obj function:
# def obj_rule(t, n):
#     return det.if_[t] * (sum(det.n_d[d] * det.hs * sum((det.VOC[i, t] + det.hr[i, r] * det.P_fuel[i, t]  # ,n]
#                                                         + det.EF_CO2[i] * det.tx_CO2[t] * det.hr[i, r]) * _b.P[
#                                                            i, r, d, s]
#                                                        for i, r in det.i_r)
#                              for d in det.d for s in det.s) + sum(det.FOC[rn, t] * det.Qg_np[rn, r] * _b.ngo_rn[rn, r]
#                                                                   for rn, r in det.rn_r)
#                          + sum(det.FOC[th, t] * det.Qg_np[th, r] * _b.ngo_th[th, r]
#                                for th, r in det.th_r)
#                          + sum(det.n_d[d] * det.hs * _b.su[th, r, d, s] * det.Qg_np[th, r]
#                                * (det.f_start[th] * det.P_fuel[th, t]  # ,n]
#                                   + det.f_start[th] * det.EF_CO2[th] * det.tx_CO2[t] + det.C_start[th])
#                                for th, r in det.th_r for d in det.d for s in det.s)
#                          + sum(det.DIC[rnew, t] * det.CCm[rnew] * det.Qg_np[rnew, r] * _b.ngb_rn[rnew, r]
#                                for rnew, r in det.rn_r if rnew in det.rnew)
#                          + sum(det.DIC[tnew, t] * det.CCm[tnew] * det.Qg_np[tnew, r] * _b.ngb_th[tnew, r]
#                                for tnew, r in det.th_r if tnew in det.tnew)
#                          + sum(det.DIC[rn, t] * det.LEC[rn] * det.Qg_np[rn, r] * _b.nge_rn[rn, r]
#                                for rn, r in det.rn_r)
#                          + sum(det.DIC[th, t] * det.LEC[th] * det.Qg_np[th, r] * _b.nge_th[th, r]
#                                for th, r in det.th_r)
#                          + det.PEN[t] * _b.RES_def
#                          + det.PENc * sum(_b.cu[r, d, s]
#                                           for r in det.r for d in det.d for s in det.s)) \
#            * 10 ** (-9)
#
# det.obj = Objective(rule=obj_rule, sense=minimize)
