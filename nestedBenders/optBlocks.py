# Generation and Transmission Expansion Planning (MILP)
# IDAES project
# author: Cristiana L. Lara
# date: 10/09/2017
# Model available at http://www.optimization-online.org/DB_FILE/2017/08/6162.pdf

from pyomo.environ import *
import math
import readData


def create_model(time_periods, max_iter, n_stage, nodes, prob):
    m = ConcreteModel()

    # Declaration of sets
    m.r = Set(initialize=['Northeast', 'West', 'Coastal', 'South', 'Panhandle'], ordered=True)
    m.i = Set(initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'nuc-st-old', 'pv-old', 'wind-old',
                          'nuc-st-new', 'wind-new', 'pv-new', 'csp-new', 'coal-igcc-new', 'coal-igcc-ccs-new',
                          'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
    m.th = Set(within=m.i, initialize=['nuc-st-old', 'nuc-st-new', 'coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new',
                                       'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new', 'ng-cc-ccs-new',
                                       'ng-ct-new'], ordered=True)
    m.rn = Set(within=m.i, initialize=['pv-old', 'pv-new', 'csp-new', 'wind-old', 'wind-new'], ordered=True)
    m.co = Set(within=m.th, initialize=['coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new'], ordered=True)
    m.ng = Set(within=m.th, initialize=['ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new', 'ng-cc-ccs-new',
                                        'ng-ct-new'], ordered=True)
    m.nu = Set(within=m.th, initialize=['nuc-st-old', 'nuc-st-new'], ordered=True)
    m.pv = Set(within=m.rn, initialize=['pv-old', 'pv-new'], ordered=True)
    m.csp = Set(within=m.rn, initialize=['csp-new'], ordered=True)
    m.wi = Set(within=m.rn, initialize=['wind-old', 'wind-new'], ordered=True)
    m.old = Set(within=m.i, initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'nuc-st-old', 'pv-old',
                                        'wind-old'], ordered=True)
    m.new = Set(within=m.i, initialize=['nuc-st-new', 'wind-new', 'pv-new', 'csp-new', 'coal-igcc-new',
                                        'coal-igcc-ccs-new', 'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
    m.rold = Set(within=m.old, initialize=['pv-old', 'wind-old'], ordered=True)
    m.rnew = Set(within=m.new, initialize=['wind-new', 'pv-new', 'csp-new'], ordered=True)
    m.told = Set(within=m.old, initialize=['nuc-st-old', 'coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old'],
                 ordered=True)
    m.tnew = Set(within=m.new, initialize=['nuc-st-new', 'coal-igcc-new', 'coal-igcc-ccs-new', 'ng-cc-new',
                                           'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)

    m.d = Set(initialize=['spring', 'summer', 'fall', 'winter'], ordered=True)
    m.s = Set(initialize=['1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00',
                          '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00',
                          '22:00', '23:00', '24:00'], ordered=True)

    m.t = RangeSet(time_periods)

    # Superset of iterations
    m.iter = RangeSet(max_iter)

    # Set of iterations for which cuts are generated
    m.k = Set(within=m.iter, dimen=1, ordered=True)

    t_stage = [(t, n) for t in m.t for n in n_stage[t]]
    m.n_stage = Set(initialize=t_stage)

    # Import parameters
    m.Ng_old = Param(m.i, m.r, default=0, initialize=readData.Ng_old)

    def i_r_filter(m, i, r):
        return i in m.new or (i in m.old and m.Ng_old[i, r] != 0)

    m.i_r = Set(initialize=m.i * m.r, filter=i_r_filter, ordered=True)

    def rn_r_filter(m, rn, r):
        return rn in m.rnew or (rn in m.rold and m.Ng_old[rn, r] != 0)

    m.rn_r = Set(initialize=m.rn * m.r, filter=rn_r_filter, ordered=True)

    def th_r_filter(m, th, r):
        return th in m.tnew or (th in m.told and m.Ng_old[th, r] != 0)

    m.th_r = Set(initialize=m.th * m.r, filter=th_r_filter, ordered=True)

    m.L = Param(m.r, m.t, m.d, m.s, default=0, initialize=readData.L)
    m.n_d = Param(m.d, default=0, initialize=readData.n_ss)
    m.L_max = Param(m.n_stage, default=0, initialize=readData.L_max_s)
    # m.L_max = Param(m.t, default=0, initialize=L_max)
    m.cf = Param(m.i, m.r, m.t, m.d, m.s, default=0, initialize=readData.cf)
    m.Qg_np = Param(m.i_r, default=0, initialize=readData.Qg_np)
    m.Ng_max = Param(m.i_r, default=0, initialize=readData.Ng_max)
    m.Qinst_UB = Param(m.i, m.t, default=0, initialize=readData.Qinst_UB)
    m.LT = Param(m.i, initialize=readData.LT, default=0)
    m.Tremain = Param(m.t, default=0, initialize=readData.Tremain)
    m.Ng_r = Param(m.old, m.r, m.t, default=0, initialize=readData.Ng_r)
    m.q_v = Param(m.i, default=0, initialize=readData.q_v)
    m.Pg_min = Param(m.i, default=0, initialize=readData.Pg_min)
    m.Ru_max = Param(m.i, default=0, initialize=readData.Ru_max)
    m.Rd_max = Param(m.i, default=0, initialize=readData.Rd_max)
    m.f_start = Param(m.i, default=0, initialize=readData.f_start)
    m.C_start = Param(m.i, default=0, initialize=readData.C_start)
    m.frac_spin = Param(m.i, default=0, initialize=readData.frac_spin)
    m.frac_Qstart = Param(m.i, default=0, initialize=readData.frac_Qstart)
    m.t_loss = Param(m.r, m.r, default=0, initialize=readData.t_loss)
    m.t_up = Param(m.r, m.r, default=0, initialize=readData.t_up)
    m.dist = Param(m.r, m.r, default=0, initialize=readData.dist)
    m.if_ = Param(m.t, default=0, initialize=readData.if_)
    m.ED = Param(m.t, default=0, initialize=readData.ED)
    m.Rmin = Param(m.t, default=0, initialize=readData.Rmin)
    m.hr = Param(m.i_r, default=0, initialize=readData.hr)
    m.P_fuel = Param(m.i, m.t, default=0, initialize=readData.P_fuel)
    # m.P_fuel = Param(m.i, m.n_stage, default=0, initialize=P_fuel)
    m.EF_CO2 = Param(m.i, default=0, initialize=readData.EF_CO2)
    m.FOC = Param(m.i, m.t, default=0, initialize=readData.FOC)
    m.VOC = Param(m.i, m.t, default=0, initialize=readData.VOC)
    m.CCm = Param(m.i, default=0, initialize=readData.CCm)
    m.DIC = Param(m.i, m.t, default=0, initialize=readData.DIC)
    m.LEC = Param(m.i, default=0, initialize=readData.LEC)
    m.PEN = Param(m.t, default=0, initialize=readData.PEN)
    m.PENc = Param(default=0, initialize=readData.PENc)
    m.tx_CO2 = Param(m.t, default=0, initialize=readData.tx_CO2)
    m.RES_min = Param(m.t, default=0, initialize=readData.RES_min)
    m.hs = Param(initialize=readData.hs, default=0)
    m.ir = Param(initialize=readData.ir, default=0)
    m.prob = Param(nodes, initialize=prob)

    print("loaded the data")
    print('Starting to build the blocks')

    # Block of Equations per time period
    def planning_block_rule(b, t, n):
        print(t)
        print(n)

        def bound_P(_b, i, r, d, s):
            if i in m.old:
                return 0, m.Qg_np[i, r] * m.Ng_old[i, r]
            else:
                return 0, m.Qg_np[i, r] * m.Ng_max[i, r]

        def bound_Pflow(_b, r, r_, d, s):
            if r_ != r:
                return 0, readData.t_up[r, r_]
            else:
                return 0, 0

        def bound_o_rn(_b, rn, r):
            if rn in m.rold:
                return 0, m.Ng_old[rn, r]
            else:
                return 0, m.Ng_max[rn, r]

        def bound_b_rn(_b, rn, r):
            if rn in m.rold:
                return 0, 0
            else:
                return 0, math.floor(m.Qinst_UB[rn, t] / m.Qg_np[rn, r])

        def bound_r_rn(_b, rn, r):
            if rn in m.rold:
                return 0, m.Ng_old[rn, r]
            else:
                return 0, m.Ng_max[rn, r]

        def bound_e_rn(_b, rn, r):
            if rn in m.rold:
                return 0, m.Ng_r[rn, r, t]
            else:
                return 0, m.Ng_max[rn, r]

        def bound_o_th(_b, th, r):
            if th in m.told:
                return 0, m.Ng_old[th, r]
            else:
                return 0, m.Ng_max[th, r]

        def bound_b_th(_b, th, r):
            if th in m.told:
                return 0, 0
            else:
                return 0, math.floor(m.Qinst_UB[th, t] / m.Qg_np[th, r])

        def bound_r_th(_b, th, r):
            if th in m.told:
                return 0, m.Ng_old[th, r]
            else:
                return 0, m.Ng_max[th, r]

        def bound_e_th(_b, th, r):
            if th in m.told:
                return 0, m.Ng_r[th, r, t]
            else:
                return 0, m.Ng_max[th, r]

        def bound_UC(_b, th, r, d, s):
            if th in m.told:
                return 0, m.Ng_old[th, r]
            else:
                return 0, m.Ng_max[th, r]

        b.P = Var(m.i_r, m.d, m.s, within=NonNegativeReals, bounds=bound_P)
        b.cu = Var(m.r, m.d, m.s, within=NonNegativeReals)
        b.RES_def = Var(within=NonNegativeReals)
        b.P_flow = Var(m.r, m.r, m.d, m.s, bounds=bound_Pflow)
        b.Q_spin = Var(m.th_r, m.d, m.s, within=NonNegativeReals)
        b.Q_Qstart = Var(m.th_r, m.d, m.s, within=NonNegativeReals)
        b.ngr_rn = Var(m.rn_r, bounds=bound_r_rn, domain=NonNegativeReals)
        for rnew, r in m.rn_r:
            if rnew in m.rnew:
                if t == 1:
                    b.ngr_rn[rnew, r].fix(0.0)
                else:
                    b.ngr_rn[rnew, r].unfix()

        b.nge_rn = Var(m.rn_r, bounds=bound_e_rn, domain=NonNegativeReals)
        for rnew, r in m.rn_r:
            if rnew in m.rnew:
                if t == 1:
                    b.nge_rn[rnew, r].fix(0.0)
                else:
                    b.nge_rn[rnew, r].unfix()

        b.ngr_th = Var(m.th_r, bounds=bound_r_th, domain=NonNegativeIntegers)
        for tnew, r in m.th_r:
            if tnew in m.tnew:
                if t == 1:
                    b.ngr_th[tnew, r].fix(0.0)
                else:
                    b.ngr_th[tnew, r].unfix()

        b.nge_th = Var(m.th_r, bounds=bound_e_th, domain=NonNegativeIntegers)
        for tnew, r in m.th_r:
            if tnew in m.tnew:
                if t == 1:
                    b.nge_th[tnew, r].fix(0.0)
                else:
                    b.nge_th[tnew, r].unfix()
        b.u = Var(m.th_r, m.d, m.s, bounds=bound_UC, domain=NonNegativeIntegers)
        b.su = Var(m.th_r, m.d, m.s, bounds=bound_UC, domain=NonNegativeIntegers)
        b.sd = Var(m.th_r, m.d, m.s, bounds=bound_UC, domain=NonNegativeIntegers)
        for th, r, d, s in m.th_r * m.d * m.s:
            if m.s.ord(s) == 1:
                b.sd[th, r, d, s].fix(0.0)
            else:
                b.sd[th, r, d, s].unfix()
        b.alphafut = Var(within=Reals)

        b.ngo_rn = Var(m.rn_r, bounds=bound_o_rn, domain=NonNegativeReals)
        b.ngb_rn = Var(m.rn_r, bounds=bound_b_rn, domain=NonNegativeReals)
        for rold, r in m.rn_r:
            if rold in m.rold:
                b.ngb_rn[rold, r].fix(0.0)
        b.ngo_th = Var(m.th_r, bounds=bound_o_th, domain=NonNegativeIntegers)

        b.ngb_th = Var(m.th_r, bounds=bound_b_th, domain=NonNegativeIntegers)
        for told, r in m.th_r:
            if told in m.told:
                b.ngb_th[told, r].fix(0.0)

        b.ngo_rn_prev = Var(m.rn_r, bounds=bound_o_rn, domain=NonNegativeReals)
        b.ngb_rn_LT = Var(m.rn_r, bounds=bound_b_rn, domain=NonNegativeReals)
        for rold, r in m.rn_r:
            if rold in m.rold:
                b.ngb_rn_LT[rold, r].fix(0.0)

        b.ngo_th_prev = Var(m.th_r, bounds=bound_o_th, domain=NonNegativeReals)
        b.ngb_th_LT = Var(m.th_r, bounds=bound_b_th, domain=NonNegativeReals)
        for told, r in m.th_r:
            if told in m.told:
                b.ngb_th_LT[told, r].fix(0.0)

        def obj_rule(_b):
            return m.if_[t] * (sum(m.n_d[d] * m.hs * sum((m.VOC[i, t] + m.hr[i, r] * m.P_fuel[i, t]  # ,n]
                                        + m.EF_CO2[i] * m.tx_CO2[t] * m.hr[i, r]) * _b.P[i, r, d, s] for i, r in m.i_r)
                                   for d in m.d for s in m.s) + sum(m.FOC[rn, t] * m.Qg_np[rn, r] * _b.ngo_rn[rn, r]
                                     for rn, r in m.rn_r)
                               + sum(m.FOC[th, t] * m.Qg_np[th, r] * _b.ngo_th[th, r]
                                     for th, r in m.th_r)
                               + sum(m.n_d[d] * m.hs * _b.su[th, r, d, s] * m.Qg_np[th, r]
                                     * (m.f_start[th] * m.P_fuel[th, t]  # ,n]
                                        + m.f_start[th] * m.EF_CO2[th] * m.tx_CO2[t] + m.C_start[th])
                                     for th, r in m.th_r for d in m.d for s in m.s)
                               + sum(m.DIC[rnew, t] * m.CCm[rnew] * m.Qg_np[rnew, r] * _b.ngb_rn[rnew, r]
                                     for rnew, r in m.rn_r if rnew in m.rnew)
                               + sum(m.DIC[tnew, t] * m.CCm[tnew] * m.Qg_np[tnew, r] * _b.ngb_th[tnew, r]
                                     for tnew, r in m.th_r if tnew in m.tnew)
                               + sum(m.DIC[rn, t] * m.LEC[rn] * m.Qg_np[rn, r] * _b.nge_rn[rn, r]
                                     for rn, r in m.rn_r)
                               + sum(m.DIC[th, t] * m.LEC[th] * m.Qg_np[th, r] * _b.nge_th[th, r]
                                     for th, r in m.th_r)
                               + m.PEN[t] * _b.RES_def
                               + m.PENc * sum(_b.cu[r, d, s]
                                              for r in m.r for d in m.d for s in m.s)) \
                   * 10 ** (-9) \
                   + _b.alphafut

        b.obj = Objective(rule=obj_rule, sense=minimize)

        def min_RN_req(_b):
            return sum(m.n_d[d] * m.hs * sum(_b.P[rn, r, d, s] - _b.cu[r, d, s]for rn, r in m.i_r if rn in m.rn) \
                       for d in m.d for s in m.s) \
                   + _b.RES_def >= m.RES_min[t] * m.ED[t]

        b.min_RN_req = Constraint(rule=min_RN_req)

        def min_reserve(_b):
            return sum(m.Qg_np[rn, r] * _b.ngo_rn[rn, r] * m.q_v[rn] for rn, r in m.i_r if rn in m.rn) \
                   + sum(m.Qg_np[th, r] * _b.ngo_th[th, r] for th, r in m.i_r if th in m.th) \
                   >= (1 + m.Rmin[t]) * m.L_max[t, n]

        b.min_reserve = Constraint(rule=min_reserve)

        def inst_RN_UB(_b, rnew):
            return sum(_b.ngb_rn[rnew, r] for r in m.r) \
                   <= m.Qinst_UB[rnew, t] / sum(m.Qg_np[rnew, r] / len(m.r) for r in m.r)

        b.inst_RN_UB = Constraint(m.rnew, rule=inst_RN_UB)

        def inst_TH_UB(_b, tnew):
            return sum(_b.ngb_th[tnew, r] for r in m.r) \
                   <= m.Qinst_UB[tnew, t] / sum(m.Qg_np[tnew, r] / len(m.r) for r in m.r)

        b.inst_TH_UB = Constraint(m.tnew, rule=inst_TH_UB)

        def en_bal(_b, r, d, s):
            return sum(_b.P[i, r, d, s] for i in m.i if (i, r) in m.i_r) \
                   + sum(_b.P_flow[r_, r, d, s] * (1 - m.t_loss[r, r_] * m.dist[r, r_]) - \
                         _b.P_flow[r, r_, d, s] for r_ in m.r if r_ != r) \
                   == m.L[r, t, d, s] + _b.cu[r, d, s]

        b.en_bal = Constraint(m.r, m.d, m.s, rule=en_bal)

        def capfactor(_b, rn, r, d, s):
            return _b.P[rn, r, d, s] == m.Qg_np[rn, r] * m.cf[rn, r, t, d, s] \
                   * _b.ngo_rn[rn, r]

        b.capfactor = Constraint(m.rn_r, m.d, m.s, rule=capfactor)

        def min_output(_b, th, r, d, s):
            return _b.u[th, r, d, s] * m.Pg_min[th] * m.Qg_np[th, r] <= _b.P[th, r, d, s]

        b.min_output = Constraint(m.th_r, m.d, m.s, rule=min_output)

        def max_output(_b, th, r, d, s):
            return _b.u[th, r, d, s] * m.Qg_np[th, r] >= _b.P[th, r, d, s] \
                   + _b.Q_spin[th, r, d, s]

        b.max_output = Constraint(m.th_r, m.d, m.s, rule=max_output)

        def unit_commit1(_b, th, r, d, s, s_):
            if m.s.ord(s_) == m.s.ord(s) - 1:
                return _b.u[th, r, d, s] == _b.u[th, r, d, s_] + _b.su[th, r, d, s] \
                       - _b.sd[th, r, d, s]
            return Constraint.Skip

        b.unit_commit1 = Constraint(m.th_r, m.d, m.s, m.s, rule=unit_commit1)

        def ramp_up(_b, th, r, d, s, s_):
            if (th, r) in m.i_r and th in m.th and (m.s.ord(s_) == m.s.ord(s) - 1):
                return _b.P[th, r, d, s] - _b.P[th, r, d, s_] <= m.Ru_max[th] * m.hs * \
                       m.Qg_np[th, r] * (_b.u[th, r, d, s] - _b.su[th, r, d, s]) + \
                       max(m.Pg_min[th], m.Ru_max[th] * m.hs) * m.Qg_np[th, r] * _b.su[th, r, d, s]
            return Constraint.Skip

        b.ramp_up = Constraint(m.th, m.r, m.d, m.s, m.s, rule=ramp_up)

        def ramp_down(_b, th, r, d, s, s_):
            if m.s.ord(s_) == m.s.ord(s) - 1:
                return _b.P[th, r, d, s_] - _b.P[th, r, d, s] <= m.Rd_max[th] * m.hs * \
                       m.Qg_np[th, r] * (_b.u[th, r, d, s] - _b.su[th, r, d, s]) + \
                       max(m.Pg_min[th], m.Rd_max[th] * m.hs) * m.Qg_np[th, r] * _b.sd[th, r, d, s]
            return Constraint.Skip

        b.ramp_down = Constraint(m.th_r, m.d, m.s, m.s, rule=ramp_down)

        def total_op_reserve(_b, r, d, s):
            return sum(_b.Q_spin[th, r, d, s] + _b.Q_Qstart[th, r, d, s] for th in m.th) \
                   >= 0.075 * m.L[r, t, d, s]
            b.total_op_reserve = Constraint(m.r, m.d, m.s, rule=total_op_reserve)

        def total_spin_reserve(_b, r, d, s):
            return sum(_b.Q_spin[th, r, d, s] for th in m.th if (th, r) in m.th_r) >= 0.015 * m.L[r, t, d, s]

        b.total_spin_reserve = Constraint(m.r, m.d, m.s, rule=total_spin_reserve)

        def reserve_cap_1(_b, th, r, d, s):
            return _b.Q_spin[th, r, d, s] <= _b.u[th, r, d, s] * m.Qg_np[th, r] * \
                   m.frac_spin[th]

        b.reserve_cap_1 = Constraint(m.th_r, m.d, m.s, rule=reserve_cap_1)

        def reserve_cap_2(_b, th, r, d, s):
            return _b.Q_Qstart[th, r, d, s] <= (_b.ngo_th[th, r] - _b.u[th, r, d, s]) \
                   * m.Qg_np[th, r] * m.frac_Qstart[th]

        b.reserve_cap_2 = Constraint(m.th_r, m.d, m.s, rule=reserve_cap_2)

        def logic_RN_1(_b, rn, r):
            if t == 1:
                return _b.ngb_rn[rn, r] - _b.ngr_rn[rn, r] == _b.ngo_rn[rn, r] - \
                       m.Ng_old[rn, r]
            else:
                return _b.ngb_rn[rn, r] - _b.ngr_rn[rn, r] == _b.ngo_rn[rn, r] - \
                       _b.ngo_rn_prev[rn, r]

        b.logic_RN_1 = Constraint(m.rn_r, rule=logic_RN_1)

        def logic_TH_1(_b, th, r):
            if t == 1:
                return _b.ngb_th[th, r] - _b.ngr_th[th, r] == _b.ngo_th[th, r] - \
                       m.Ng_old[th, r]
            else:
                return _b.ngb_th[th, r] - _b.ngr_th[th, r] == _b.ngo_th[th, r] - \
                       _b.ngo_th_prev[th, r]

        b.logic_TH_1 = Constraint(m.th_r, rule=logic_TH_1)

        def logic_RNew_2(_b, rnew, r):
            if rnew in m.rnew:
                return _b.ngb_rn_LT[rnew, r] == _b.ngr_rn[rnew, r] + _b.nge_rn[rnew, r]
            return Constraint.Skip

        b.logic_RNew_2 = Constraint(m.rn_r, rule=logic_RNew_2)

        def logic_TNew_2(_b, tnew, r):
            if tnew in m.tnew:
                return _b.ngb_th_LT[tnew, r] == _b.ngr_th[tnew, r] + _b.nge_th[tnew, r]
            return Constraint.Skip

        b.logic_TNew_2 = Constraint(m.th_r, rule=logic_TNew_2)

        def logic_ROld_2(_b, rold, r):
            if rold in m.rold:
                return _b.ngr_rn[rold, r] + _b.nge_rn[rold, r] == m.Ng_r[rold, r, t]
            return Constraint.Skip

        b.logic_ROld_2 = Constraint(m.rn_r, rule=logic_ROld_2)

        def logic_TOld_2(_b, told, r):
            if told in m.told:
                return _b.ngr_th[told, r] + _b.nge_th[told, r] == m.Ng_r[told, r, t]
            return Constraint.Skip

        b.logic_TOld_2 = Constraint(m.th_r, rule=logic_TOld_2)

        def logic_3(_b, th, r, d, s):
            return _b.u[th, r, d, s] <= _b.ngo_th[th, r]

        b.logic_3 = Constraint(m.th_r, m.d, m.s, rule=logic_3)

        b.link_equal1 = ConstraintList()  # , rule=link_equal1)

        b.link_equal2 = ConstraintList()  # m.th, m.r, rule=link_equal2)

        b.link_equal3 = ConstraintList()  # m.rn, m.r, rule=link_equal3)

        b.link_equal4 = ConstraintList()  # m.th, m.r, rule=link_equal4)

        b.fut_cost = ConstraintList()

    m.Bl = Block(m.n_stage, rule=planning_block_rule)  # TODO: this is the line that generate the blocks

    # Decomposition Parameters
    m.ngo_rn_par = Param(m.rn_r, m.n_stage, default=0, initialize=0, mutable=True)
    m.ngo_th_par = Param(m.th_r, m.n_stage, default=0, initialize=0, mutable=True)
    m.ngb_rn_par = Param(m.rn_r, m.n_stage, default=0, initialize=0, mutable=True)
    m.ngb_th_par = Param(m.th_r, m.n_stage, default=0, initialize=0, mutable=True)
    m.ngo_rn_par_k = Param(m.rn_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.ngo_th_par_k = Param(m.th_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.ngb_rn_par_k = Param(m.rn_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.ngb_th_par_k = Param(m.th_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.cost = Param(m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.mltp_o_rn = Param(m.rn_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.mltp_o_th = Param(m.th_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.mltp_b_rn = Param(m.rn_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.mltp_b_th = Param(m.th_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
    m.cost_t = Param(m.n_stage, m.iter, default=0, initialize=0, mutable=True)

    return m
