# Generation and Transmission Expansion Planning (Multi-stage Stochastic Integer Programming)
# IDAES project
# author: Cristiana L. Lara
# date: 10/09/2017
# Model available at http://www.optimization-online.org/DB_FILE/2017/08/6162.pdf

from pyomo.environ import *
import math
import readData


def create_model(stages, time_periods, t_per_stage, max_iter):
    m = ConcreteModel()

    # ################################## Declare of sets ##################################
    '''
        Set Notation:
        m.r: regions

        m.i: generators
        m.th: thermal generators
        m.rn: renewable generators
        m.co: coal-based generators
            coal-st-old1: cluster of existing coal steam turbine generators
            coal-igcc-new: cluster of potential coal IGCC generators
            coal-igcc-ccs-new: cluster of potential coal IGCC generators with carbon capture
        m.ng: natural gas (NG) generators
            ng-ct-old: cluster of existing NG combustion-fired turbine generators
            ng-cc-old: cluster of existing NG combined-cycle generators
            ng-st-old: cluster of existing NG steam-turbine generators
            ng-cc-new: cluster of potential NG combined-cycle generators
            ng-cc-ccs-new: cluster of potential NG combined-cycle generators with carbon capture
            ng-ct-new: cluster of potential NG combustion-fired turbine generators
        m.nu: nuclear generators
            nuc-st-old: cluster of existing nuclear steam turbine generators
            nuc-st-new: cluster of potential nuclear steam turbine generators
        m.pv: solar photovoltaic generators
            pv-old: cluster of existing solar photovoltaic generators
            pv-new: cluster of potential solar photovoltaic generators
        m.csp: concentrated solar panels
            csp-new: cluster of potential concentrated solar panels
        m.wi: wind turbines
            wind-old: cluster of existing wind turbines
            wind-new: cluster of potential wind turbines
        m.old: subset of existing generators
        m.new: subset of potential generators
        m.rold: subset of existing renewable generators
        m.rnew: subset of potential renewable generators
        m.told: subset of existing thermal generators
        m.tnew: subset of potential thermal generators
        
        m.j: clusters of potential storage unit

        m.d: set of representative days

        m.hours: set of subperiods within the days

        m.t: set of time periods

        m.stage: set of stages in the scenario tree
    '''
    m.r = Set(initialize=['Northeast', 'West', 'Coastal', 'South', 'Panhandle'], ordered=True)
    m.i = Set(initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'pv-old', 'wind-old',
                          'wind-new', 'pv-new', 'csp-new', 'coal-igcc-new', 'coal-igcc-ccs-new',
                          'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new', 'nuc-st-old', 'nuc-st-new'], ordered=True)
    m.th = Set(within=m.i, initialize=['coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new',
                                       'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new', 'ng-cc-ccs-new',
                                       'ng-ct-new', 'nuc-st-old', 'nuc-st-new'], ordered=True)
    m.rn = Set(within=m.i, initialize=['pv-old', 'pv-new', 'csp-new', 'wind-old', 'wind-new'], ordered=True)
    m.co = Set(within=m.th, initialize=['coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new'], ordered=True)
    m.ng = Set(within=m.th, initialize=['ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new', 'ng-cc-ccs-new',
                                        'ng-ct-new'], ordered=True)
    m.nu = Set(within=m.th, initialize=['nuc-st-old', 'nuc-st-new'], ordered=True)
    m.pv = Set(within=m.rn, initialize=['pv-old', 'pv-new'], ordered=True)
    m.csp = Set(within=m.rn, initialize=['csp-new'], ordered=True)
    m.wi = Set(within=m.rn, initialize=['wind-old', 'wind-new'], ordered=True)
    m.old = Set(within=m.i, initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'pv-old',
                                        'wind-old', 'nuc-st-old'], ordered=True)
    m.new = Set(within=m.i, initialize=['wind-new', 'pv-new', 'csp-new', 'coal-igcc-new', 'nuc-st-new',
                                        'coal-igcc-ccs-new', 'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
    m.rold = Set(within=m.old, initialize=['pv-old', 'wind-old'], ordered=True)
    m.rnew = Set(within=m.new, initialize=['wind-new', 'pv-new', 'csp-new'], ordered=True)
    m.told = Set(within=m.old, initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old', 'nuc-st-old'],
                 ordered=True)
    m.tnew = Set(within=m.new, initialize=['coal-igcc-new', 'coal-igcc-ccs-new', 'ng-cc-new', 'nuc-st-new',
                                           'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
    m.j = Set(initialize=['Li_ion', 'Lead_acid', 'Flow'], ordered=True)

    m.d = Set(initialize=['spring', 'summer', 'fall', 'winter'], ordered=True)
    #  Misleading (seasons not used) but used because of structure of old data

    m.hours = Set(initialize=['1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00',
                              '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00',
                              '22:00', '23:00', '24:00'], ordered=True)

    m.t = RangeSet(time_periods)

    m.stages = RangeSet(stages)

    t_stage = [(t, stage) for stage in m.stages for t in t_per_stage[stage]]
    m.t_stage = Set(initialize=t_stage)

    # Superset of iterations
    m.iter = RangeSet(max_iter)

    # Set of iterations for which cuts are generated
    m.k = Set(within=m.iter, dimen=1, ordered=True)

    # ################################## Import parameters ############################################
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

    '''
    Parameter notation
    
    m.L: load demand in region r in sub-period s of representative day d of year t (MW)
    m.n_d: weight of representative day d
    m.L_max: peak load in year t (MW)
    m.cf: capacity factor of renewable generation cluster i in region r at sub-period s, of representative day d of r
        year t (fraction of the nameplate capacity)
    m.Qg_np: generator nameplate capacity (MW)
    m.Ng_max: max number of generators in cluster i of region r
    m.Qinst_UB: Yearly upper bound on installation capacity by generator type
    m.LT: expected lifetime of generation cluster i (years)
    m.Tremain: remaining time until the end of the time horizon at year t (years)
    m.Ng_r: number of generators in cluster i of region r that achieved their expected lifetime
    m.q_v: capacity value of generation cluster i (fraction of the nameplate capacity)
    m.Pg_min: minimum operating output of a generator in cluster i ∈ ITH (fraction of the nameplate capacity)
    m.Ru_max: maximum ramp-up rate for cluster i ∈ ITH (fraction of nameplate capacity)
    m.Rd_max: maximum ramp-down rate for cluster i ∈ ITH (fraction of nameplate capacity)
    m.f_start: fuel usage at startup (MMbtu/MW)
    m.C_start: 􏰄xed startup cost for generator cluster i ($/MW)
    m.frac_spin: maximum fraction of nameplate capacity of each generator that can contribute to spinning reserves
        (fraction of nameplate capacity)
    m.frac_Qstart: maximum fraction of nameplate capacity of each generator that can contribute to quick-start reserves
        (fraction of nameplate capacity)
    m.t_loss: transmission loss factor between region r and region r ̸= r (%/miles)
    m.t_up: transmission line capacity
    m.dist: distance between region r and region r′ ̸= r (miles)
    m.if_: discount factor for year t
    m.ED: energy demand during year t (MWh)
    m.Rmin: system's minimum reserve margin for year t (fraction of the peak load)
    m.hr: heat rate of generator cluster i (MMBtu/MWh)
    m.P_fuel: price of fuel for generator cluster i in year t ($/MMBtu)
    m.EF_CO2: full lifecycle CO2 emission factor for generator cluster i (kgCO2/MMBtu)
    m.FOC: 􏰄xed operating cost of generator cluster i ($/MW)
    m.VOC: variable O&M cost of generator cluster i ($/MWh)
    m.CCm: capital cost multiplier of generator cluster i (unitless)
    m.DIC: discounted investment cost of generator cluster i in year t ($/MW)
    m.LEC: life extension cost for generator cluster i (fraction of the investment cost of corresponding new generator)
    m.PEN: penalty for not meeting renewable energy quota target during year t ($/MWh)
    m.PENc: penalty for curtailment during year t ($/MWh)
    m.tx_CO2: carbon tax in year t ($/kg CO2)
    m.RES_min: minimum renewable energy production requirement during year t (fraction of annual energy demand)
    m.hs: duration of sub-period s (hours)
    m.ir: interest rate
    m.storage_inv_cost: investment cost of storage unit of type j in year t [$/MW]
    m.P_min_charge: min power storage charge for unit j [MW]
    m.P_max_charge: max power storage charge for unit j [MW]
    m.P_min_discharge: min power storage discharge for unit j [MW]
    m.P_max_discharge: max power storage discharge for unit j [MW]
    m.min_storage_cap: min storage capacity for unit j [MWh]
    m.max_storage_cap: max storage capacity for unit j [MWh]
    m.eff_rate_charge: efficiency rate to charge energy in storage unit j
    m.eff_rate_discharge: efficiency rate to discharge energy in storage unit j
    m.storage_lifetime: storage lifetime (years)
    '''

    m.L = Param(m.r, m.t, m.d, m.hours, default=0, mutable=True)  # initialize=readData.L)
    m.n_d = Param(m.d, default=0, initialize=readData.n_ss)
    # m.L_max = Param(m.t_stage, default=0, mutable=True)
    m.L_max = Param(m.t, default=0, initialize=readData.L_max)
    m.cf = Param(m.i, m.r, m.t, m.d, m.hours, default=0, mutable=True)  # initialize=readData.cf)
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
    # m.P_fuel = Param(m.i, m.t, default=0, initialize=readData.P_fuel)
    m.P_fuel = Param(m.i, m.t_stage, default=0, mutable=True)
    m.EF_CO2 = Param(m.i, default=0, initialize=readData.EF_CO2)
    m.FOC = Param(m.i, m.t, default=0, initialize=readData.FOC)
    m.VOC = Param(m.i, m.t, default=0, initialize=readData.VOC)
    m.CCm = Param(m.i, default=0, initialize=readData.CCm)
    m.DIC = Param(m.i, m.t, default=0, initialize=readData.DIC)
    m.LEC = Param(m.i, default=0, initialize=readData.LEC)
    m.PEN = Param(m.t, default=0, initialize=readData.PEN)
    m.PENc = Param(default=0, initialize=readData.PENc)
    m.tx_CO2 = Param(m.t, default=0, initialize=readData.tx_CO2)
    # m.tx_CO2 = Param(m.t_stage, default=0, mutable=True)
    m.RES_min = Param(m.t, default=0, initialize=readData.RES_min)
    m.hs = Param(initialize=readData.hs, default=0)
    m.ir = Param(initialize=readData.ir, default=0)

    # Storage
    m.storage_inv_cost = Param(m.j, m.t, default=0, initialize=readData.storage_inv_cost)
    m.P_min_charge = Param(m.j, default=0, initialize=readData.P_min_charge)
    m.P_max_charge = Param(m.j, default=0, initialize=readData.P_max_charge)
    m.P_min_discharge = Param(m.j, default=0, initialize=readData.P_min_discharge)
    m.P_max_discharge = Param(m.j, default=0, initialize=readData.P_max_discharge)
    m.min_storage_cap = Param(m.j, default=0, initialize=readData.min_storage_cap)
    m.max_storage_cap = Param(m.j, default=0, initialize=readData.max_storage_cap)
    m.eff_rate_charge = Param(m.j, default=0, initialize=readData.eff_rate_charge)
    m.eff_rate_discharge = Param(m.j, default=0, initialize=readData.eff_rate_discharge)
    m.storage_lifetime = Param(m.j, default=0, initialize=readData.storage_lifetime)

    # Block of Equations per time period
    def planning_block_rule(b, stage):

        def bound_P(_b, i, r, t, d, s):
            if i in m.old:
                return 0, m.Qg_np[i, r] * m.Ng_old[i, r]
            else:
                return 0, m.Qg_np[i, r] * m.Ng_max[i, r]

        def bound_Pflow(_b, r, r_, t, d, s):
            if r_ != r:
                return 0, readData.t_up[r, r_]
            else:
                return 0, 0

        def bound_o_rn(_b, rn, r, t):
            if rn in m.rold:
                return 0, m.Ng_old[rn, r]
            else:
                return 0, m.Ng_max[rn, r]

        def bound_o_rn_prev(_b, rn, r):
            if rn in m.rold:
                return 0, m.Ng_old[rn, r]
            else:
                return 0, m.Ng_max[rn, r]

        def bound_b_rn(_b, rn, r, t):
            if rn in m.rold:
                return 0, 0
            else:
                return 0, math.floor(m.Qinst_UB[rn, t] / m.Qg_np[rn, r])

        def bound_r_rn(_b, rn, r, t):
            if rn in m.rold:
                return 0, m.Ng_old[rn, r]
            else:
                return 0, m.Ng_max[rn, r]

        def bound_e_rn(_b, rn, r, t):
            if rn in m.rold:
                return 0, m.Ng_r[rn, r, t]
            else:
                return 0, m.Ng_max[rn, r]

        def bound_o_th(_b, th, r, t):
            if th in m.told:
                return 0, m.Ng_old[th, r]
            else:
                return 0, m.Ng_max[th, r]

        def bound_o_th_prev(_b, th, r):
            if th in m.told:
                return 0, m.Ng_old[th, r]
            else:
                return 0, m.Ng_max[th, r]

        def bound_b_th(_b, th, r, t):
            if th in m.told:
                return 0, 0
            else:
                return 0, math.floor(m.Qinst_UB[th, t] / m.Qg_np[th, r])

        def bound_r_th(_b, th, r, t):
            if th in m.told:
                return 0, m.Ng_old[th, r]
            else:
                return 0, m.Ng_max[th, r]

        def bound_e_th(_b, th, r, t):
            if th in m.told:
                return 0, m.Ng_r[th, r, t]
            else:
                return 0, m.Ng_max[th, r]

        def bound_UC(_b, th, r, t, d, s):
            if th in m.told:
                return 0, m.Ng_old[th, r]
            else:
                return 0, m.Ng_max[th, r]

        '''
        Variables notation:
        b.P: power output of generation cluster i in region r during sub-period s of representative day d of year t (MW)
        b.cu: curtailment slack generation in region r during sub-period s of representative day d of year t (MW)
        b.RES_def: de􏰂cit from renewable energy quota target during year t (MWh)
        b.P_flow: power transfer from region r to region r̸=r during sub-period s of representative day d of year t (MW)
        b.Q_spin:spinning reserve capacity of generation cluster i in region r during sub-period s of representative day 
            d of year t (MW)
        b.Q_Qstart: quick-start capacity reserve of generation cluster i in region r during sub-period s of 
            representative day d of year t (MW)
        b.ngr_rn: number of generators that retire in cluster i ∈ IRN of region r in year t (continuous relaxation)
        b.nge_rn: number of generators that had their life extended in cluster i ∈ IRN of region r in year t 
            (continuous relaxation)
        b.ngb_rn: number of generators that are built in cluster i ∈ IRN of region r in year t (continuous relaxation)
        b.ngo_rn: number of generators that are operational in cluster i ∈ IRN of region r in year t (continuous r
            relaxation)
        b.ngr_th: number of generators that retire in cluster i ∈ ITH of region r in year t (integer variable)
        b.nge_th: number of generators that had their life extended in cluster i ∈ ITH of region r in year t (integer r
            variable)
        b.ngb_th: number of generators that are built in cluster i ∈ ITH of region r in year t (integer variable) r
        b.ngo_th: number of generators that are operational in cluster i ∈ ITH of region r in year t (integer variable)
        b.u: number of thermal generators ON in cluster i ∈ Ir of region r during sub-period s of representative day 
            d of year t (integer variable)
        b.su: number of generators starting up in cluster i during sub-period s of representative day d in year t 
            (integer variable)
        b.sd: number of generators shutting down in cluster i during sub-period s of representative day d in year t 
            (integer variable)
        b.p_charged: power charged into storage in region r, day d, hour h, year t [MW]
        b.p_discharged : power discharged into storage in region r, day d, hour h, year t [MW]
        b.p_storage_level: ending state of charge of storage in region r, day d, hour h, year t [MWh]
        b.p_storage_level_end_hour: ending state of charge of storage in region r, day d, hour h, year t [MWh]
        b.nsb: Number of storage units of type j installed in region r, year t (relaxed to continuous)
        m.nso: Number of storage units of type j operational in region r, year t (relaxed to continuous)
        m.nsr: Number of storage units of type j retired in region r, year t (relaxed to continuous)
        '''

        b.P = Var(m.i_r, t_per_stage[stage], m.d, m.hours, within=NonNegativeReals, bounds=bound_P)
        b.cu = Var(m.r, t_per_stage[stage], m.d, m.hours, within=NonNegativeReals)
        b.RES_def = Var(t_per_stage[stage], within=NonNegativeReals)
        b.P_flow = Var(m.r, m.r, t_per_stage[stage], m.d, m.hours, bounds=bound_Pflow)
        b.Q_spin = Var(m.th_r, t_per_stage[stage], m.d, m.hours, within=NonNegativeReals)
        b.Q_Qstart = Var(m.th_r, t_per_stage[stage], m.d, m.hours, within=NonNegativeReals)
        b.ngr_rn = Var(m.rn_r, t_per_stage[stage], bounds=bound_r_rn, domain=NonNegativeReals)
        for t in t_per_stage[stage]:
            if t == 1:
                for rnew, r in m.rn_r:
                    if rnew in m.rnew:
                        b.ngr_rn[rnew, r, t].fix(0.0)
            else:
                for rnew, r in m.rn_r:
                    if rnew in m.rnew:
                        b.ngr_rn[rnew, r, t].unfix()

        b.nge_rn = Var(m.rn_r, t_per_stage[stage], bounds=bound_e_rn, domain=NonNegativeReals)
        for t in t_per_stage[stage]:
            if t == 1:
                for rnew, r in m.rn_r:
                    if rnew in m.rnew:
                        b.nge_rn[rnew, r, t].fix(0.0)
            else:
                for rnew, r in m.rn_r:
                    if rnew in m.rnew:
                        b.nge_rn[rnew, r, t].unfix()

        b.ngr_th = Var(m.th_r, t_per_stage[stage], bounds=bound_r_th, domain=NonNegativeIntegers)
        for t in t_per_stage[stage]:
            if t == 1:
                for tnew, r in m.th_r:
                    if tnew in m.tnew:
                        b.ngr_th[tnew, r, t].fix(0.0)
            else:
                for tnew, r in m.th_r:
                    if tnew in m.tnew:
                        b.ngr_th[tnew, r, t].unfix()

        b.nge_th = Var(m.th_r, t_per_stage[stage], bounds=bound_e_th, domain=NonNegativeIntegers)
        for t in t_per_stage[stage]:
            if t == 1:
                for tnew, r in m.th_r:
                    if tnew in m.tnew:
                        b.nge_th[tnew, r, t].fix(0.0)
            else:
                for tnew, r in m.th_r:
                    if tnew in m.tnew:
                        b.nge_th[tnew, r, t].unfix()
        b.u = Var(m.th_r, t_per_stage[stage], m.d, m.hours, bounds=bound_UC, domain=NonNegativeIntegers)
        b.su = Var(m.th_r, t_per_stage[stage], m.d, m.hours, bounds=bound_UC, domain=NonNegativeIntegers)
        b.sd = Var(m.th_r, t_per_stage[stage], m.d, m.hours, bounds=bound_UC, domain=NonNegativeIntegers)
        for th, r, t, d, s in m.th_r * t_per_stage[stage] * m.d * m.hours:
            if m.hours.ord(s) == 1:
                b.sd[th, r, t, d, s].fix(0.0)
            else:
                b.sd[th, r, t, d, s].unfix()
        b.alphafut = Var(within=Reals, domain=NonNegativeReals)

        b.ngo_rn = Var(m.rn_r, t_per_stage[stage], bounds=bound_o_rn, domain=NonNegativeReals)
        b.ngb_rn = Var(m.rn_r, t_per_stage[stage], bounds=bound_b_rn, domain=NonNegativeReals)
        for t in t_per_stage[stage]:
            for rold, r in m.rn_r:
                if rold in m.rold:
                    b.ngb_rn[rold, r, t].fix(0.0)

        b.ngo_th = Var(m.th_r, t_per_stage[stage], bounds=bound_o_th, domain=NonNegativeIntegers)
        b.ngb_th = Var(m.th_r, t_per_stage[stage], bounds=bound_b_th, domain=NonNegativeIntegers)
        for t in t_per_stage[stage]:
            for told, r in m.th_r:
                if told in m.told:
                    b.ngb_th[told, r, t].fix(0.0)

        b.ngo_rn_prev = Var(m.rn_r, bounds=bound_o_rn_prev, domain=NonNegativeReals)
        b.ngo_th_prev = Var(m.th_r, bounds=bound_o_th_prev, domain=NonNegativeReals)

        # Storage related Variables
        b.p_charged = Var(m.j, m.r, t_per_stage[stage], m.d, m.hours, within=NonNegativeReals)
        b.p_discharged = Var(m.j, m.r, t_per_stage[stage], m.d, m.hours, within=NonNegativeReals)
        b.p_storage_level = Var(m.j, m.r, t_per_stage[stage], m.d, m.hours, within=NonNegativeReals)
        b.p_storage_level_end_hour = Var(m.r, t_per_stage[stage], m.d, m.hours, within=NonNegativeReals)
        b.nsr = Var(m.j, m.r, t_per_stage[stage], within=NonNegativeReals)
        b.nsb = Var(m.j, m.r, t_per_stage[stage], within=NonNegativeReals)
        b.nso = Var(m.j, m.r, t_per_stage[stage], within=NonNegativeReals)

        b.nso_prev = Var(m.j, m.r, within=NonNegativeReals)

        def obj_rule(_b):
            return sum(m.if_[t] * (sum(m.n_d[d] * m.hs * sum((m.VOC[i, t] + m.hr[i, r] * m.P_fuel[i, t, stage]
                                                              + m.EF_CO2[i] * m.tx_CO2[t] * m.hr[i, r]) * _b.P[
                                                                 i, r, t, d, s]
                                                             for i, r in m.i_r)
                                       for d in m.d for s in m.hours) + sum(m.FOC[rn, t] * m.Qg_np[rn, r] *
                                                                            _b.ngo_rn[rn, r, t] for rn, r in m.rn_r)
                                   + sum(m.FOC[th, t] * m.Qg_np[th, r] * _b.ngo_th[th, r, t]
                                         for th, r in m.th_r)
                                   + sum(m.n_d[d] * m.hs * _b.su[th, r, t, d, s] * m.Qg_np[th, r]
                                         * (m.f_start[th] * m.P_fuel[th, t, stage]
                                            + m.f_start[th] * m.EF_CO2[th] * m.tx_CO2[t] + m.C_start[th])
                                         for th, r in m.th_r for d in m.d for s in m.hours)
                                   + sum(m.DIC[rnew, t] * m.CCm[rnew] * m.Qg_np[rnew, r] * _b.ngb_rn[rnew, r, t]
                                         for rnew, r in m.rn_r if rnew in m.rnew)
                                   + sum(m.DIC[tnew, t] * m.CCm[tnew] * m.Qg_np[tnew, r] * _b.ngb_th[tnew, r, t]
                                         for tnew, r in m.th_r if tnew in m.tnew)
                                   + sum(m.DIC[rn, t] * m.LEC[rn] * m.Qg_np[rn, r] * _b.nge_rn[rn, r, t]
                                         for rn, r in m.rn_r)
                                   + sum(m.DIC[th, t] * m.LEC[th] * m.Qg_np[th, r] * _b.nge_th[th, r, t]
                                         for th, r in m.th_r)
                                   + sum(m.storage_inv_cost[j, t] * m.max_storage_cap[j] * _b.nsb[j, r, t]
                                         for j in m.j for r in m.r)
                                   + m.PEN[t] * _b.RES_def[t]
                                   + m.PENc * sum(_b.cu[r, t, d, s]
                                                  for r in m.r for d in m.d for s in m.hours))
                       for t in t_per_stage[stage]) \
                   * 10 ** (-9) \
                   + _b.alphafut

        b.obj = Objective(rule=obj_rule, sense=minimize)

        def min_RN_req(_b, t):
            return sum(m.n_d[d] * m.hs * sum(_b.P[rn, r, t, d, s] - _b.cu[r, t, d, s] for rn, r in m.i_r if rn in m.rn) \
                       for d in m.d for s in m.hours) \
                   + _b.RES_def[t] >= m.RES_min[t] * m.ED[t]

        b.min_RN_req = Constraint(t_per_stage[stage], rule=min_RN_req)

        def min_reserve(_b, t):
            return sum(m.Qg_np[rn, r] * _b.ngo_rn[rn, r, t] * m.q_v[rn] for rn, r in m.i_r if rn in m.rn) \
                   + sum(m.Qg_np[th, r] * _b.ngo_th[th, r, t] for th, r in m.i_r if th in m.th) \
                   >= (1 + m.Rmin[t]) * m.L_max[t]

        b.min_reserve = Constraint(t_per_stage[stage], rule=min_reserve)

        def inst_RN_UB(_b, rnew, t):
            return sum(_b.ngb_rn[rnew, r, t] for r in m.r) \
                   <= m.Qinst_UB[rnew, t] / sum(m.Qg_np[rnew, r] / len(m.r) for r in m.r)

        b.inst_RN_UB = Constraint(m.rnew, t_per_stage[stage], rule=inst_RN_UB)

        def inst_TH_UB(_b, tnew, t):
            return sum(_b.ngb_th[tnew, r, t] for r in m.r) \
                   <= m.Qinst_UB[tnew, t] / sum(m.Qg_np[tnew, r] / len(m.r) for r in m.r)

        b.inst_TH_UB = Constraint(m.tnew, t_per_stage[stage], rule=inst_TH_UB)

        def en_bal(_b, r, t, d, s):
            return sum(_b.P[i, r, t, d, s] for i in m.i if (i, r) in m.i_r) \
                   + sum(_b.P_flow[r_, r, t, d, s] * (1 - m.t_loss[r, r_] * m.dist[r, r_]) -
                         _b.P_flow[r, r_, t, d, s] for r_ in m.r if r_ != r) \
                   + sum(_b.p_discharged[j, r, t, d, s] for j in m.j) \
                   == m.L[r, t, d, s] + sum(_b.p_charged[j, r, t, d, s] for j in m.j) + _b.cu[r, t, d, s]

        b.en_bal = Constraint(m.r, t_per_stage[stage], m.d, m.hours, rule=en_bal)

        def capfactor(_b, rn, r, t, d, s):
            return _b.P[rn, r, t, d, s] == m.Qg_np[rn, r] * m.cf[rn, r, t, d, s] \
                   * _b.ngo_rn[rn, r, t]

        b.capfactor = Constraint(m.rn_r, t_per_stage[stage], m.d, m.hours, rule=capfactor)

        def min_output(_b, th, r, t, d, s):
            return _b.u[th, r, t, d, s] * m.Pg_min[th] * m.Qg_np[th, r] <= _b.P[th, r, t, d, s]

        b.min_output = Constraint(m.th_r, t_per_stage[stage], m.d, m.hours, rule=min_output)

        def max_output(_b, th, r, t, d, s):
            return _b.u[th, r, t, d, s] * m.Qg_np[th, r] >= _b.P[th, r, t, d, s] \
                   + _b.Q_spin[th, r, t, d, s]

        b.max_output = Constraint(m.th_r, t_per_stage[stage], m.d, m.hours, rule=max_output)

        def unit_commit1(_b, th, r, t, d, s, s_):
            if m.hours.ord(s_) == m.hours.ord(s) - 1:
                return _b.u[th, r, t, d, s] == _b.u[th, r, t, d, s_] + _b.su[th, r, t, d, s] \
                       - _b.sd[th, r, t, d, s]
            return Constraint.Skip

        b.unit_commit1 = Constraint(m.th_r, t_per_stage[stage], m.d, m.hours, m.hours, rule=unit_commit1)

        def ramp_up(_b, th, r, t, d, s, s_):
            if (th, r) in m.i_r and th in m.th and (m.hours.ord(s_) == m.hours.ord(s) - 1):
                return _b.P[th, r, t, d, s] - _b.P[th, r, t, d, s_] <= m.Ru_max[th] * m.hs * \
                       m.Qg_np[th, r] * (_b.u[th, r, t, d, s] - _b.su[th, r, t, d, s]) + \
                       max(m.Pg_min[th], m.Ru_max[th] * m.hs) * m.Qg_np[th, r] * _b.su[th, r, t, d, s]
            return Constraint.Skip

        b.ramp_up = Constraint(m.th, m.r, t_per_stage[stage], m.d, m.hours, m.hours, rule=ramp_up)

        def ramp_down(_b, th, r, t, d, s, s_):
            if m.hours.ord(s_) == m.hours.ord(s) - 1:
                return _b.P[th, r, t, d, s_] - _b.P[th, r, t, d, s] <= m.Rd_max[th] * m.hs * \
                       m.Qg_np[th, r] * (_b.u[th, r, t, d, s] - _b.su[th, r, t, d, s]) + \
                       max(m.Pg_min[th], m.Rd_max[th] * m.hs) * m.Qg_np[th, r] * _b.sd[th, r, t, d, s]
            return Constraint.Skip

        b.ramp_down = Constraint(m.th_r, t_per_stage[stage], m.d, m.hours, m.hours, rule=ramp_down)

        def total_op_reserve(_b, r, t, d, s):
            return sum(_b.Q_spin[th, r, t, d, s] + _b.Q_Qstart[th, r, t, d, s] for th in m.th if (th, r) in m.th_r) \
                   >= 0.075 * m.L[r, t, d, s]

        b.total_op_reserve = Constraint(m.r, t_per_stage[stage], m.d, m.hours, rule=total_op_reserve)

        def total_spin_reserve(_b, r, t, d, s):
            return sum(_b.Q_spin[th, r, t, d, s] for th in m.th if (th, r) in m.th_r) >= 0.015 * m.L[r, t, d, s]

        b.total_spin_reserve = Constraint(m.r, t_per_stage[stage], m.d, m.hours, rule=total_spin_reserve)

        def reserve_cap_1(_b, th, r, t, d, s):
            return _b.Q_spin[th, r, t, d, s] <= m.Qg_np[th, r] * m.frac_spin[th] * _b.u[th, r, t, d, s]

        b.reserve_cap_1 = Constraint(m.th_r, t_per_stage[stage], m.d, m.hours, rule=reserve_cap_1)

        def reserve_cap_2(_b, th, r, t, d, s):
            return _b.Q_Qstart[th, r, t, d, s] <= m.Qg_np[th, r] * m.frac_Qstart[th] * \
                   (_b.ngo_th[th, r, t] - _b.u[th, r, t, d, s])

        b.reserve_cap_2 = Constraint(m.th_r, t_per_stage[stage], m.d, m.hours, rule=reserve_cap_2)

        def logic_RN_1(_b, rn, r, t):
            if t == 1:
                return _b.ngb_rn[rn, r, t] - _b.ngr_rn[rn, r, t] == _b.ngo_rn[rn, r, t] - \
                       m.Ng_old[rn, r]
            elif t != t_per_stage[stage][0]:
                return _b.ngb_rn[rn, r, t] - _b.ngr_rn[rn, r, t] == _b.ngo_rn[rn, r, t] - \
                       _b.ngo_rn[rn, r, t - 1]
            else:
                return _b.ngb_rn[rn, r, t] - _b.ngr_rn[rn, r, t] == _b.ngo_rn[rn, r, t] - \
                       _b.ngo_rn_prev[rn, r]

        b.logic_RN_1 = Constraint(m.rn_r, t_per_stage[stage], rule=logic_RN_1)

        def logic_TH_1(_b, th, r, t):
            if t == 1:
                return _b.ngb_th[th, r, t] - _b.ngr_th[th, r, t] == _b.ngo_th[th, r, t] - \
                       m.Ng_old[th, r]
            elif t != t_per_stage[stage][0]:
                return _b.ngb_th[th, r, t] - _b.ngr_th[th, r, t] == _b.ngo_th[th, r, t] - \
                       _b.ngo_th[th, r, t - 1]
            else:
                return _b.ngb_th[th, r, t] - _b.ngr_th[th, r, t] == _b.ngo_th[th, r, t] - \
                       _b.ngo_th_prev[th, r]

        b.logic_TH_1 = Constraint(m.th_r, t_per_stage[stage], rule=logic_TH_1)

        def logic_ROld_2(_b, rold, r, t):
            if rold in m.rold:
                return _b.ngr_rn[rold, r, t] + _b.nge_rn[rold, r, t] == m.Ng_r[rold, r, t]
            return Constraint.Skip

        b.logic_ROld_2 = Constraint(m.rn_r, t_per_stage[stage], rule=logic_ROld_2)

        def logic_TOld_2(_b, told, r, t):
            if told in m.told:
                return _b.ngr_th[told, r, t] + _b.nge_th[told, r, t] == m.Ng_r[told, r, t]
            return Constraint.Skip

        b.logic_TOld_2 = Constraint(m.th_r, t_per_stage[stage], rule=logic_TOld_2)

        def logic_3(_b, th, r, t, d, s):
            return _b.u[th, r, t, d, s] <= _b.ngo_th[th, r, t]

        b.logic_3 = Constraint(m.th_r, t_per_stage[stage], m.d, m.hours, rule=logic_3)

        # Storage constraints
        def storage_units_balance(_b, j, r, t):
            if t == 1:
                return _b.nsb[j, r, t] - _b.nsr[j, r, t] == _b.nso[j, r, t]
            elif t != t_per_stage[stage][0]:
                return _b.nsb[j, r, t] - _b.nsr[j, r, t] == _b.nso[j, r, t] - _b.nso[j, r, t - 1]
            else:
                return _b.nsb[j, r, t] - _b.nsr[j, r, t] == _b.nso[j, r, t] - _b.nso_prev[j, r]

        b.storage_units_balance = Constraint(m.j, m.r, t_per_stage[stage], rule=storage_units_balance)

        def min_charge(_b, j, r, t, d, s):
            return m.P_min_charge[j] * _b.nso[j, r, t] <= _b.p_charged[j, r, t, d, s]

        b.min_charge = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, rule=min_charge)

        def max_charge(_b, j, r, t, d, s):
            return _b.p_charged[j, r, t, d, s] <= m.P_max_charge[j] * _b.nso[j, r, t]

        b.max_charge = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, rule=max_charge)

        def min_discharge(_b, j, r, t, d, s):
            return m.P_min_discharge[j] * _b.nso[j, r, t] <= _b.p_discharged[j, r, t, d, s]

        b.min_discharge = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, rule=min_discharge)

        def max_discharge(_b, j, r, t, d, s):
            return _b.p_discharged[j, r, t, d, s] <= m.P_max_discharge[j] * _b.nso[j, r, t]

        b.max_discharge = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, rule=max_discharge)

        def min_storage_level(_b, j, r, t, d, s):
            return m.min_storage_cap[j] * _b.nso[j, r, t] <= _b.p_storage_level[j, r, t, d, s]

        b.min_storage_level = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, rule=min_storage_level)

        def max_storage_level(_b, j, r, t, d, s):
            return _b.p_storage_level[j, r, t, d, s] <= m.max_storage_cap[j] * _b.nso[j, r, t]

        b.max_storage_level = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, rule=max_storage_level)

        def storage_balance(_b, j, r, t, d, s, s_):
            if m.hours.ord(s_) == m.hours.ord(s) - 1 and m.hours.ord(s) > 1:
                return _b.p_storage_level[j, r, t, d, s] == _b.p_storage_level[j, r, t, d, s_] \
                       + m.eff_rate_charge[j] * _b.p_charged[j, r, t, d, s] \
                       - (1 / m.eff_rate_discharge[j]) * _b.p_discharged[j, r, t, d, s]
            return Constraint.Skip

        b.storage_balance = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, m.hours, rule=storage_balance)

        def storage_balance_1HR(_b, j, r, t, d, s):
            if m.hours.ord(s) == 1:
                return _b.p_storage_level[j, r, t, d, s] == 0.5 * m.max_storage_cap[j] * _b.nso[j, r, t] \
                       + m.eff_rate_charge[j] * _b.p_charged[j, r, t, d, s] \
                       - (1 / m.eff_rate_discharge[j]) * _b.p_discharged[j, r, t, d, s]
            return Constraint.Skip

        b.storage_balance_1HR = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, rule=storage_balance_1HR)

        def storage_heuristic(_b, j, r, t, d, s):
            if m.hours.ord(s) == m.hours.last():
                return _b.p_storage_level_end_hour[j, r, t, d, s] == 0.5 * m.max_storage_cap[j] * _b.nso[j, r, t]
            return Constraint.Skip

        b.storage_heuristic = Constraint(m.j, m.r, t_per_stage[stage], m.d, m.hours, rule=storage_heuristic)

        b.link_equal1 = ConstraintList()

        b.link_equal2 = ConstraintList()

        b.link_equal3 = ConstraintList()

        b.fut_cost = ConstraintList()

    m.Bl = Block(m.stages, rule=planning_block_rule)

    return m
