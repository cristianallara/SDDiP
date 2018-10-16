# Generation and Transmission Expansion
# IDAES project
# author: Cristiana L. Lara
# date: 02/07/2017

from pyomo.environ import *
from GTEPdata import *

m = ConcreteModel(
		doc="Generation and Transmission Expansion Planning (MILP)",
		name="GTEP")


#####################Declaration of sets############################

m.r = Set(initialize=['Northeast', 'West', 'Coastal', 'South', 'Panhandle'],
			doc="load nodes (ERCOT regions)", ordered=True)
m.i = Set(initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old',
                      'nuc-st-old', 'pv-old', 'wind-old', 'nuc-st-new', 'wind-new',
		      'pv-new', 'csp-new', 'coal-igcc-new', 'coal-igcc-ccs-new',
		      'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new'],
			doc="generation clusters", ordered=True)
m.th = Set(within=m.i,
	   initialize=['nuc-st-old', 'nuc-st-new', 'coal-st-old1', 'coal-igcc-new',
                        'coal-igcc-ccs-new', 'ng-ct-old', 'ng-cc-old', 'ng-st-old',
						'ng-cc-new','ng-cc-ccs-new', 'ng-ct-new'],
	   doc="thermal clusters", ordered=True)
m.rn = Set(within=m.i,
	   initialize=['pv-old', 'pv-new', 'csp-new', 'wind-old', 'wind-new'],
	   doc="renewable clusters", ordered=True)
m.co = Set(within=m.th,
	   initialize=['coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new'],
	   doc="coal clusters", ordered=True)
m.ng = Set(within=m.th,
	   initialize=['ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new',
						'ng-cc-ccs-new', 'ng-ct-new'],
	   doc="natural gas clusters", ordered=True)
m.nu = Set(within=m.th,
	   initialize=['nuc-st-old', 'nuc-st-new'],
			doc="nuclear clusters", ordered=True)
m.pv = Set(within=m.rn,
	   initialize=['pv-old', 'pv-new'],
	   doc="solar PV clusters", ordered=True)
m.csp = Set(within=m.rn,
	    initialize=['csp-new'],
	    doc="solar CSP clusters", ordered=True)
m.wi = Set(within=m.rn,
	   initialize=['wind-old', 'wind-new'],
			doc="wind clusters", ordered=True)
m.old = Set(within=m.i,
	    initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old',
                        'nuc-st-old', 'pv-old', 'wind-old'],
			doc="existing clusters", ordered=True)
m.new = Set(within=m.i,
	    initialize=['nuc-st-new', 'wind-new', 'pv-new', 'csp-new', 'coal-igcc-new',
			'coal-igcc-ccs-new', 'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new'],
	    doc="potential clusters", ordered=True)
m.rold = Set(within=m.old,
	     initialize=['pv-old', 'wind-old'],
	     doc="existing renewable clusters", ordered=True)
m.rnew = Set(within=m.new,
 	     initialize=['wind-new', 'pv-new', 'csp-new'],
	     doc="potential renewable clusters", ordered=True)
m.told = Set(within=m.old,
 	     initialize=['nuc-st-old', 'coal-st-old1', 'ng-ct-old', 'ng-cc-old',
			 'ng-st-old'],
	     doc="existing thermal clusters", ordered=True)
m.tnew = Set(within=m.new,
 	     initialize=['nuc-st-new', 'coal-igcc-new','coal-igcc-ccs-new',
			 'ng-cc-new','ng-cc-ccs-new', 'ng-ct-new'],
	     doc="potential thermal clusters", ordered=True)

m.t = RangeSet(5)

	  doc="time period (years)", ordered=True)
m.d = Set(initialize=['spring', 'summer', 'fall', 'winter'],
	   doc="season", ordered=True)
m.s = Set(initialize=['1:00','2:00','3:00','4:00','5:00','6:00','7:00','8:00',
                      '9:00','10:00','11:00','12:00','13:00','14:00','15:00',
                      '16:00','17:00','18:00','19:00','20:00','21:00','22:00',
                      '23:00','24:00'],
	  doc="subperiods", ordered=True)


####################Import parameters#########################
m.L = Param(m.r,m.t,m.ss,m.s, default=0, initialize=L, doc="load demand (MW)")
m.n_ss = Param(m.ss, default=0, initialize=n_ss, doc="days per season")
m.L_max = Param(m.t, default=0, initialize=L_max, doc="peak load in year t (MW)")
m.cf = Param(m.i, m.r, m.t, m.ss, m.s, default=0, initialize=cf, doc="capacity factor")
m.Qg_np = Param(m.i, m.r, default=0, initialize=Qg_np, doc="nameplate capacity (MW)")
m.Ng_old = Param(m.i, m.r, default=0, initialize=Ng_old, doc="number of existing generators \
				in each cluster")
m.Ng_max = Param(m.i, m.r, default=0, initialize=Ng_max, doc="maximum number of generators in \
				each potential cluster")
m.Qinst_UB = Param(m.i, m.t, default=0, initialize=Qinst_UB, doc="upper bound on the yearly \
				installation (MW/year)")
m.LT = Param(m.i, initialize=LT, default=0, doc="expected lifetime (years)")
m.Tremain = Param(m.t, default=0, initialize=Tremain, doc="remaining time in years")
m.Ng_r = Param(m.old, m.r, m.t, default=0, initialize=Ng_r, doc="# generators to retired \
				at year t in cluster i")
m.q_v = Param(m.i, default=0, initialize=q_v, doc="capacity value (fraction)")
m.Pg_min = Param(m.i, default=0, initialize=Pg_min, doc="minimum generation (% nameplate)")
m.Ru_max = Param(m.i, default=0, initialize=Ru_max, doc="maximum ramp-up rate (%nameplate/hour)")
m.Rd_max = Param(m.i, default=0, initialize=Rd_max, doc="maximum ramp-down rate (%nameplate/hour)")
m.f_start = Param(m.i, default=0, initialize=f_start, doc="fuel usage at startup (MMBtu/MW)")
m.C_start = Param(m.i, default=0, initialize=C_start, doc="fixed startup cost ($/MW)")
m.frac_spin = Param(m.i, default=0, initialize=frac_spin, doc="maximum spinning reserve contribution \
				(fraction)")
m.frac_Qstart = Param(m.i, default=0, initialize=frac_Qstart, doc="maximum quick-start reserve contribution \
				(fraction)")
m.t_loss = Param(m.r, m.r, default=0, initialize=t_loss, doc="transmission loss between nodes (%/mile)")
m.t_up = Param(m.r, m.r, default=0, initialize=t_up, doc="transmission limit (MW)")
m.dist = Param(m.r, m.r, default=0, initialize=dist, doc="distance between regions (miles)")
m.if_ = Param(m.t, default=0, initialize=if_, doc="interest factor at year t")
m.ED = Param(m.t, default=0, initialize=ED, doc="energy demand at year t (MWh)")
m.Rmin = Param(m.t, default=0, initialize=Rmin, doc="system minimum reserve margin for year t \
				(fraction)")
m.hr = Param(m.i, m.r, default=0, initialize=hr, doc="heat rate (MMBtu/MWh)")
m.P_fuel = Param(m.i, m.t, default=0, initialize=P_fuel, doc="price of fuel ($/MMBtu)")
m.EF_CO2 = Param(m.i, default=0, initialize=EF_CO2,  doc="full lyfecycle emission factor (kg CO2/MMBtu)")
m.FOC = Param(m.i, m.t, default=0, initialize=FOC, doc="fixed operating cost ($/MW)")
m.VOC = Param(m.i, m.t, default=0, initialize=VOC, doc="variable operating cost ($/MWh)")
m.CCm = Param(m.i, default=0, initialize=CCm, doc="capital cost multiplier (unitless)")
m.DIC = Param(m.i, m.t, default=0, initialize=DIC, doc="discounted investment cost ($/MW)")
m.LEC = Param(m.i, default=0, initialize=LEC, doc="life extension cost (% DIC of corresponding \
				new generator)")
m.PEN = Param(m.t, default=0, initialize=PEN, doc="penalty for not meeting the renewable quota \
				target ($/MWh)")
m.PENc = Param(default=0, initialize=PENc, doc="penalty for curtailment")
m.tx_CO2 = Param(m.t, default=0, initialize=tx_CO2, doc="carbon tax in year t ($/kg CO2)")
m.RES_min = Param(m.t, default=0, initialize=RES_min, doc="minimum renewable generation \
				requirement for year t")
m.hs = Param(initialize=hs, default=0, doc="durating of subperiod s (hour)")
m.ir = Param(initialize=ir, default=0, doc="nominal interest rate")

def i_r_filter(m,i,r):
    return i in m.new or (i in m.old and m.Ng_old[i,r] != 0)
m.i_r = Set(initialize=m.i*m.r, filter=i_r_filter)
#m.i_r.pprint()

####################Declaration of variables#########################

#Bounds
def bound_P(m,i,r,t,ss,s):
	if i in m.old:
		return (0, m.Qg_np[i,r]*m.Ng_old[i,r])
	else:
		return (0, m.Qg_np[i,r]*m.Ng_max[i,r])

m.P = Var(m.i, m.r, m.t, m.ss, m.s, within=NonNegativeReals, bounds=bound_P,
	  doc="power output (MW)")
m.cu = Var(m.r, m.t, m.ss, m.s, within=NonNegativeReals,
	   doc="curtailment slack in reagion r (MW)")
m.RES_def = Var(m.t, within=NonNegativeReals,
                doc="deficit from renewable quota target (MW)")

def bound_Pflow(m,r,r_,t,ss,s):
    if r_ != r:
        return (0, t_up[r,r_])
    else:
        return (0,0)

m.P_flow = Var(m.r, m.r, m.t, m.ss, m.s, within=NonNegativeReals, bounds=bound_Pflow,
	       doc="power flow from one region to the other (MW)")
#m.P_flow.fix(0.0)
m.Q_spin = Var(m.th, m.r, m.t, m.ss, m.s, within=NonNegativeReals,
	       doc="spinning reserve (MW)")
m.Q_Qstart = Var(m.th, m.r, m.t, m.ss, m.s, within=NonNegativeReals,
		 doc="quick-start reserve (MW)")

#Bounds
def bound_o_rn(m,rn,r,t):
	if rn in m.rold:
		return (0, m.Ng_old[rn,r])
	else:
		return (0, m.Ng_max[rn,r])

def bound_b_rn(m,rn,r,t):
	if rn in m.rold:
		return (0, 0)
	else:
		return (0, floor(m.Qinst_UB[rn,t]/m.Qg_np[rn,r]))

def bound_r_rn(m,rn,r,t):
	if rn in m.rold:
		return (0, m.Ng_old[rn,r])
	else:
		return (0, m.Ng_max[rn,r])

def bound_e_rn(m,rn,r,t):
	if rn in m.rold:
		return (0, m.Ng_r[rn,r,t])
	else:
		return (0, m.Ng_max[rn,r])

m.ngo_rn = Var(m.rn, m.r, m.t, bounds=bound_o_rn, domain= NonNegativeReals,
			doc="number of renewable generaters that are operational \
				in cluster i, region r, year t")
m.ngb_rn = Var(m.rn, m.r, m.t, bounds=bound_b_rn, domain= NonNegativeReals,
			doc="number of renewable generaters that are built \
				in cluster i, region r, year t")
for rold, r, t in m.rold * m.r * m.t:
    m.ngb_rn[rold,r,t].fix(0.0)

m.ngr_rn = Var(m.rn, m.r, m.t, bounds=bound_r_rn, domain= NonNegativeReals,
			doc="number of renewable generaters that are retired \
				in cluster i, region r, year t")
for rnew, r, t  in  m.rnew * m.r * m.t:
    if t == 1:
        m.ngr_rn[rnew,r,t].fix(0.0)
    else:
        m.ngr_rn[rnew,r,t].unfix()


m.nge_rn = Var(m.rn, m.r, m.t, bounds=bound_e_rn, domain= NonNegativeReals,
			doc="number of renewable generaters that had their life \
				extended in cluster i, region r, year t")
for rnew, r, t in m.rnew * m.r * m.t:
    if t == 1:
        m.nge_rn[rnew,r,t].fix(0.0)
    else:
        m.nge_rn[rnew,r,t].unfix()

#Bounds
def bound_o_th(m,th,r,t):
	if th in m.told:
		return (0, m.Ng_old[th,r])
	else:
		return (0, m.Ng_max[th,r])

def bound_b_th(m,th,r,t):
	if th in m.told:
		return (0, 0)
	else:
		return (0, floor(m.Qinst_UB[th,t]/m.Qg_np[th,r]))

def bound_r_th(m,th,r,t):
	if th in m.told:
		return (0, m.Ng_old[th,r])
	else:
		return (0, m.Ng_max[th,r])

def bound_e_th(m,th,r,t):
	if th in m.told:
		return (0, m.Ng_r[th,r,t])
	else:
		return (0, m.Ng_max[th,r])

m.ngo_th = Var(m.th, m.r, m.t, bounds=bound_o_th, domain=NonNegativeIntegers,
			doc="number of thermal generaters that are operational \
				in cluster i, region r, year t")

m.ngb_th = Var(m.th, m.r, m.t, bounds=bound_b_th, domain=NonNegativeIntegers,
			doc="number of thermal generaters that are built \
				in cluster i, region r, year t")
for told, r, t in  m.told * m.r * m.t:
    m.ngb_th[told,r,t].fix(0.0)

m.ngr_th = Var(m.th, m.r, m.t, bounds=bound_r_th, domain=NonNegativeIntegers,
			doc="number of thermal generaters that are retired \
				in cluster i, region r, year t")
for tnew, r, t in m.tnew * m.r * m.t:
    if t == 1:
        m.ngr_th[tnew,r,t].fix(0.0)
    else:
        m.ngr_th[tnew,r,t].unfix()

m.nge_th = Var(m.th, m.r, m.t, bounds=bound_e_th, domain=NonNegativeIntegers,
			doc="number of thermal generaters that had their life \
				extended in cluster i, region r, year t")
for tnew, r, t in m.tnew * m.r * m.t:
    if t==1:
        m.nge_th[tnew,r,t].fix(0.0)
    else:
        m.nge_th[tnew,r,t].unfix()


#Bounds
def bound_UC(m,th,r,t,ss,s):
	if th in m.told:
		return (0, m.Ng_old[th,r])
	else:
		return (0, m.Ng_max[th,r])

m.u = Var(m.th, m.r, m.t, m.ss, m.s, bounds=bound_UC, domain=NonNegativeIntegers,
			 doc="unit commitment status")
m.su = Var(m.th, m.r, m.t, m.ss, m.s, bounds=bound_UC, domain=NonNegativeIntegers,
			doc="startup indicator")
m.sd = Var(m.th, m.r, m.t, m.ss, m.s, bounds=bound_UC, domain=NonNegativeIntegers,
			doc="shutdown indicator")

for th,r,t,ss,s in m.th*m.r*m.t*m.ss*m.s:
    if m.s.ord(s) == 1:
        m.sd[th,r,t,ss,s].fix(0.0)
    else:
        m.sd[th,r,t,ss,s].unfix()


##########################Equations#################################

#Objective function
def obj_rule(m):
	return sum(m.if_[t]*
			(sum(m.n_ss[ss]*m.hs* \
                            sum((m.VOC[i,t] + m.hr[i,r]*m.P_fuel[i,t] \
				+ m.EF_CO2[i]*m.tx_CO2[t]*m.hr[i,r])*m.P[i,r,t,ss,s] \
                                for i,r in m.i_r) \
                            for ss in m.ss for s in m.s) \
			+ sum(m.FOC[rn,t]*m.Qg_np[rn,r]*m.ngo_rn[rn,r,t] \
			    for rn,r in m.i_r if rn in m.rn) \
			+ sum(m.FOC[th,t]*m.Qg_np[th,r]*m.ngo_th[th,r,t] \
			    for th,r in m.i_r if th in m.th) \
			+ sum(m.n_ss[ss]*m.hs*m.su[th,r,t,ss,s]*m.Qg_np[th,r] \
				*(m.f_start[th]*m.P_fuel[th,t] + m.f_start[th]*m.EF_CO2[th]*m.tx_CO2[t] \
                                  + m.C_start[th]) \
                            for th,r in m.i_r if th in m.th for ss in m.ss for s in m.s) \
			+ sum(m.DIC[rnew,t]*m.CCm[rnew]*m.Qg_np[rnew,r]*m.ngb_rn[rnew,r,t] \
			    for rnew,r in m.i_r if rnew in m.rnew) \
			+ sum(m.DIC[tnew,t]*m.CCm[tnew]*m.Qg_np[tnew,r]*m.ngb_th[tnew,r,t] \
			    for tnew,r in m.i_r if tnew in m.tnew) \
			+ sum(m.DIC[rn,t]*m.LEC[rn]*m.Qg_np[rn,r]*m.nge_rn[rn,r,t] \
			    for rn,r in m.i_r if rn in m.rn) \
			+ sum(m.DIC[th,t]*m.LEC[th]*m.Qg_np[th,r]*m.nge_th[th,r,t] \
			    for th,r in m.i_r if th in m.th) \
			+ m.PEN[t]*m.RES_def[t] \
			+ m.PENc*sum(m.cu[r,t,ss,s] for r in m.r for ss in m.ss for s in m.s)) \
		   for t in m.t) \
		* 10**(-9)
m.obj = Objective(rule=obj_rule, sense=minimize)

# Investment Constraints
def min_RN_req(m,t):
	return sum(m.n_ss[ss]*m.hs*sum(m.P[rn,r,t,ss,s]-m.cu[r,t,ss,s] \
		                       for rn,r in m.i_r if rn in m.rn) \
                   for ss in m.ss for s in m.s) + m.RES_def[t] >= m.RES_min[t]*m.ED[t]
m.min_RN_req = Constraint(m.t, rule=min_RN_req)

def min_reserve(m,t):
	return sum(m.Qg_np[rn,r]*m.ngo_rn[rn,r,t]*m.q_v[rn] for rn,r in m.i_r if rn in m.rn) \
	     + sum(m.Qg_np[th,r]*m.ngo_th[th,r,t] for th,r in m.i_r if th in m.th) \
			>= (1 + m.Rmin[t])*m.L_max[t]
m.min_reserve = Constraint(m.t, rule=min_reserve)

def inst_RN_UB(m,rnew,t):
	return sum(m.ngb_rn[rnew,r,t] for r in m.r) \
               <= m.Qinst_UB[rnew,t] / sum(m.Qg_np[rnew,r] /len(m.r) for r in m.r)
m.inst_RN_UB = Constraint(m.rnew, m.t, rule=inst_RN_UB)

def inst_TH_UB(m,tnew,t):
	return sum(m.ngb_th[tnew,r,t] for r in m.r) \
               <= m.Qinst_UB[tnew,t] / sum(m.Qg_np[tnew,r] /len(m.r) for r in m.r)
m.inst_TH_UB = Constraint(m.tnew, m.t, rule=inst_TH_UB)

#Operating Constraints

def en_bal(m,r,t,ss,s):
	return sum(m.P[i,r,t,ss,s] for i in m.i if (i,r) in m.i_r) \
             + sum(m.P_flow[r_,r,t,ss,s]*(1-m.t_loss[r,r_]*m.dist[r,r_])-m.P_flow[r,r_,t,ss,s] \
                   for r_ in m.r if r_ != r) == m.L[r,t,ss,s] + m.cu[r,t,ss,s]
m.en_bal = Constraint(m.r, m.t, m.ss, m.s, rule=en_bal)

def capfactor(m,rn,r,t,ss,s):
	if (rn,r) in m.i_r and rn in m.rn:
		return m.P[rn,r,t,ss,s] == m.Qg_np[rn,r]*m.cf[rn,r,t,ss,s] \
			*m.ngo_rn[rn,r,t]
	return Constraint.Skip
m.capfactor = Constraint(m.rn, m.r, m.t, m.ss, m.s, rule=capfactor)

def min_output(m,th,r,t,ss,s):
	if (th,r) in m.i_r and th in m.th:
		return m.u[th,r,t,ss,s]*m.Pg_min[th]*m.Qg_np[th,r] <= m.P[th,r,t,ss,s]
	return Constraint.Skip
m.min_output = Constraint(m.th,m.r,m.t,m.ss,m.s, rule=min_output)

def max_output(m,th,r,t,ss,s):
	if (th,r) in m.i_r and th in m.th:
		return m.u[th,r,t,ss,s]*m.Qg_np[th,r] >= m.P[th,r,t,ss,s] + m.Q_spin[th,r,t,ss,s]
	return Constraint.Skip
m.max_output = Constraint(m.th,m.r,m.t,m.ss,m.s, rule=max_output)

def unit_commit1(m,th,r,t,ss,s,s_):
	if (th,r) in m.i_r and th in m.th and (m.s.ord(s_) == m.s.ord(s)-1):
		return m.u[th,r,t,ss,s] == m.u[th,r,t,ss,s_] + m.su[th,r,t,ss,s] - m.sd[th,r,t,ss,s]
	return Constraint.Skip
m.unit_commit1 = Constraint(m.th,m.r,m.t,m.ss,m.s,m.s, rule=unit_commit1)

def ramp_up(m,th,r,t,ss,s,s_):
	if (th,r) in m.i_r and th in m.th and (m.s.ord(s_) == m.s.ord(s)-1):
		return m.P[th,r,t,ss,s]-m.P[th,r,t,ss,s_] <= m.Ru_max[th]*m.hs* \
				m.Qg_np[th,r]*(m.u[th,r,t,ss,s]-m.su[th,r,t,ss,s]) + \
				max(m.Pg_min[th],m.Ru_max[th]*m.hs)*m.Qg_np[th,r]*m.su[th,r,t,ss,s]
	return Constraint.Skip
m.ramp_up = Constraint(m.th,m.r,m.t,m.ss,m.s,m.s, rule=ramp_up)

def ramp_down(m,th,r,t,ss,s,s_):
	if (th,r) in m.i_r and th in m.th and (m.s.ord(s_) == m.s.ord(s)-1):
		return m.P[th,r,t,ss,s_]-m.P[th,r,t,ss,s] <= m.Rd_max[th]*m.hs* \
				m.Qg_np[th,r]*(m.u[th,r,t,ss,s]-m.su[th,r,t,ss,s]) + \
				max(m.Pg_min[th],m.Rd_max[th]*m.hs)*m.Qg_np[th,r]*m.sd[th,r,t,ss,s]
	return Constraint.Skip
m.ramp_down = Constraint(m.th,m.r,m.t,m.ss,m.s,m.s, rule=ramp_down)

def total_op_reserve(m,r,t,ss,s):
	return sum(m.Q_spin[th,r,t,ss,s]+m.Q_Qstart[th,r,t,ss,s] for th in m.th) \
			>= 0.075*m.L[r,t,ss,s]
m.total_op_reserve = Constraint(m.r,m.t,m.ss,m.s, rule=total_op_reserve)

def total_spin_reserve(m,r,t,ss,s):
	return sum(m.Q_spin[th,r,t,ss,s] for th in m.th) >= 0.015*m.L[r,t,ss,s]
m.total_spin_reserve = Constraint(m.r,m.t,m.ss,m.s, rule=total_spin_reserve)

def reserve_cap_1(m,th,r,t,ss,s):
	if (th,r) in m.i_r:
		return m.Q_spin[th,r,t,ss,s] <= m.u[th,r,t,ss,s]*m.Qg_np[th,r]*\
				m.frac_spin[th]
	return Constraint.Skip
m.reserve_cap_1 = Constraint(m.th,m.r,m.t,m.ss,m.s, rule=reserve_cap_1)

def reserve_cap_2(m,th,r,t,ss,s):
	if (th,r) in m.i_r:
		return m.Q_Qstart[th,r,t,ss,s] <= (m.ngo_th[th,r,t] - m.u[th,r,t,ss,s])* \
				m.Qg_np[th,r]*m.frac_Qstart[th]
        return Constraint.Skip
m.reserve_cap_2 = Constraint(m.th,m.r,m.t,m.ss,m.s, rule=reserve_cap_2)
m.reserve_cap_2.pprint()

def logic_RN_1a(m,rn,r,t):
	if (rn,r) in m.i_r and t == 1:
		return m.ngb_rn[rn,r,t] - m.ngr_rn[rn,r,t] == m.ngo_rn[rn,r,t] - \
					m.Ng_old[rn,r]
	return Constraint.Skip
m.logic_RN_1a = Constraint(m.rn,m.r,m.t, rule=logic_RN_1a)

def logic_RN_1b(m,rn,r,t,tt):
	if (rn,r) in m.i_r and t != 1 and tt == (t-1):
	        return m.ngb_rn[rn,r,t] - m.ngr_rn[rn,r,t] == m.ngo_rn[rn,r,t] - \
					m.ngo_rn[rn,r,tt]
	return Constraint.Skip
m.logic_RN_1b = Constraint(m.rn,m.r,m.t,m.t, rule=logic_RN_1b)

def logic_TH_1a(m,th,r,t):
	if (th,r) in m.i_r and t == 1:
	        return m.ngb_th[th,r,t] - m.ngr_th[th,r,t] == m.ngo_th[th,r,t] - \
					m.Ng_old[th,r]
	return Constraint.Skip
m.logic_TH_1a = Constraint(m.th,m.r,m.t, rule=logic_TH_1a)

def logic_TH_1b(m,th,r,t,tt):
	if (th,r) in m.i_r and t != 1 and tt == (t-1):
	        return m.ngb_th[th,r,t] - m.ngr_th[th,r,t] == m.ngo_th[th,r,t] - \
					m.ngo_th[th,r,tt]
	return Constraint.Skip
m.logic_TH_1b = Constraint(m.th,m.r,m.t,m.t, rule=logic_TH_1b)


def logic_RNew_2(m,rnew,r,t):
	if (rnew,r) in m.i_r and rnew in m.rnew:
		return sum(m.ngr_rn[rnew,r,tt]+m.nge_rn[rnew,r,tt] for tt in m.t if tt <= t) \
				== sum(m.ngb_rn[rnew,r,ttt] for ttt in m.t if ttt <= (t-m.LT[rnew]))
	return Constraint.Skip
m.logic_RNew_2 = Constraint(m.rnew,m.r,m.t, rule=logic_RNew_2)

def logic_TNew_2(m,tnew,r,t):
	if (tnew,r) in m.i_r and tnew in m.tnew:
		return sum(m.ngr_th[tnew,r,tt]+m.nge_th[tnew,r,tt] for tt in m.t if tt <= t) \
				== sum(m.ngb_th[tnew,r,ttt] for ttt in m.t if ttt <= (t-m.LT[tnew]))
	return Constraint.Skip
m.logic_TNew_2 = Constraint(m.tnew,m.r,m.t, rule=logic_TNew_2)

def logic_ROld_2(m,rold,r,t):
	if (rold,r) in m.i_r:
		return m.ngr_rn[rold,r,t]+m.nge_rn[rold,r,t] == m.Ng_r[rold,r,t]
	return Constraint.Skip
m.logic_ROld_2 = Constraint(m.rold, m.r, m.t, rule=logic_ROld_2)

def logic_TOld_2(m,told,r,t):
	if (told,r) in m.i_r:
		return m.ngr_th[told,r,t]+m.nge_th[told,r,t] == m.Ng_r[told,r,t]
	return Constraint.Skip
m.logic_TOld_2 = Constraint(m.told, m.r, m.t, rule=logic_TOld_2)

def logic_3(m,th,r,t,ss,s):
	if (th,r) in m.i_r:
		return m.u[th,r,t,ss,s] <= m.ngo_th[th,r,t]
	return Constraint.Skip
m.logic_3 = Constraint(m.th,m.r,m.t,m.ss,m.s, rule=logic_3)

#Solve the model using Gurobi
"""
mipsolver = SolverFactory('gurobi')
mipsolver.options['mipgap']=0.0
mipsolver.options['SiftMethod']=2
mipsolver.options['threads']=10
results = mipsolver.solve(m, tee=True) # solves and updates instance
results.write(num=1)
"""
#Solve the model using CPLEX
mipsolver = SolverFactory('cplex')
mipsolver.options['mipgap']=0.01
mipsolver.options['lpmethod']=4
mipsolver.options['threads']=10
results = mipsolver.solve(m, tee=True) # solves and updates instance
results.write(num=1)

#Print ngo results
#m.ngo_rn.pprint()
#m.ngo_th.pprint()

#Cost breakdown
fixOp = sum(m.if_[t]*(sum(m.FOC[rn,t]*m.Qg_np[rn,r]*m.ngo_rn[rn,r,t].value \
                      for rn,r in m.i_r if rn in m.rn) + \
                      sum(m.FOC[th,t]*m.Qg_np[th,r]*m.ngo_th[th,r,t].value \
                      for th,r in m.i_r if th in m.th)) for t in m.t)* 10**(-9)

varOp = sum(m.if_[t]*sum(m.n_ss[ss]*m.hs*m.VOC[i,t]*m.P[i,r,t,ss,s].value \
                          for i,r in m.i_r for ss in m.ss for s in m.s) \
                      for t in m.t)*10**(-9)

start = sum(m.if_[t]*sum(m.n_ss[ss]*m.hs*m.su[th,r,t,ss,s].value*m.Qg_np[th,r]\
                         *(m.f_start[th]*m.P_fuel[th,t]+m.f_start[th]*m.EF_CO2[th]*m.tx_CO2[t]+m.C_start[th]) \
                         for th,r in m.i_r if th in m.th for ss in m.ss for s in m.s) \
            for t in m.t)*10**(-9)

inv = sum( m.if_[t]*(sum(m.DIC[rnew,t]*m.CCm[rnew]*m.Qg_np[rnew,r]*m.ngb_rn[rnew,r,t].value \
                         for rnew,r in m.i_r if rnew in m.rnew) \
                    +sum(m.DIC[tnew,t]*m.CCm[tnew]*m.Qg_np[tnew,r]*m.ngb_th[tnew,r,t].value \
                         for tnew,r in m.i_r if tnew in m.tnew)) for t in m.t)*10**(-9)

ext = sum( m.if_[t]*(sum(m.DIC[rn,t]*m.LEC[rn]*m.Qg_np[rn,r]*m.nge_rn[rn,r,t].value \
                         for rn,r in m.i_r if rn in m.rn)\
                    +sum(m.DIC[th,t]*m.LEC[th]*m.Qg_np[th,r]*m.nge_th[th,r,t].value \
                         for th,r in m.i_r if th in m.th))\
                    for t in m.t)*10**(-9)

fuel = sum( m.if_[t]*sum(m.n_ss[ss]*m.hs*m.hr[i,r]*m.P_fuel[i,t]*m.P[i,r,t,ss,s].value \
                          for i,r in m.i_r for ss in m.ss for s in m.s)\
                      for t in m.t)*10**(-9)

print ("Fixed Operating Cost", fixOp)
print ("Variable Operating Cost", varOp)
print ("Startup Cost", start)
print ("Investment Cost", inv)
print ("Life Extension Cost", ext)
print ("Fuel Cost", fuel)

tot = fixOp + varOp + start + inv + fuel + ext

print ("obj", tot)

#Generation per technology
def generation(m, i, t):
    if i in m.rn:
        return sum(m.ngo_rn[i,r,t].value*m.Qg_np[i,r] for r in m.r if (i,r) in m.i_r)*10**(-3)
    else:
        return sum(m.ngo_th[i,r,t].value*m.Qg_np[i,r] for r in m.r if (i,r) in m.i_r)*10**(-3)
m.generation = Expression(m.i,m.t, rule=generation)

m.generation.display()

m.gs = Set(initialize=['coal', 'NG', 'nuc', 'solar','wind'],
	   doc="energy sources", ordered=True)

#Generation per source
def generation_source(m,gs,t):
    if gs in m.gs and gs == 'coal':
        return sum(m.generation[i,t] for i in m.co)
    elif gs in m.gs and gs == 'NG':
        return sum(m.generation[i,t] for i in m.ng)
    elif gs in m.gs and gs == 'nuc':
        return sum(m.generation[i,t] for i in m.nu)
    elif gs in m.gs and gs == 'solar':
        return sum(m.generation[i,t] for i in m.pv)
    elif gs in m.gs and gs == 'wind':
		return sum(m.generation[i,t] for i in m.wi)

m.generation_source = Expression(m.gs,m.t, rule=generation_source)
m.generation_source.display()

generation_plot = {}
for t in m.t:
	for gs in m.gs:
		generation_plot[gs,t]=m.generation_source[gs,t]


import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

years = range(2015, 2044, 1)

ng = []
coal = []
nuc = []
solar = []
wind = []

for t in m.t:
    termng = generation_plot['NG', t]
    ng.append(termng)
    termcoal = generation_plot['coal', t]
    coal.append(termcoal)
    termnuc = generation_plot['nuc', t]
    nuc.append(termnuc)
    termsolar = generation_plot['solar', t]
    solar.append(termsolar)
    termwind = generation_plot['wind', t]
    wind.append(termwind)

plt.bar(years, nuc, color = "yellow")
plt.bar(years, coal, color = "blue", bottom = [nuc[i] for i in range(len(years))])
plt.bar(years, ng, color = "red", bottom = [nuc[i] + coal[i] for i in range(len(years))])
plt.bar(years, wind, color = "green", bottom = [nuc[i] + coal[i] + ng[i] for i in range(len(years))])
plt.bar(years, solar, color = "orange", bottom = [nuc[i] + coal[i] + ng[i] + wind[i] for i in range(len(years))])
plt.xlabel("Year")
plt.ylabel("Capacity (GW)")
plt.title("Generation Capacity - Total ERCOT")
plt.legend(["Nuclear", "Coal", "Natural Gas", "Solar", "Wind"], loc = "best")
plt.ylim(0, 180)
#plt.figure(figsize=(200, 200))
plt.savefig('capacity.png')
