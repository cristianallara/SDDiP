import sqlite3 as sql
from GTEP_scenario_tree_10years import *

#conn = sql.connect("GTEP_data.sqlite")
conn = sql.connect("GTEP_data_10years.db")
c = conn.cursor()

def processparam(p):
	d = dict()
	for row in c.execute("SELECT * FROM {}".format(p)):
		l =len(row)
		if l == 2:
			try:
				d[int(row[0])] = row[1]
			except ValueError:
				d[str(row[0])] = row[1]
		elif l == 3:
			try:
				d[str(row[0]), int(row[1])] = row[2]
			except ValueError:
				d[str(row[0]), str(row[1])] = row[2]
		elif l == 4:
			try:
				d[str(row[0]), str(row[1]), int(row[2])] = row[3]
			except ValueError:
				d[str(row[0]), str(row[1]), str(row[2])] = row[3]
		elif l == 5:
			try:
				d[str(row[0]), int(row[1]), str(row[2]), str(row[3])] = row[4]
			except ValueError:
				d[str(row[0]), str(row[1]), str(row[2]), str(row[3])] = row[4]
		elif l == 6:
			try:
				d[str(row[0]), str(row[1]), int(row[2]), str(row[3]), str(row[4])] = row[5]
			except ValueError:
				d[str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])] = row[5]
	return d

params = ['L', 'n_ss', 'L_max', 'cf', 'Qg_np', 'Ng_old' , 'Ng_max',
	  'Qinst_UB', 'LT', 'Tremain', 'Ng_r', 'q_v', 'Pg_min',
	  'Ru_max', 'Rd_max', 'f_start', 'C_start', 'frac_spin',
	  'frac_Qstart', 't_loss', 't_up', 'dist', 'if_', 'ED',
	  'Rmin', 'hr', 'P_fuel', 'EF_CO2', 'FOC', 'VOC', 'CCm',
	  'DIC', 'LEC', 'PEN', 'tx_CO2', 'RES_min']

for p in params:
	globals()[p]=processparam(p)

hs = 1
ir = 0.057
PENc = 0

t_up['West','Coastal']=0
t_up['Coastal', 'West']=0
t_up['Coastal', 'Panhandle']=0
t_up['South', 'Panhandle']=0
t_up['Panhandle', 'Coastal']=0
t_up['Panhandle', 'South']=0

#Different scenarios for peak load:
L_max_R = L_max
L_max_L = {}
L_max_H = {}
for t in L_max:
	L_max_L[t] = 0.9*L_max[t]
	L_max_H[t] = 1.1*L_max[t]

L_max = {}
for t in stages:
	for n in N_stage[t]:
		if t == 1:
			L_max[t,n] = L_max_R[t]
		else:
			m = n[-1]
			if m == 'L':
				L_max[t,n] = L_max_L[t]
			elif m == 'R':
				L_max[t,n] = L_max_R[t]
			elif m == 'H':
				L_max[t,n] = L_max_H[t]
print('finished loading data')

# print(L_max)
# for i in ['coal-st-old1', 'coal-igcc-new',
# 'coal-igcc-ccs-new', 'ng-ct-old', 'ng-cc-old', 'ng-st-old','ng-cc-new',
# 'ng-cc-ccs-new', 'ng-ct-new']:
# 	P_fuel[i,1,'O']=P_fuel_R[i,1] #first stage woth no uncertainty
# 	for t in range(2,6):
# 		for n in N_stage[t]:
# 			m = n[-1]
# 			if m == 'L':
# 				P_fuel[i,t,n] = P_fuel_L[i,t]
# 			elif m == 'R':
# 				P_fuel[i,t,n] = P_fuel_R[i,t]
# 			else:
# 				P_fuel[i,t,n] = P_fuel_H[i,t]
#
# for i in  ['nuc-st-old', 'nuc-st-new']:
# 	for t in stages:
# 		for n in N_stage[t]:
# 			P_fuel[i,t,n] = 0.72
