import sqlite3 as sql


def read_data(database_file, stages, n_stage):
    conn = sql.connect(database_file)
    c = conn.cursor()

    def process_param(sql_param):
        d = dict()
        for row in c.execute("SELECT * FROM {}".format(sql_param)):
            p_len = len(row)
            if p_len == 2:
                try:
                    d[int(row[0])] = row[1]
                except ValueError:
                    d[str(row[0])] = row[1]
            elif p_len == 3:
                try:
                    d[str(row[0]), int(row[1])] = row[2]
                except ValueError:
                    d[str(row[0]), str(row[1])] = row[2]
            elif p_len == 4:
                try:
                    d[str(row[0]), str(row[1]), int(row[2])] = row[3]
                except ValueError:
                    d[str(row[0]), str(row[1]), str(row[2])] = row[3]
            elif p_len == 5:
                try:
                    d[str(row[0]), int(row[1]), str(row[2]), str(row[3])] = row[4]
                except ValueError:
                    d[str(row[0]), str(row[1]), str(row[2]), str(row[3])] = row[4]
            elif p_len == 6:
                try:
                    d[str(row[0]), str(row[1]), int(row[2]), str(row[3]), str(row[4])] = row[5]
                except ValueError:
                    d[str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])] = row[5]
        return d

    params = ['L', 'n_ss', 'L_max', 'cf', 'Qg_np', 'Ng_old', 'Ng_max', 'Qinst_UB', 'LT', 'Tremain', 'Ng_r', 'q_v',
              'Pg_min', 'Ru_max', 'Rd_max', 'f_start', 'C_start', 'frac_spin', 'frac_Qstart', 't_loss', 't_up', 'dist',
              'if_', 'ED', 'Rmin', 'hr', 'P_fuel', 'EF_CO2', 'FOC', 'VOC', 'CCm', 'DIC', 'LEC', 'PEN', 'tx_CO2',
              'RES_min']

    for p in params:
        globals()[p] = process_param(p)

    conn.close()

    globals()['hs'] = 1
    globals()['ir'] = 0.057
    globals()['PENc'] = 0
    t_up['West', 'Coastal'] = 0
    t_up['Coastal', 'West'] = 0
    t_up['Coastal', 'Panhandle'] = 0
    t_up['South', 'Panhandle'] = 0
    t_up['Panhandle', 'Coastal'] = 0
    t_up['Panhandle', 'South'] = 0

    # Different scenarios for peak load:
    L_max_scenario = {'R': L_max, 'L': {t: 0.9 * L_max[t] for t in L_max}, 'H': {t: 1.1 * L_max[t] for t in L_max}}
    # print(L_max_scenario)

    L_max_s = {}
    for t in stages:
        for n in n_stage[t]:
            if t == 1:
                L_max_s[t, n] = L_max_scenario['R'][t]
            else:
                m = n[-1]
                if m == 'L':
                    L_max_s[t, n] = L_max_scenario['L'][t]
                elif m == 'R':
                    L_max_s[t, n] = L_max_scenario['R'][t]
                elif m == 'H':
                    L_max_s[t, n] = L_max_scenario['H'][t]
    globals()["L_max_s"] = L_max_s
    print('finished loading data')


