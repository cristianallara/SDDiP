import sqlite3 as sql
import os.path


def read_data(database_file, stages, n_stage):
    # print(os.path.exists(database_file))
    # print(database_file)
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
    L_max_scenario = {'L': L_max, 'M': {t: 1.05 * L_max[t] for t in L_max}, 'H': {t: 1.2 * L_max[t] for t in L_max}}
    # print(L_max_scenario)

    L_max_s = {}
    for t in stages:
        for n in n_stage[t]:
            if t == 1:
                L_max_s[t, n] = L_max_scenario['M'][t]
            else:
                m = n[-1]
                if m == 'L':
                    L_max_s[t, n] = L_max_scenario['L'][t]
                elif m == 'M':
                    L_max_s[t, n] = L_max_scenario['M'][t]
                elif m == 'H':
                    L_max_s[t, n] = L_max_scenario['H'][t]
    globals()["L_max_s"] = L_max_s

    # Different scenarios for carbon tax:
    # tx_CO2_scenario = {'L': 0, 'M': 0.025, 'H': 0.050}
    #
    # tx_CO2 = {}
    # for t in stages:
    #     for n in n_stage[t]:
    #         if t == 1:
    #             tx_CO2[t, n] = 0
    #         else:
    #             m = n[-1]
    #             if m == 'L':
    #                 tx_CO2[t, n] = tx_CO2_scenario['L']
    #             elif m == 'M':
    #                 tx_CO2[t, n] = tx_CO2_scenario['M']
    #             elif m == 'H':
    #                 tx_CO2[t, n] = tx_CO2_scenario['H']
    # globals()["tx_CO2"] = tx_CO2

    print('finished loading data')


