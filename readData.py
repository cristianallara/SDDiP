import sqlite3 as sql
import os.path


def read_data(database_file, stages, n_stage, t_per_stage):
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

    ####################################################################################################################
    # Storage data

    # Lithium-ion battery (utility) in $/MW -> c-rate 1/1.2 h
    # Lead-acid battery (residency) in $/MW -> c-rate 1/3 h
    # Redox-Flow battery (residency) in $/MW -> c-rate 1/3 h
    # source: https://www.nature.com/articles/nenergy2017110.pdf

    storage_inv_cost = {('Li_ion', 1): 1637618.014, ('Li_ion', 2): 1350671.054, ('Li_ion', 3): 1203144.473,
                        ('Li_ion', 4): 1099390.353, ('Li_ion', 5): 1017197.661, #('Li_ion', 6): 948004.5483,
                        # ('Li_ion', 7): 887689.7192, ('Li_ion', 8): 834006.1087, ('Li_ion', 9): 785632.3346,
                        # ('Li_ion', 10): 741754.6519, ('Li_ion', 11): 701857.294, ('Li_ion', 12): 665604.5808,
                        # ('Li_ion', 13): 632766.397, ('Li_ion', 14): 603165.7101, ('Li_ion', 15): 576639.6204,
                        # ('Li_ion', 16): 553012.1704, ('Li_ion', 17): 532079.791, ('Li_ion', 18): 513609.5149,
                        # ('Li_ion', 19): 497347.4123, ('Li_ion', 20): 483032.4302, ('Li_ion', 21): 470410.5636,
                        # ('Li_ion', 22): 459245.8805, ('Li_ion', 23): 449327.115, ('Li_ion', 24): 440470.1878,
                        # ('Li_ion', 25): 432517.7752, ('Li_ion', 26): 425337.1293, ('Li_ion', 27): 418817.0939,
                        # ('Li_ion', 28): 412864.9434, ('Li_ion', 29): 407403.3945, ('Li_ion', 30): 402367.9548,
                        ('Lead_acid', 1): 4346125.294, ('Lead_acid', 2): 3857990.578, ('Lead_acid', 3): 3458901.946,
                        ('Lead_acid', 4): 3117666.824, ('Lead_acid', 5): 2818863.27, #('Lead_acid', 6): 2553828.021,
                        # ('Lead_acid', 7): 2317228.867, ('Lead_acid', 8): 2105569.661, ('Lead_acid', 9): 1916483.132,
                        # ('Lead_acid', 10): 1748369.467, ('Lead_acid', 11): 1600168.567, ('Lead_acid', 12): 1471137.002,
                        # ('Lead_acid', 13): 1360557.098, ('Lead_acid', 14): 1267402.114, ('Lead_acid', 15): 1190102.412,
                        # ('Lead_acid', 16): 1126569.481, ('Lead_acid', 17): 1074464.42, ('Lead_acid', 18): 1031526.418,
                        # ('Lead_acid', 19): 995794.3254, ('Lead_acid', 20): 965683.7645, ('Lead_acid', 21): 939966.8799,
                        # ('Lead_acid', 22): 917711.3511, ('Lead_acid', 23): 898213.752, ('Lead_acid', 24): 880942.0531,
                        # ('Lead_acid', 25): 865490.7549, ('Lead_acid', 26): 851547.5121, ('Lead_acid', 27): 838868.8383,
                        # ('Lead_acid', 28): 827262.5533, ('Lead_acid', 29): 816575.0934, ('Lead_acid', 30): 806682.2877,
                        ('Flow', 1): 4706872.908, ('Flow', 2): 3218220.336, ('Flow', 3): 2810526.973,
                        ('Flow', 4): 2555010.035, ('Flow', 5): 2362062.488#, ('Flow', 6): 2203531.648,
                        # ('Flow', 7): 2067165.77, ('Flow', 8): 1946678.078, ('Flow', 9): 1838520.24,
                        # ('Flow', 10): 1740573.662, ('Flow', 11): 1651531.463, ('Flow', 12): 1570567.635,
                        # ('Flow', 13): 1497136.957, ('Flow', 14): 1430839.31, ('Flow', 15): 1371321.436,
                        # ('Flow', 16): 1318208.673, ('Flow', 17): 1271067.144, ('Flow', 18): 1229396.193,
                        # ('Flow', 19): 1192645.164, ('Flow', 20): 1160243.518, ('Flow', 21): 1131632.602,
                        # ('Flow', 22): 1106290.889, ('Flow', 23): 1083749.479, ('Flow', 24): 1063598.466,
                        # ('Flow', 25): 1045486.576, ('Flow', 26): 1029116.738, ('Flow', 27): 1014239.726,
                        # ('Flow', 28): 1000647.301, ('Flow', 29): 988165.6602, ('Flow', 30): 976649.5874
                        }
    globals()["storage_inv_cost"] = storage_inv_cost

    # Power rating [MW]
    P_min_charge = {'Li_ion': 0, 'Lead_acid': 0, 'Flow': 0}
    globals()["P_min_charge"] = P_min_charge
    P_max_charge = {'Li_ion': 40, 'Lead_acid': 36, 'Flow': 2}
    globals()["P_max_charge"] = P_max_charge
    P_min_discharge = {'Li_ion': 0, 'Lead_acid': 0, 'Flow': 0}
    globals()["P_min_discharge"] = P_min_discharge
    P_max_discharge = {'Li_ion': 40, 'Lead_acid': 36, 'Flow': 2}
    globals()["P_max_discharge"] = P_max_discharge

    # Rated energy capacity [MWh]
    min_storage_cap = {'Li_ion': 0, 'Lead_acid': 0, 'Flow': 0}
    globals()["min_storage_cap"] = min_storage_cap
    max_storage_cap = {'Li_ion': 20, 'Lead_acid': 24, 'Flow': 12}
    globals()["max_storage_cap"] = max_storage_cap

    # Charge efficiency (fraction)
    eff_rate_charge = {'Li_ion': 0.95, 'Lead_acid': 0.85, 'Flow': 0.75}
    globals()["eff_rate_charge"] = eff_rate_charge
    eff_rate_discharge = {'Li_ion': 0.85, 'Lead_acid': 0.85, 'Flow': 0.75}
    globals()["eff_rate_discharge"] = eff_rate_discharge

    # Storage lifetime (years)
    storage_lifetime = {'Li_ion': 15, 'Lead_acid': 15, 'Flow': 20}
    globals()["storage_lifetime"] = storage_lifetime

    ####################################################################################################################

    # Different scenarios for peak load:
    L_max_scenario = {'L': L_max, 'M': {t: 1.05 * L_max[t] for t in L_max}, 'H': {t: 1.2 * L_max[t] for t in L_max}}
    # print(L_max_scenario)

    L_max_s = {}
    for stage in stages:
        for n in n_stage[stage]:
            for t in t_per_stage[stage]:
                if stage == 1:
                    L_max_s[t, stage, n] = L_max_scenario['M'][t]
                else:
                    m = n[-1]
                    if m == 'L':
                        L_max_s[t, stage, n] = L_max_scenario['L'][t]
                    elif m == 'M':
                        L_max_s[t, stage, n] = L_max_scenario['M'][t]
                    elif m == 'H':
                        L_max_s[t, stage, n] = L_max_scenario['H'][t]
    globals()["L_max_s"] = L_max_s
    # print(L_max_s)

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


