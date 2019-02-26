import sqlite3 as sql
import os
import pandas as pd


def read_data(database_file, curPath, stages, n_stage, t_per_stage):
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

    params = ['n_ss', 'L_max', 'Qg_np', 'Ng_old', 'Ng_max', 'Qinst_UB', 'LT', 'Tremain', 'Ng_r', 'q_v',
              'Pg_min', 'Ru_max', 'Rd_max', 'f_start', 'C_start', 'frac_spin', 'frac_Qstart', 't_loss', 't_up', 'dist',
              'if_', 'ED', 'Rmin', 'hr', 'P_fuel', 'EF_CO2', 'FOC', 'VOC', 'CCm', 'DIC', 'LEC', 'PEN', 'tx_CO2',
              'RES_min']  # , 'L', 'cf']

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
    # Storage data (Lithium-ion battery (utility) -> c-rate 1/1.2 h;
    #               Lead-acid battery (residency) -> c-rate 1/3 h;
    #               Redox-Flow battery (residency) -> c-rate 1/3 h
    # source: https://www.nature.com/articles/nenergy2017110.pdf

    # Storage Investment Cost in $/MW
    storage_inv_cost = {('Li_ion', 1): 1637618.014, ('Li_ion', 2): 1350671.054, ('Li_ion', 3): 1203144.473,
                        ('Li_ion', 4): 1099390.353, ('Li_ion', 5): 1017197.661,  # ('Li_ion', 6): 948004.5483,
                        # ('Li_ion', 7): 887689.7192, ('Li_ion', 8): 834006.1087, ('Li_ion', 9): 785632.3346,
                        # ('Li_ion', 10): 741754.6519, ('Li_ion', 11): 701857.294, ('Li_ion', 12): 665604.5808,
                        # ('Li_ion', 13): 632766.397, ('Li_ion', 14): 603165.7101, ('Li_ion', 15): 576639.6204,
                        # ('Li_ion', 16): 553012.1704, ('Li_ion', 17): 532079.791, ('Li_ion', 18): 513609.5149,
                        # ('Li_ion', 19): 497347.4123, ('Li_ion', 20): 483032.4302,
                        ('Lead_acid', 1): 4346125.294, ('Lead_acid', 2): 3857990.578, ('Lead_acid', 3): 3458901.946,
                        ('Lead_acid', 4): 3117666.824, ('Lead_acid', 5): 2818863.27,  # ('Lead_acid', 6): 2553828.021,
                        # ('Lead_acid', 7): 2317228.867, ('Lead_acid', 8): 2105569.661, ('Lead_acid', 9): 1916483.132,
                        # ('Lead_acid', 10): 1748369.467,('Lead_acid', 11): 1600168.567, ('Lead_acid', 12): 1471137.002,
                        # ('Lead_acid', 13): 1360557.098,('Lead_acid', 14): 1267402.114, ('Lead_acid', 15): 1190102.412,
                        # ('Lead_acid', 16): 1126569.481, ('Lead_acid', 17): 1074464.42, ('Lead_acid', 18): 1031526.418,
                        # ('Lead_acid', 19): 995794.3254, ('Lead_acid', 20): 965683.7645,
                        ('Flow', 1): 4706872.908, ('Flow', 2): 3218220.336, ('Flow', 3): 2810526.973,
                        ('Flow', 4): 2555010.035, ('Flow', 5): 2362062.488  # , ('Flow', 6): 2203531.648,
                        # ('Flow', 7): 2067165.77, ('Flow', 8): 1946678.078, ('Flow', 9): 1838520.24,
                        # ('Flow', 10): 1740573.662, ('Flow', 11): 1651531.463, ('Flow', 12): 1570567.635,
                        # ('Flow', 13): 1497136.957, ('Flow', 14): 1430839.31, ('Flow', 15): 1371321.436,
                        # ('Flow', 16): 1318208.673, ('Flow', 17): 1271067.144, ('Flow', 18): 1229396.193,
                        # ('Flow', 19): 1192645.164, ('Flow', 20): 1160243.518
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

    # Operational uncertainty data
    list_of_repr_days_per_scenario = ['spring', 'summer', 'fall', 'winter']
    # Misleading (seasons not used) but used because of structure of old data

    # ############ LOAD ############
    L_NE_1 = pd.read_csv(os.path.join(curPath, 'data/L_Northeast.csv'), index_col=0, header=0).iloc[:, :4]
    L_NE_1.columns = list_of_repr_days_per_scenario
    L_NE_2 = pd.read_csv(os.path.join(curPath, 'data/L_Northeast.csv'), index_col=0, header=0).iloc[:, 4:]
    L_NE_2.columns = list_of_repr_days_per_scenario
    L_W_1 = pd.read_csv(os.path.join(curPath, 'data/L_West.csv'), index_col=0, header=0).iloc[:, :4]
    L_W_1.columns = list_of_repr_days_per_scenario
    L_W_2 = pd.read_csv(os.path.join(curPath, 'data/L_West.csv'), index_col=0, header=0).iloc[:, 4:]
    L_W_2.columns = list_of_repr_days_per_scenario
    L_C_1 = pd.read_csv(os.path.join(curPath, 'data/L_Coastal.csv'), index_col=0, header=0).iloc[:, :4]
    L_C_1.columns = list_of_repr_days_per_scenario
    L_C_2 = pd.read_csv(os.path.join(curPath, 'data/L_Coastal.csv'), index_col=0, header=0).iloc[:, 4:]
    L_C_2.columns = list_of_repr_days_per_scenario
    L_S_1 = pd.read_csv(os.path.join(curPath, 'data/L_South.csv'), index_col=0, header=0).iloc[:, :4]
    L_S_1.columns = list_of_repr_days_per_scenario
    L_S_2 = pd.read_csv(os.path.join(curPath, 'data/L_South.csv'), index_col=0, header=0).iloc[:, 4:]
    L_S_2.columns = list_of_repr_days_per_scenario
    L_PH_1 = pd.read_csv(os.path.join(curPath, 'data/L_Panhandle.csv'), index_col=0, header=0).iloc[:, :4]
    L_PH_1.columns = list_of_repr_days_per_scenario
    L_PH_2 = pd.read_csv(os.path.join(curPath, 'data/L_Panhandle.csv'), index_col=0, header=0).iloc[:, 4:]
    L_PH_2.columns = list_of_repr_days_per_scenario

    L_1 = {}
    L_2 = {}
    for t in L_max:
        d_idx = 0
        for d in list_of_repr_days_per_scenario:
            s_idx = 0
            for s in list(L_NE_1.index):
                L_1['Northeast', t, d, s] = L_NE_1.iat[s_idx, d_idx]
                L_1['West', t, d, s] = L_W_1.iat[s_idx, d_idx]
                L_1['Coastal', t, d, s] = L_C_1.iat[s_idx, d_idx]
                L_1['South', t, d, s] = L_S_1.iat[s_idx, d_idx]
                L_1['Panhandle', t, d, s] = L_PH_1.iat[s_idx, d_idx]
                L_2['Northeast', t, d, s] = L_NE_2.iat[s_idx, d_idx]
                L_2['West', t, d, s] = L_W_2.iat[s_idx, d_idx]
                L_2['Coastal', t, d, s] = L_C_2.iat[s_idx, d_idx]
                L_2['South', t, d, s] = L_S_2.iat[s_idx, d_idx]
                L_2['Panhandle', t, d, s] = L_PH_2.iat[s_idx, d_idx]
                s_idx += 1
            d_idx += 1

    L_by_scenario = [L_1, L_2]
    # print(L_by_scenario)
    globals()["L_by_scenario"] = L_by_scenario

    # ############ CAPACITY FACTOR ############    
    # -> solar CSP
    CF_CSP_NE_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_Northeast.csv'), index_col=0, header=0
                              ).iloc[:, :4]
    CF_CSP_NE_1.columns = list_of_repr_days_per_scenario
    CF_CSP_NE_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_Northeast.csv'), index_col=0, header=0
                              ).iloc[:, 4:]
    CF_CSP_NE_2.columns = list_of_repr_days_per_scenario
    CF_CSP_W_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_West.csv'), index_col=0, header=0).iloc[:, :4]
    CF_CSP_W_1.columns = list_of_repr_days_per_scenario
    CF_CSP_W_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_West.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_CSP_W_2.columns = list_of_repr_days_per_scenario
    CF_CSP_C_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_Coastal.csv'), index_col=0, header=0).iloc[:, :4]
    CF_CSP_C_1.columns = list_of_repr_days_per_scenario
    CF_CSP_C_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_Coastal.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_CSP_C_2.columns = list_of_repr_days_per_scenario
    CF_CSP_S_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_South.csv'), index_col=0, header=0).iloc[:, :4]
    CF_CSP_S_1.columns = list_of_repr_days_per_scenario
    CF_CSP_S_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_South.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_CSP_S_2.columns = list_of_repr_days_per_scenario
    CF_CSP_PH_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_Panhandle.csv'), index_col=0, header=0
                              ).iloc[:, :4]
    CF_CSP_PH_1.columns = list_of_repr_days_per_scenario
    CF_CSP_PH_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_CSP_Panhandle.csv'), index_col=0, header=0
                              ).iloc[:, 4:]
    CF_CSP_PH_2.columns = list_of_repr_days_per_scenario
    
    # -> solar PVSAT
    CF_PV_NE_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_Northeast.csv'), index_col=0, header=0
                              ).iloc[:, :4]
    CF_PV_NE_1.columns = list_of_repr_days_per_scenario
    CF_PV_NE_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_Northeast.csv'), index_col=0, header=0
                              ).iloc[:, 4:]
    CF_PV_NE_2.columns = list_of_repr_days_per_scenario
    CF_PV_W_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_West.csv'), index_col=0, header=0).iloc[:, :4]
    CF_PV_W_1.columns = list_of_repr_days_per_scenario
    CF_PV_W_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_West.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_PV_W_2.columns = list_of_repr_days_per_scenario
    CF_PV_C_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_Coastal.csv'), index_col=0, header=0).iloc[:, :4]
    CF_PV_C_1.columns = list_of_repr_days_per_scenario
    CF_PV_C_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_Coastal.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_PV_C_2.columns = list_of_repr_days_per_scenario
    CF_PV_S_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_South.csv'), index_col=0, header=0).iloc[:, :4]
    CF_PV_S_1.columns = list_of_repr_days_per_scenario
    CF_PV_S_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_South.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_PV_S_2.columns = list_of_repr_days_per_scenario
    CF_PV_PH_1 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_Panhandle.csv'), index_col=0, header=0
                              ).iloc[:, :4]
    CF_PV_PH_1.columns = list_of_repr_days_per_scenario
    CF_PV_PH_2 = pd.read_csv(os.path.join(curPath, 'data/CF_solar_PVSAT_Panhandle.csv'), index_col=0, header=0
                              ).iloc[:, 4:]
    CF_PV_PH_2.columns = list_of_repr_days_per_scenario
    
    # -> wind (old turbines)
    CF_wind_NE_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_Northeast.csv'), index_col=0, header=0
                              ).iloc[:, :4]
    CF_wind_NE_1.columns = list_of_repr_days_per_scenario
    CF_wind_NE_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_Northeast.csv'), index_col=0, header=0
                              ).iloc[:, 4:]
    CF_wind_NE_2.columns = list_of_repr_days_per_scenario
    CF_wind_W_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_West.csv'), index_col=0, header=0).iloc[:, :4]
    CF_wind_W_1.columns = list_of_repr_days_per_scenario
    CF_wind_W_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_West.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_wind_W_2.columns = list_of_repr_days_per_scenario
    CF_wind_C_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_Coastal.csv'), index_col=0, header=0).iloc[:, :4]
    CF_wind_C_1.columns = list_of_repr_days_per_scenario
    CF_wind_C_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_Coastal.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_wind_C_2.columns = list_of_repr_days_per_scenario
    CF_wind_S_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_South.csv'), index_col=0, header=0).iloc[:, :4]
    CF_wind_S_1.columns = list_of_repr_days_per_scenario
    CF_wind_S_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_South.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_wind_S_2.columns = list_of_repr_days_per_scenario
    CF_wind_PH_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_Panhandle.csv'), index_col=0, header=0
                              ).iloc[:, :4]
    CF_wind_PH_1.columns = list_of_repr_days_per_scenario
    CF_wind_PH_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_Panhandle.csv'), index_col=0, header=0
                              ).iloc[:, 4:]
    CF_wind_PH_2.columns = list_of_repr_days_per_scenario
    
    # -> wind new (new turbines)
    CF_wind_new_NE_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_Northeast.csv'), index_col=0, header=0
                              ).iloc[:, :4]
    CF_wind_new_NE_1.columns = list_of_repr_days_per_scenario
    CF_wind_new_NE_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_Northeast.csv'), index_col=0, header=0
                              ).iloc[:, 4:]
    CF_wind_new_NE_2.columns = list_of_repr_days_per_scenario
    CF_wind_new_W_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_West.csv'), index_col=0, header=0).iloc[:, :4]
    CF_wind_new_W_1.columns = list_of_repr_days_per_scenario
    CF_wind_new_W_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_West.csv'), index_col=0, header=0).iloc[:, 4:]
    CF_wind_new_W_2.columns = list_of_repr_days_per_scenario
    CF_wind_new_C_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_Coastal.csv'), index_col=0, header=0
                                  ).iloc[:, :4]
    CF_wind_new_C_1.columns = list_of_repr_days_per_scenario
    CF_wind_new_C_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_Coastal.csv'), index_col=0, header=0
                                  ).iloc[:, 4:]
    CF_wind_new_C_2.columns = list_of_repr_days_per_scenario
    CF_wind_new_S_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_South.csv'), index_col=0, header=0
                                  ).iloc[:, :4]
    CF_wind_new_S_1.columns = list_of_repr_days_per_scenario
    CF_wind_new_S_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_South.csv'), index_col=0, header=0
                                  ).iloc[:, 4:]
    CF_wind_new_S_2.columns = list_of_repr_days_per_scenario
    CF_wind_new_PH_1 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_Panhandle.csv'), index_col=0, header=0
                              ).iloc[:, :4]
    CF_wind_new_PH_1.columns = list_of_repr_days_per_scenario
    CF_wind_new_PH_2 = pd.read_csv(os.path.join(curPath, 'data/CF_wind_new_Panhandle.csv'), index_col=0, header=0
                              ).iloc[:, 4:]
    CF_wind_new_PH_2.columns = list_of_repr_days_per_scenario

    cf_1 = {}
    cf_2 = {}
    for t in L_max:
        d_idx = 0
        for d in list_of_repr_days_per_scenario:
            s_idx = 0
            for s in list(L_NE_1.index):
                for i in ['csp-new']:
                    cf_1[i, 'Northeast', t, d, s] = CF_CSP_NE_1.iat[s_idx, d_idx]
                    cf_1[i, 'West', t, d, s] = CF_CSP_W_1.iat[s_idx, d_idx]
                    cf_1[i, 'Coastal', t, d, s] = CF_CSP_C_1.iat[s_idx, d_idx]
                    cf_1[i, 'South', t, d, s] = CF_CSP_S_1.iat[s_idx, d_idx]
                    cf_1[i, 'Panhandle', t, d, s] = CF_CSP_PH_1.iat[s_idx, d_idx]
                    cf_2[i, 'Northeast', t, d, s] = CF_CSP_NE_2.iat[s_idx, d_idx]
                    cf_2[i, 'West', t, d, s] = CF_CSP_W_2.iat[s_idx, d_idx]
                    cf_2[i, 'Coastal', t, d, s] = CF_CSP_C_2.iat[s_idx, d_idx]
                    cf_2[i, 'South', t, d, s] = CF_CSP_S_2.iat[s_idx, d_idx]
                    cf_2[i, 'Panhandle', t, d, s] = CF_CSP_PH_2.iat[s_idx, d_idx]
                for i in ['pv-old', 'pv-new']:
                    cf_1[i, 'Northeast', t, d, s] = CF_PV_NE_1.iat[s_idx, d_idx]
                    cf_1[i, 'West', t, d, s] = CF_PV_W_1.iat[s_idx, d_idx]
                    cf_1[i, 'Coastal', t, d, s] = CF_PV_C_1.iat[s_idx, d_idx]
                    cf_1[i, 'South', t, d, s] = CF_PV_S_1.iat[s_idx, d_idx]
                    cf_1[i, 'Panhandle', t, d, s] = CF_PV_PH_1.iat[s_idx, d_idx]
                    cf_2[i, 'Northeast', t, d, s] = CF_PV_NE_2.iat[s_idx, d_idx]
                    cf_2[i, 'West', t, d, s] = CF_PV_W_2.iat[s_idx, d_idx]
                    cf_2[i, 'Coastal', t, d, s] = CF_PV_C_2.iat[s_idx, d_idx]
                    cf_2[i, 'South', t, d, s] = CF_PV_S_2.iat[s_idx, d_idx]
                    cf_2[i, 'Panhandle', t, d, s] = CF_PV_PH_2.iat[s_idx, d_idx]
                for i in ['wind-old']:
                    cf_1[i, 'Northeast', t, d, s] = CF_wind_NE_1.iat[s_idx, d_idx]
                    cf_1[i, 'West', t, d, s] = CF_wind_W_1.iat[s_idx, d_idx]
                    cf_1[i, 'Coastal', t, d, s] = CF_wind_C_1.iat[s_idx, d_idx]
                    cf_1[i, 'South', t, d, s] = CF_wind_S_1.iat[s_idx, d_idx]
                    cf_1[i, 'Panhandle', t, d, s] = CF_wind_PH_1.iat[s_idx, d_idx]
                    cf_2[i, 'Northeast', t, d, s] = CF_wind_NE_2.iat[s_idx, d_idx]
                    cf_2[i, 'West', t, d, s] = CF_wind_W_2.iat[s_idx, d_idx]
                    cf_2[i, 'Coastal', t, d, s] = CF_wind_C_2.iat[s_idx, d_idx]
                    cf_2[i, 'South', t, d, s] = CF_wind_S_2.iat[s_idx, d_idx]
                    cf_2[i, 'Panhandle', t, d, s] = CF_wind_PH_2.iat[s_idx, d_idx]
                for i in ['wind-new']:
                    cf_1[i, 'Northeast', t, d, s] = CF_wind_new_NE_1.iat[s_idx, d_idx]
                    cf_1[i, 'West', t, d, s] = CF_wind_new_W_1.iat[s_idx, d_idx]
                    cf_1[i, 'Coastal', t, d, s] = CF_wind_new_C_1.iat[s_idx, d_idx]
                    cf_1[i, 'South', t, d, s] = CF_wind_new_S_1.iat[s_idx, d_idx]
                    cf_1[i, 'Panhandle', t, d, s] = CF_wind_new_PH_1.iat[s_idx, d_idx]
                    cf_2[i, 'Northeast', t, d, s] = CF_wind_new_NE_2.iat[s_idx, d_idx]
                    cf_2[i, 'West', t, d, s] = CF_wind_new_W_2.iat[s_idx, d_idx]
                    cf_2[i, 'Coastal', t, d, s] = CF_wind_new_C_2.iat[s_idx, d_idx]
                    cf_2[i, 'South', t, d, s] = CF_wind_new_S_2.iat[s_idx, d_idx]
                    cf_2[i, 'Panhandle', t, d, s] = CF_wind_new_PH_2.iat[s_idx, d_idx]
                s_idx += 1
            d_idx += 1

    cf_by_scenario = [cf_1, cf_2]
    # print(cf_by_scenario)
    globals()["cf_by_scenario"] = cf_by_scenario

    ####################################################################################################################

    # Strategic uncertainty data

    # Different scenarios for PEAK LOAD:
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

    # Different scenarios for CARBON TAX:
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


