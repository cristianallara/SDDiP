import sqlite3 as sql
import os
import pandas as pd


def read_data(database_file, curPath, stages, n_stage, t_per_stage):
    print(os.path.exists(database_file))
    print(database_file)
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
              'if_', 'ED', 'Rmin', 'hr', 'EF_CO2', 'FOC', 'VOC', 'CCm', 'DIC', 'LEC', 'PEN', 'tx_CO2',
              'RES_min', 'P_fuel']  # , 'L', 'cf',

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
                        ('Li_ion', 4): 1099390.353, ('Li_ion', 5): 1017197.661,
                        ('Li_ion', 6): 948004.5483,
                        ('Li_ion', 7): 887689.7192, ('Li_ion', 8): 834006.1087, ('Li_ion', 9): 785632.3346,
                        ('Li_ion', 10): 741754.6519,
                        ('Li_ion', 11): 701857.294, ('Li_ion', 12): 665604.5808,
                        ('Li_ion', 13): 632766.397, ('Li_ion', 14): 603165.7101, ('Li_ion', 15): 576639.6204,
                        ('Lead_acid', 1): 4346125.294, ('Lead_acid', 2): 3857990.578, ('Lead_acid', 3): 3458901.946,
                        ('Lead_acid', 4): 3117666.824, ('Lead_acid', 5): 2818863.27,
                        ('Lead_acid', 6): 2553828.021,
                        ('Lead_acid', 7): 2317228.867, ('Lead_acid', 8): 2105569.661, ('Lead_acid', 9): 1916483.132,
                        ('Lead_acid', 10): 1748369.467,
                        ('Lead_acid', 11): 1600168.567, ('Lead_acid', 12): 1471137.002,
                        ('Lead_acid', 13): 1360557.098,('Lead_acid', 14): 1267402.114, ('Lead_acid', 15): 1190102.412,
                        ('Flow', 1): 4706872.908, ('Flow', 2): 3218220.336, ('Flow', 3): 2810526.973,
                        ('Flow', 4): 2555010.035, ('Flow', 5): 2362062.488,
                        ('Flow', 6): 2203531.648,
                        ('Flow', 7): 2067165.77, ('Flow', 8): 1946678.078, ('Flow', 9): 1838520.24,
                        ('Flow', 10): 1740573.662
                        , ('Flow', 11): 1651531.463, ('Flow', 12): 1570567.635,
                        ('Flow', 13): 1497136.957, ('Flow', 14): 1430839.31, ('Flow', 15): 1371321.436
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

    # Different scenarios for CARBON TAX:
    # tx_CO2_scenario = {(2, 'L'): 0, (2, 'M'): 0.050, (2, 'H'): 0.100,
    #                    (3, 'L'): 0, (3, 'M'): 0.065, (3, 'H'): 0.131,
    #                    (4, 'L'): 0, (4, 'M'): 0.081, (4, 'H'): 0.162,
    #                    (5, 'L'): 0, (5, 'M'): 0.096, (5, 'H'): 0.192,
    #                    (6, 'L'): 0, (6, 'M'): 0.112, (6, 'H'): 0.223,
    #                    (7, 'L'): 0, (7, 'M'): 0.127, (7, 'H'): 0.254,
    #                    (8, 'L'): 0, (8, 'M'): 0.142, (8, 'H'): 0.285,
    #                    (9, 'L'): 0, (9, 'M'): 0.158, (9, 'H'): 0.315,
    #                    (10, 'L'): 0, (10, 'M'): 0.173, (10, 'H'): 0.346,
    #                    (11, 'L'): 0, (11, 'M'): 0.188, (11, 'H'): 0.377,
    #                    (12, 'L'): 0, (12, 'M'): 0.204, (12, 'H'): 0.408,
    #                    (13, 'L'): 0, (13, 'M'): 0.219, (13, 'H'): 0.438,
    #                    (14, 'L'): 0, (14, 'M'): 0.235, (14, 'H'): 0.469,
    #                    (15, 'L'): 0, (15, 'M'): 0.250, (15, 'H'): 0.500
    #                    }
    #
    # tx_CO2 = {}
    # for stage in stages:
    #     for t in t_per_stage[stage]:
    #         if stage == 1:
    #             tx_CO2[t, stage, 'O'] = 0
    #         else:
    #             for n in ['L', 'M', 'H']:
    #                 tx_CO2[t, stage, n] = tx_CO2_scenario[t, n]
    # globals()["tx_CO2"] = tx_CO2

    # Different scenarios for NG PRICE:
    ng_price_scenarios = {(1, 'L'): 3.117563, (1, 'M'): 3.4014395, (1, 'H'): 4.249755,
                          (2, 'L'): 2.976701, (2, 'M'): 3.357056, (2, 'H'): 4.188047,
                          (3, 'L'): 2.974117, (3, 'M'): 3.4164015, (3, 'H'): 4.228118,
                          (4, 'L'): 3.082466, (4, 'M'): 3.578708, (4, 'H'): 4.403251,
                          (5, 'L'): 3.236482, (5, 'M'): 3.8122265, (5, 'H'): 4.745406,
                          (6, 'L'): 3.394663, (6, 'M'): 3.9940535, (6, 'H'): 5.088468,
                          (7, 'L'): 3.479183, (7, 'M'): 4.0682835, (7, 'H'): 5.442574,
                          (8, 'L'): 3.504514, (8, 'M'): 4.117297, (8, 'H'): 5.565526,
                          (9, 'L'): 3.498631, (9, 'M'): 4.188261, (9, 'H'): 5.82389,
                          (10, 'L'): 3.490988, (10, 'M'): 4.219348, (10, 'H'): 5.905959,
                          (11, 'L'): 3.483505, (11, 'M'): 4.250815, (11, 'H'): 5.955,
                          (12, 'L'): 3.496959, (12, 'M'): 4.2411075, (12, 'H'): 6.013945,
                          (13, 'L'): 3.534126, (13, 'M'): 4.3724575, (13, 'H'): 6.17547,
                          (14, 'L'): 3.57645, (14, 'M'): 4.4414835, (14, 'H'): 6.240099,
                          (15, 'L'): 3.585003, (15, 'M'): 4.47585855, (15, 'H'): 6.41513
                          }
    for t in range(1, 16):
        ng_price_scenarios[t, 'H'] = 1.5 * ng_price_scenarios[t, 'H']
    print(ng_price_scenarios)
    # min, median and max values from the scenarios of EIA outlook for NG price for electricity
    # https://www.eia.gov/outlooks/aeo/data/browser/#/?id=3-AEO2019&cases=ref2019&sourcekey=0

    th_generators = ['coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new', 'ng-ct-old',
                     'ng-cc-old', 'ng-st-old', 'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new', 'nuc-st-old', 'nuc-st-new']
    ng_generators = ['ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new', 'ng-cc-ccs-new', 'ng-ct-new']

    P_fuel_scenarios = {}
    for stage in stages:
        for t in t_per_stage[stage]:
            for i in th_generators:
                if stage == 1:
                    P_fuel_scenarios[i, t, stage, 'O'] = P_fuel[i, t]
                else:
                    for n in ['L', 'M', 'H']:
                        if i in ng_generators:
                            P_fuel_scenarios[i, t, stage, n] = ng_price_scenarios[t, n]
                        else:
                            P_fuel_scenarios[i, t, stage, n] = P_fuel[i, t]
    globals()["P_fuel_scenarios"] = P_fuel_scenarios
    # print(P_fuel_scenarios)

    print('finished loading data')
