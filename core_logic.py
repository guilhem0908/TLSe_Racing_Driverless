import numpy as np
import casadi as ca
import math
import math_utils

class BattModel:
    def __init__(self, pwr_pars):
        self.pars = pwr_pars
        self.temp_batt_n = None
        self.temp_batt_s = None
        self.temp_batt = None
        self.dtemp = None
        self.dsoc = None
        self.temp_min = None
        self.temp_max = None
        self.temp_guess = None
        self.soc_min = None
        self.soc_max = None
        self.soc_guess = None
        self.soc_batt_n = None
        self.soc_batt_s = None
        self.soc_batt = None
        self.v_dc = None
        self.i_batt = None
        self.Ri = None
        self.f_nlp = None
        self.f_sol = None
        self.p_loss_total = None
        self.p_out_batt = None
        self.p_internal_batt = None
        self.r_batt_inverse = None
        self.p_losses_opt = []
        self.initialize()

    def initialize(self):
        self.temp_batt_n = ca.SX.sym('temp_batt_n')
        self.temp_batt_s = self.pars["temp_batt_max"] - 10
        self.temp_batt = self.temp_batt_s * self.temp_batt_n
        self.temp_min = self.pars["T_env"] / self.temp_batt_s
        self.temp_max = self.pars["temp_batt_max"] / self.temp_batt_s
        self.temp_guess = self.pars["T_env"] / self.temp_batt_s
        self.soc_batt_n = ca.SX.sym('soc_batt_n')
        self.soc_batt_s = 1
        self.soc_batt = self.soc_batt_s * self.soc_batt_n
        self.soc_min = 0 / self.soc_batt_s
        self.soc_max = 1 / self.soc_batt_s
        self.soc_guess = 0.5
        self.get_thermal_resistance()
        self.ocv_voltage()

    def get_increment(self, sf, temp_cool_b):
        self.dtemp = sf * ((self.p_loss_total * 1000 - self.r_batt_inverse * (self.temp_batt - temp_cool_b)) / (self.pars["C_therm_cell"] * self.pars["N_cells_serial"] * self.pars["N_cells_parallel"]))

    def get_soc(self, sf):
        self.dsoc = - sf * ((self.p_out_batt + self.p_loss_total) / 3600 / self.pars["C_batt"])

    def battery_loss(self, p_des, p_loss_inv, p_loss_mot, p_in_inv=None):
        if self.pars.get("simple_loss", False):
            p_in_inv *= self.pars["N_machines"]
            p_internal_batt = ((self.pars["V_OC_simple"] ** 2) / (2 * self.pars["R_i_simple"])) - self.pars["V_OC_simple"] * np.sqrt((self.pars["V_OC_simple"] ** 2 - 4 * p_in_inv * 1000 * self.pars["R_i_simple"])) / (2 * self.pars["R_i_simple"])
            self.p_internal_batt = 0.001 * p_internal_batt
            self.p_loss_total = self.p_internal_batt - p_in_inv
            self.p_out_batt = p_in_inv
        else:
            p_out_batt = (p_des + p_loss_inv + p_loss_mot) * 1000
            self.i_batt = p_out_batt / self.v_dc
            p_internal_batt = ((self.v_dc ** 2) / (2 * self.Ri)) - self.v_dc * np.sqrt((self.v_dc ** 2 - 4 * p_out_batt * self.Ri)) / (2 * self.Ri)
            p_loss = p_internal_batt - p_out_batt
            self.p_loss_total = 0.001 * p_loss
            self.p_out_batt = 0.001 * p_out_batt

    def ocv_voltage(self):
        self.v_dc = self.pars["N_cells_serial"] * (1.245 * self.soc_batt ** 3 - 1.679 * self.soc_batt ** 2 + 1.064 * self.soc_batt + 3.566)

    def internal_resistance(self):
        self.Ri = self.pars["N_cells_serial"] / self.pars["N_cells_parallel"] * (self.pars["R_i_offset"] - self.pars["R_i_slope"] * self.temp_batt)

    def get_thermal_resistance(self):
        self.r_batt_inverse = 1 / 0.002

    def ini_nlp_state(self, x, u):
        self.f_nlp = ca.Function('f_nlp', [x, u], [self.p_loss_total, self.p_out_batt], ['x', 'u'], ['p_loss_total', 'p_out_batt'])

    def extract_sol(self, w, sol_states):
        self.f_sol = ca.Function('f_sol', [w], [self.p_losses_opt], ['w'], ['p_losses_opt'])
        p_losses_opt = self.f_sol(sol_states)
        self.p_loss_total = p_losses_opt[0::2]
        self.p_out_batt = p_losses_opt[1::2]

class EMachineModel:
    def __init__(self, pwr_pars):
        self.pars = pwr_pars
        self.temp_mot_n = None
        self.temp_mot_s = None
        self.temp_mot = None
        self.dtemp = None
        self.temp_min = None
        self.temp_max = None
        self.temp_guess = None
        self.f_nlp = None
        self.f_sol = None
        self.i_eff = None
        self.omega_machine = None
        self.p_input = None
        self.p_loss_copper = None
        self.p_loss_stator_iron = None
        self.p_loss_rotor = None
        self.p_loss_total = None
        self.p_loss_total_all_machines = None
        self.r_machine = None
        self.p_losses_opt = []
        self.initialize()

    def initialize(self):
        self.temp_mot_n = ca.SX.sym('temp_mot_n')
        self.temp_mot_s = self.pars["temp_mot_max"] - 50
        self.temp_mot = self.temp_mot_s * self.temp_mot_n
        self.temp_min = self.pars["T_env"] / self.temp_mot_s
        self.temp_max = self.pars["temp_mot_max"] / self.temp_mot_s
        self.temp_guess = self.pars["T_env"] / self.temp_mot_s
        self.get_thermal_resistance()

    def get_states(self, f_drive, v):
        self.i_eff = (f_drive * self.pars["r_wheel"] / self.pars["MotorConstant"] / self.pars["transmission"]) / self.pars["N_machines"]
        self.omega_machine = v / (2 * np.pi * self.pars["r_wheel"]) * self.pars["transmission"] * 60

    def get_increment(self, sf, temp_cool_12, temp_cool_13):
        self.dtemp = sf * ((self.p_loss_total * 1000 - (self.temp_mot - (temp_cool_12 + temp_cool_13) / 2) / self.r_machine) / (self.pars["C_therm_machine"]))

    def get_loss(self, p_wheel):
        if self.pars.get("simple_loss", False):
            p_machine_in = self.pars["machine_simple_a"] * (p_wheel / self.pars["N_machines"]) ** 2 + self.pars["machine_simple_b"] * (p_wheel / self.pars["N_machines"]) + self.pars["machine_simple_c"]
            self.p_input = p_machine_in
            self.p_loss_total = p_machine_in - p_wheel / self.pars["N_machines"]
        else:
            temp_mot = self.temp_mot
            omega_machine = self.omega_machine
            i_eff = self.i_eff
            p_loss_copper = ((((temp_mot - 20) * self.pars["C_TempCopper"]) + 1) * self.pars["R_Phase"]) * (i_eff ** 2) * (3 / 2)
            p_loss_stator_iron = 2.885e-13 * omega_machine ** 4 - 1.114e-08 * omega_machine ** 3 + 0.0001123 * omega_machine ** 2 + 0.1657 * omega_machine + 272
            p_loss_rotor = 8.143e-14 * omega_machine ** 4 - 2.338e-09 * omega_machine ** 3 + 1.673e-05 * omega_machine ** 2 + 0.112 * omega_machine - 113.6
            p_loss_total = (p_loss_copper + p_loss_stator_iron + p_loss_rotor) * 0.001
            self.p_loss_copper = 0.001 * p_loss_copper
            self.p_loss_stator_iron = 0.001 * p_loss_stator_iron
            self.p_loss_rotor = 0.001 * p_loss_rotor
            self.p_loss_total = p_loss_total

    def get_machines_cum_losses(self):
        self.p_loss_total_all_machines = self.p_loss_total * self.pars["N_machines"]

    def get_thermal_resistance(self):
        A_cool_machine = 2 * np.pi * self.pars["r_stator_ext"] * self.pars["l_machine"] * self.pars["A_cool_inflate_machine"]
        r_cond_stator = np.log(self.pars["r_stator_ext"] / self.pars["r_stator_int"]) / (2 * np.pi * self.pars["k_iro"] * self.pars["l_machine"])
        r_cond_rotor = np.log(self.pars["r_rotor_ext"] / self.pars["r_rotor_int"]) / (2 * np.pi * self.pars["k_iro"] * self.pars["l_machine"])
        r_cond_shaft = 1 / (4 * np.pi * self.pars["k_iro"] * self.pars["l_machine"])
        r_conv_fluid = 1 / (self.pars["h_fluid_mi"] * A_cool_machine)
        r_conv_airgap = 1 / (2 * np.pi * self.pars["h_air_gap"] * self.pars["r_stator_int"] * self.pars["l_machine"])
        r1 = r_cond_stator + r_conv_fluid
        r2 = r_cond_shaft + r_cond_rotor + r_conv_airgap
        self.r_machine = (r1 * r2) / (r1 + r2)

    def ini_nlp_state(self, x, u):
        if self.pars.get("simple_loss", False):
            self.f_nlp = ca.Function('f_nlp', [x, u], [self.p_loss_total, self.p_input], ['x', 'u'], ['p_loss_total', 'p_input'])
        else:
            self.f_nlp = ca.Function('f_nlp', [x, u], [self.p_loss_total, self.p_loss_copper, self.p_loss_stator_iron, self.p_loss_rotor, self.i_eff], ['x', 'u'], ['p_loss_total', 'p_loss_copper', 'p_loss_stator_iron', 'p_loss_rotor', 'i_eff'])

    def extract_sol(self, w, sol_states):
        self.f_sol = ca.Function('f_sol', [w], [self.p_losses_opt], ['w'], ['p_losses_opt'])
        p_losses_opt = self.f_sol(sol_states)
        if self.pars.get("simple_loss", False):
            self.p_loss_total = p_losses_opt[0::2]
            self.p_input = p_losses_opt[1::2]
        else:
            self.p_loss_total = p_losses_opt[0::5]
            self.p_loss_copper = p_losses_opt[1::5]
            self.p_loss_stator_iron = p_losses_opt[2::5]
            self.p_loss_rotor = p_losses_opt[3::5]
            self.i_eff = p_losses_opt[4::5]

class InverterModel:
    def __init__(self, pwr_pars):
        self.pars = pwr_pars
        self.temp_inv_n = None
        self.temp_inv_s = None
        self.temp_inv = None
        self.dtemp = None
        self.temp_min = None
        self.temp_max = None
        self.temp_guess = None
        self.f_nlp = None
        self.f_sol = None
        self.p_in_inv = None
        self.p_loss_switch = None
        self.p_loss_cond = None
        self.p_loss_total = None
        self.p_loss_total_all_inverters = None
        self.r_inv = None
        self.p_losses_opt = []
        self.initialize()

    def initialize(self):
        self.temp_inv_n = ca.SX.sym('temp_inv_n')
        self.temp_inv_s = self.pars["temp_inv_max"] - 30
        self.temp_inv = self.temp_inv_s * self.temp_inv_n
        self.temp_min = self.pars["T_env"] / self.temp_inv_s
        self.temp_max = self.pars["temp_inv_max"] / self.temp_inv_s
        self.temp_guess = self.pars["T_env"] / self.temp_inv_s
        self.get_thermal_resistance()

    def get_increment(self, sf, temp_cool_mi, temp_cool_12):
        self.dtemp = sf * ((self.p_loss_total * 1000 - (self.temp_inv - (temp_cool_mi + temp_cool_12) / 2) / self.r_inv) / (self.pars["C_therm_inv"]))

    def get_loss(self, i_eff, v_dc, p_out_inv=None):
        if self.pars.get("simple_loss", False):
            self.p_in_inv = self.pars["inverter_simple_a"] * p_out_inv ** 2 + self.pars["inverter_simple_b"] * p_out_inv + self.pars["inverter_simple_c"]
            self.p_loss_total = (self.p_in_inv - p_out_inv)
        else:
            p_loss_switch = (v_dc / self.pars["V_ref"]) * ((3 * self.pars["f_sw"]) * (i_eff / self.pars["I_ref"]) * (self.pars["E_on"] + self.pars["E_off"] + self.pars["E_rr"]))
            p_loss_cond = 3 * i_eff * (self.pars["V_ce_offset"] + (self.pars["V_ce_slope"] * i_eff))
            self.p_loss_switch = 0.001 * p_loss_switch
            self.p_loss_cond = 0.001 * p_loss_cond
            self.p_loss_total = (p_loss_switch + p_loss_cond) * 0.001

    def get_inverters_cum_losses(self):
        self.p_loss_total_all_inverters = self.p_loss_total * self.pars["N_machines"]

    def get_thermal_resistance(self):
        self.r_inv = 1 / (self.pars["h_fluid_mi"] * self.pars["A_cool_inv"])

    def ini_nlp_state(self, x, u):
        if self.pars.get("simple_loss", False):
            self.f_nlp = ca.Function('f_nlp', [x, u], [self.p_loss_total, self.p_in_inv], ['x', 'u'], ['p_loss_total', 'p_inv_in'])
        else:
            self.f_nlp = ca.Function('f_nlp', [x, u], [self.p_loss_total, self.p_loss_switch, self.p_loss_cond], ['x', 'u'], ['p_loss_total', 'p_loss_switch', 'p_loss_cond'])

    def extract_sol(self, w, sol_states):
        self.f_sol = ca.Function('f_sol', [w], [self.p_losses_opt], ['w'], ['p_losses_opt'])
        p_losses_opt = self.f_sol(sol_states)
        if self.pars.get("simple_loss", False):
            self.p_loss_total = p_losses_opt[0::2]
            self.p_in_inv = p_losses_opt[1::2]
        else:
            self.p_loss_total = p_losses_opt[0::3]
            self.p_loss_switch = p_losses_opt[1::3]
            self.p_loss_cond = p_losses_opt[2::3]

class RadiatorModel:
    def __init__(self, pwr_pars):
        self.pars = pwr_pars
        self.temp_cool_mi_n = None
        self.temp_cool_mi_s = None
        self.temp_cool_mi = None
        self.temp_cool_b_n = None
        self.temp_cool_b_s = None
        self.temp_cool_b = None
        self.temp_cool_mi_min = None
        self.temp_cool_mi_max = None
        self.temp_cool_mi_guess = None
        self.temp_cool_b_min = None
        self.temp_cool_b_max = None
        self.temp_cool_b_guess = None
        self.temp_cool_12 = None
        self.temp_cool_13 = None
        self.dtemp_cool_mi = None
        self.dtemp_cool_b = None
        self.r_rad = None
        self.f_nlp = None
        self.f_sol = None
        self.temps_opt = []
        self.initialize()

    def initialize(self):
        self.temp_cool_mi_n = ca.SX.sym('temp_cool_mi_n')
        self.temp_cool_mi_s = self.pars["temp_inv_max"] - 30
        self.temp_cool_mi = self.temp_cool_mi_s * self.temp_cool_mi_n
        self.temp_cool_b_n = ca.SX.sym('temp_cool_b_n')
        self.temp_cool_b_s = self.pars["temp_batt_max"] - 10
        self.temp_cool_b = self.temp_cool_b_s * self.temp_cool_b_n
        self.temp_cool_mi_min = self.pars["T_env"] / self.temp_cool_mi_s
        self.temp_cool_mi_max = (self.pars["temp_inv_max"] - 10) / self.temp_cool_mi_s
        self.temp_cool_mi_guess = (self.pars["T_env"]) / self.temp_cool_mi_s
        self.temp_cool_b_min = self.pars["T_env"] / self.temp_cool_b_s
        self.temp_cool_b_max = self.pars["temp_batt_max"] / self.temp_cool_b_s
        self.temp_cool_b_guess = self.pars["T_env"] / self.temp_cool_b_s
        self.get_thermal_resistance()

    def get_thermal_resistance(self):
        self.r_rad = 1 / (self.pars["h_air"] * self.pars["A_cool_rad"])

    def get_intermediate_temps(self, temp_inv, r_inv):
        self.temp_cool_12 = (self.temp_cool_mi * (self.pars["c_heat_fluid"] * self.pars["flow_rate_inv"] * r_inv - 1) + 2 * temp_inv) / (1 + self.pars["c_heat_fluid"] * self.pars["flow_rate_inv"] * r_inv)
        self.temp_cool_13 = (self.temp_cool_mi * (2 * self.pars["c_heat_fluid"] * self.pars["flow_rate_rad"] * self.r_rad + 1) - 2 * self.pars["T_env"]) / (-1 + 2 * self.pars["c_heat_fluid"] * self.pars["flow_rate_rad"] * self.r_rad)

    def get_increment_mi(self, sf, temp_mot, temp_inv, r_inv, r_machine):
        self.dtemp_cool_mi = sf * ((self.pars["N_machines"] * ((temp_mot - (self.temp_cool_12 + self.temp_cool_13) / 2) / r_machine + (temp_inv - (self.temp_cool_mi + self.temp_cool_12) / 2) / r_inv) - ((self.temp_cool_mi + self.temp_cool_13) / 2 - self.pars["T_env"]) / self.r_rad) / (self.pars["m_therm_fluid_mi"] * self.pars["c_heat_fluid"]))

    def get_increment_b(self, sf, temp_batt, temp_cool_b, R_eq_B_inv):
        self.dtemp_cool_b = sf * ((R_eq_B_inv * (temp_batt - temp_cool_b) - (temp_cool_b - self.pars["T_env"]) / self.r_rad) / (self.pars["m_therm_fluid_b"] * self.pars["c_heat_fluid"]))

    def ini_nlp_state(self, x, u):
        self.f_nlp = ca.Function('f_nlp', [x, u], [self.temp_cool_mi, self.temp_cool_b], ['x', 'u'], ['temp_cool_mi', 'temp_cool_b'])

    def extract_sol(self, w, sol_states):
        self.f_sol = ca.Function('f_sol', [w], [self.temps_opt], ['w'], ['temps_opt'])
        temps_opt = self.f_sol(sol_states)
        self.temp_cool_mi = temps_opt[0::2]
        self.temp_cool_b = temps_opt[1::2]

def import_track(file_path, imp_opts, width_veh):
    csv_data_temp = np.loadtxt(file_path, comments='#', delimiter=',')
    if np.shape(csv_data_temp)[1] == 3:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2] / 2
        w_tr_l = w_tr_r
    elif np.shape(csv_data_temp)[1] == 4:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2]
        w_tr_l = csv_data_temp[:, 3]
    elif np.shape(csv_data_temp)[1] == 5:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 3]
        w_tr_l = csv_data_temp[:, 4]
    refline_ = np.tile(refline_, (imp_opts["num_laps"], 1))
    w_tr_r = np.tile(w_tr_r, imp_opts["num_laps"])
    w_tr_l = np.tile(w_tr_l, imp_opts["num_laps"])
    reftrack_imp = np.column_stack((refline_, w_tr_r, w_tr_l))
    if imp_opts["flip_imp_track"]:
        reftrack_imp = np.flipud(reftrack_imp)
    if imp_opts["set_new_start"]:
        ind_start = np.argmin(np.power(reftrack_imp[:, 0] - imp_opts["new_start"][0], 2) + np.power(reftrack_imp[:, 1] - imp_opts["new_start"][1], 2))
        reftrack_imp = np.roll(reftrack_imp, reftrack_imp.shape[0] - ind_start, axis=0)
    return reftrack_imp

def prep_track(reftrack_imp, reg_smooth_opts, stepsize_opts, debug=True, min_width=None):
    reftrack_interp = math_utils.spline_approximation(track=reftrack_imp, k_reg=reg_smooth_opts["k_reg"], s_reg=reg_smooth_opts["s_reg"], stepsize_prep=stepsize_opts["stepsize_prep"], stepsize_reg=stepsize_opts["stepsize_reg"], debug=debug)
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))
    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = math_utils.calc_splines(path=refpath_interp_cl)
    math_utils.check_normals_crossing(track=reftrack_interp, normvec_normalized=normvec_normalized_interp, horizon=10)
    if min_width is not None:
        for i in range(reftrack_interp.shape[0]):
            cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]
            if cur_width < min_width:
                reftrack_interp[i, 2] += (min_width - cur_width) / 2
                reftrack_interp[i, 3] += (min_width - cur_width) / 2
    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp

def calc_min_bound_dists(trajectory, bound1, bound2, length_veh, width_veh):
    bounds = np.vstack((bound1, bound2))
    fl = np.array([-width_veh / 2, length_veh / 2])
    fr = np.array([width_veh / 2, length_veh / 2])
    rl = np.array([-width_veh / 2, -length_veh / 2])
    rr = np.array([width_veh / 2, -length_veh / 2])
    min_dists = np.zeros(trajectory.shape[0])
    mat_rot = np.zeros((2, 2))
    for i in range(trajectory.shape[0]):
        mat_rot[0, 0] = math.cos(trajectory[i, 3])
        mat_rot[0, 1] = -math.sin(trajectory[i, 3])
        mat_rot[1, 0] = math.sin(trajectory[i, 3])
        mat_rot[1, 1] = math.cos(trajectory[i, 3])
        fl_ = trajectory[i, 1:3] + np.matmul(mat_rot, fl)
        fr_ = trajectory[i, 1:3] + np.matmul(mat_rot, fr)
        rl_ = trajectory[i, 1:3] + np.matmul(mat_rot, rl)
        rr_ = trajectory[i, 1:3] + np.matmul(mat_rot, rr)
        fl__mindist = np.sqrt(np.power(bounds[:, 0] - fl_[0], 2) + np.power(bounds[:, 1] - fl_[1], 2))
        fr__mindist = np.sqrt(np.power(bounds[:, 0] - fr_[0], 2) + np.power(bounds[:, 1] - fr_[1], 2))
        rl__mindist = np.sqrt(np.power(bounds[:, 0] - rl_[0], 2) + np.power(bounds[:, 1] - rl_[1], 2))
        rr__mindist = np.sqrt(np.power(bounds[:, 0] - rr_[0], 2) + np.power(bounds[:, 1] - rr_[1], 2))
        min_dists[i] = np.amin((fl__mindist, fr__mindist, rl__mindist, rr__mindist))
    return min_dists

def check_traj(reftrack, reftrack_normvec_normalized, trajectory, ggv, ax_max_machines, v_max, length_veh, width_veh, debug, dragcoeff, mass_veh, curvlim):
    bound_r = reftrack[:, :2] + reftrack_normvec_normalized * np.expand_dims(reftrack[:, 2], 1)
    bound_l = reftrack[:, :2] - reftrack_normvec_normalized * np.expand_dims(reftrack[:, 3], 1)
    bound_r_tmp = np.column_stack((bound_r, np.zeros((bound_r.shape[0], 2))))
    bound_l_tmp = np.column_stack((bound_l, np.zeros((bound_l.shape[0], 2))))
    bound_r_interp = math_utils.interp_track(track=bound_r_tmp, stepsize=1.0)
    bound_l_interp = math_utils.interp_track(track=bound_l_tmp, stepsize=1.0)
    calc_min_bound_dists(trajectory=trajectory, bound1=bound_r_interp, bound2=bound_l_interp, length_veh=length_veh, width_veh=width_veh)
    if ggv is not None:
        radii = np.abs(np.divide(1.0, trajectory[:, 4], out=np.full(trajectory.shape[0], np.inf), where=trajectory[:, 4] != 0))
        ay_profile = np.divide(np.power(trajectory[:, 5], 2), radii)
        ax_drag = -np.power(trajectory[:, 5], 2) * dragcoeff / mass_veh
        ax_wo_drag = trajectory[:, 6] - ax_drag
    return bound_r, bound_l

def opt_mintime(reftrack, coeffs_x, coeffs_y, normvectors, pars, print_debug=False, plot_debug=False):
    no_points_orig = reftrack.shape[0]
    if pars["optim_opts"]["step_non_reg"] > 0:
        reftrack, discr_points = math_utils.nonreg_sampling(track=reftrack, eps_kappa=pars["optim_opts"]["eps_kappa"], step_non_reg=pars["optim_opts"]["step_non_reg"])
        refpath_cl = np.vstack((reftrack[:, :2], reftrack[0, :2]))
        coeffs_x, coeffs_y, a_interp, normvectors = math_utils.calc_splines(path=refpath_cl)
    else:
        discr_points = np.arange(reftrack.shape[0])
        a_interp = None
    spline_lengths_refline = math_utils.calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    kappa_refline = math_utils.calc_head_curv_num(path=reftrack[:, :2], el_lengths=spline_lengths_refline, is_closed=True, stepsize_curv_preview=pars["curv_calc_opts"]["d_preview_curv"], stepsize_curv_review=pars["curv_calc_opts"]["d_review_curv"], stepsize_psi_preview=pars["curv_calc_opts"]["d_preview_head"], stepsize_psi_review=pars["curv_calc_opts"]["d_review_head"])[1]
    kappa_refline_cl = np.append(kappa_refline, kappa_refline[0])
    discr_points_cl = np.append(discr_points, no_points_orig)
    w_tr_left_cl = np.append(reftrack[:, 3], reftrack[0, 3])
    w_tr_right_cl = np.append(reftrack[:, 2], reftrack[0, 2])
    h = pars["stepsize_opts"]["stepsize_reg"]
    steps = [i for i in range(discr_points_cl.size)]
    N = steps[-1]
    s_opt = np.asarray(discr_points_cl) * h
    kappa_interp = ca.interpolant('kappa_interp', 'linear', [steps], kappa_refline_cl)
    w_tr_left_interp = ca.interpolant('w_tr_left_interp', 'linear', [steps], w_tr_left_cl)
    w_tr_right_interp = ca.interpolant('w_tr_right_interp', 'linear', [steps], w_tr_right_cl)
    d = 3
    tau = np.append(0, ca.collocation_points(d, 'legendre'))
    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    B = np.zeros(d + 1)
    for j in range(d + 1):
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])
        D[j] = p(1.0)
        p_der = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = p_der(tau[r])
        pint = np.polyint(p)
        B[j] = pint(1.0)
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        nx = 11
        nx_pwr = 6
    else:
        nx = 5
        nx_pwr = 0
    v_n = ca.SX.sym('v_n')
    v_s = 50
    v = v_s * v_n
    beta_n = ca.SX.sym('beta_n')
    beta_s = 0.5
    beta = beta_s * beta_n
    omega_z_n = ca.SX.sym('omega_z_n')
    omega_z_s = 1
    omega_z = omega_z_s * omega_z_n
    n_n = ca.SX.sym('n_n')
    n_s = 5.0
    n = n_s * n_n
    xi_n = ca.SX.sym('xi_n')
    xi_s = 1.0
    xi = xi_s * xi_n
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        machine = EMachineModel(pwr_pars=pars["pwr_params_mintime"])
        batt = BattModel(pwr_pars=pars["pwr_params_mintime"])
        inverter = InverterModel(pwr_pars=pars["pwr_params_mintime"])
        radiators = RadiatorModel(pwr_pars=pars["pwr_params_mintime"])
        x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s, machine.temp_mot_s, batt.temp_batt_s, inverter.temp_inv_s, radiators.temp_cool_mi_s, radiators.temp_cool_b_s, batt.soc_batt_s])
        x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n, machine.temp_mot_n, batt.temp_batt_n, inverter.temp_inv_n, radiators.temp_cool_mi_n, radiators.temp_cool_b_n, batt.soc_batt_n)
    else:
        x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s])
        x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n)
    nu = 4
    delta_n = ca.SX.sym('delta_n')
    delta_s = 0.5
    delta = delta_s * delta_n
    f_drive_n = ca.SX.sym('f_drive_n')
    f_drive_s = 7500.0
    f_drive = f_drive_s * f_drive_n
    f_brake_n = ca.SX.sym('f_brake_n')
    f_brake_s = 20000.0
    f_brake = f_brake_s * f_brake_n
    gamma_y_n = ca.SX.sym('gamma_y_n')
    gamma_y_s = 5000.0
    gamma_y = gamma_y_s * gamma_y_n
    u_s = np.array([delta_s, f_drive_s, f_brake_s, gamma_y_s])
    u = ca.vertcat(delta_n, f_drive_n, f_brake_n, gamma_y_n)
    veh = pars["vehicle_params_mintime"]
    tire = pars["tire_params_mintime"]
    g = pars["veh_params"]["g"]
    mass = pars["veh_params"]["mass"]
    kappa = ca.SX.sym('kappa')
    f_xdrag = pars["veh_params"]["dragcoeff"] * v ** 2
    f_xroll_fl = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
    f_xroll_fr = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
    f_xroll_rl = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
    f_xroll_rr = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
    f_xroll = tire["c_roll"] * mass * g
    f_zstat_fl = 0.5 * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
    f_zstat_fr = 0.5 * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
    f_zstat_rl = 0.5 * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
    f_zstat_rr = 0.5 * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
    f_zlift_fl = 0.5 * veh["liftcoeff_front"] * v ** 2
    f_zlift_fr = 0.5 * veh["liftcoeff_front"] * v ** 2
    f_zlift_rl = 0.5 * veh["liftcoeff_rear"] * v ** 2
    f_zlift_rr = 0.5 * veh["liftcoeff_rear"] * v ** 2
    f_zdyn_fl = (-0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll) - veh["k_roll"] * gamma_y)
    f_zdyn_fr = (-0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll) + veh["k_roll"] * gamma_y)
    f_zdyn_rl = (0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll) - (1.0 - veh["k_roll"]) * gamma_y)
    f_zdyn_rr = (0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll) + (1.0 - veh["k_roll"]) * gamma_y)
    f_z_fl = f_zstat_fl + f_zlift_fl + f_zdyn_fl
    f_z_fr = f_zstat_fr + f_zlift_fr + f_zdyn_fr
    f_z_rl = f_zstat_rl + f_zlift_rl + f_zdyn_rl
    f_z_rr = f_zstat_rr + f_zlift_rr + f_zdyn_rr
    alpha_fl = delta - ca.atan((v * ca.sin(beta) + veh["wheelbase_front"] * omega_z) / (v * ca.cos(beta) - 0.5 * veh["track_width_front"] * omega_z))
    alpha_fr = delta - ca.atan((v * ca.sin(beta) + veh["wheelbase_front"] * omega_z) / (v * ca.cos(beta) + 0.5 * veh["track_width_front"] * omega_z))
    alpha_rl = ca.atan((-v * ca.sin(beta) + veh["wheelbase_rear"] * omega_z) / (v * ca.cos(beta) - 0.5 * veh["track_width_rear"] * omega_z))
    alpha_rr = ca.atan((-v * ca.sin(beta) + veh["wheelbase_rear"] * omega_z) / (v * ca.cos(beta) + 0.5 * veh["track_width_rear"] * omega_z))
    f_y_fl = (pars["optim_opts"]["mue"] * f_z_fl * (1 + tire["eps_front"] * f_z_fl / tire["f_z0"]) * ca.sin(tire["C_front"] * ca.atan(tire["B_front"] * alpha_fl - tire["E_front"] * (tire["B_front"] * alpha_fl - ca.atan(tire["B_front"] * alpha_fl)))))
    f_y_fr = (pars["optim_opts"]["mue"] * f_z_fr * (1 + tire["eps_front"] * f_z_fr / tire["f_z0"]) * ca.sin(tire["C_front"] * ca.atan(tire["B_front"] * alpha_fr - tire["E_front"] * (tire["B_front"] * alpha_fr - ca.atan(tire["B_front"] * alpha_fr)))))
    f_y_rl = (pars["optim_opts"]["mue"] * f_z_rl * (1 + tire["eps_rear"] * f_z_rl / tire["f_z0"]) * ca.sin(tire["C_rear"] * ca.atan(tire["B_rear"] * alpha_rl - tire["E_rear"] * (tire["B_rear"] * alpha_rl - ca.atan(tire["B_rear"] * alpha_rl)))))
    f_y_rr = (pars["optim_opts"]["mue"] * f_z_rr * (1 + tire["eps_rear"] * f_z_rr / tire["f_z0"]) * ca.sin(tire["C_rear"] * ca.atan(tire["B_rear"] * alpha_rr - tire["E_rear"] * (tire["B_rear"] * alpha_rr - ca.atan(tire["B_rear"] * alpha_rr)))))
    f_x_fl = 0.5 * f_drive * veh["k_drive_front"] + 0.5 * f_brake * veh["k_brake_front"] - f_xroll_fl
    f_x_fr = 0.5 * f_drive * veh["k_drive_front"] + 0.5 * f_brake * veh["k_brake_front"] - f_xroll_fr
    f_x_rl = 0.5 * f_drive * (1 - veh["k_drive_front"]) + 0.5 * f_brake * (1 - veh["k_brake_front"]) - f_xroll_rl
    f_x_rr = 0.5 * f_drive * (1 - veh["k_drive_front"]) + 0.5 * f_brake * (1 - veh["k_brake_front"]) - f_xroll_rr
    ax = (f_x_rl + f_x_rr + (f_x_fl + f_x_fr) * ca.cos(delta) - (f_y_fl + f_y_fr) * ca.sin(delta) - pars["veh_params"]["dragcoeff"] * v ** 2) / mass
    ay = ((f_x_fl + f_x_fr) * ca.sin(delta) + f_y_rl + f_y_rr + (f_y_fl + f_y_fr) * ca.cos(delta)) / mass
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        p_des = (f_drive * v * 0.001)
        machine.get_states(f_drive=f_drive, v=v)
        machine.get_loss(p_wheel=p_des)
        machine.get_machines_cum_losses()
        inverter.get_loss(i_eff=machine.i_eff, v_dc=batt.v_dc, p_out_inv=machine.p_input)
        inverter.get_inverters_cum_losses()
        batt.internal_resistance()
        batt.battery_loss(p_des=p_des, p_loss_mot=machine.p_loss_total_all_machines, p_loss_inv=inverter.p_loss_total_all_inverters, p_in_inv=inverter.p_in_inv)
        radiators.get_intermediate_temps(temp_inv=inverter.temp_inv, r_inv=inverter.r_inv)
    sf = (1.0 - n * kappa) / (v * (ca.cos(xi + beta)))
    dv = (sf / mass) * ((f_x_rl + f_x_rr) * ca.cos(beta) + (f_x_fl + f_x_fr) * ca.cos(delta - beta) + (f_y_rl + f_y_rr) * ca.sin(beta) - (f_y_fl + f_y_fr) * ca.sin(delta - beta) - f_xdrag * ca.cos(beta))
    dbeta = sf * (-omega_z + (-(f_x_rl + f_x_rr) * ca.sin(beta) + (f_x_fl + f_x_fr) * ca.sin(delta - beta) + (f_y_rl + f_y_rr) * ca.cos(beta) + (f_y_fl + f_y_fr) * ca.cos(delta - beta) + f_xdrag * ca.sin(beta)) / (mass * v))
    domega_z = (sf / veh["I_z"]) * ((f_x_rr - f_x_rl) * veh["track_width_rear"] / 2 - (f_y_rl + f_y_rr) * veh["wheelbase_rear"] + ((f_x_fr - f_x_fl) * ca.cos(delta) + (f_y_fl - f_y_fr) * ca.sin(delta)) * veh["track_width_front"] / 2 + ((f_y_fl + f_y_fr) * ca.cos(delta) + (f_x_fl + f_x_fr) * ca.sin(delta)) * veh["track_width_front"])
    dn = sf * v * ca.sin(xi + beta)
    dxi = sf * omega_z - kappa
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        machine.get_increment(sf=sf, temp_cool_12=radiators.temp_cool_12, temp_cool_13=radiators.temp_cool_13)
        inverter.get_increment(sf=sf, temp_cool_mi=radiators.temp_cool_mi, temp_cool_12=radiators.temp_cool_12)
        batt.get_increment(sf=sf, temp_cool_b=radiators.temp_cool_b)
        radiators.get_increment_mi(sf=sf, temp_mot=machine.temp_mot, temp_inv=inverter.temp_inv, r_inv=inverter.r_inv, r_machine=machine.r_machine)
        radiators.get_increment_b(sf=sf, temp_batt=batt.temp_batt, temp_cool_b=radiators.temp_cool_b, R_eq_B_inv=batt.r_batt_inverse)
        batt.get_soc(sf=sf)
        dx = ca.vertcat(dv, dbeta, domega_z, dn, dxi, machine.dtemp, batt.dtemp, inverter.dtemp, radiators.dtemp_cool_mi, radiators.dtemp_cool_b, batt.dsoc) / x_s
    else:
        dx = ca.vertcat(dv, dbeta, domega_z, dn, dxi) / x_s
    delta_min = -veh["delta_max"] / delta_s
    delta_max = veh["delta_max"] / delta_s
    f_drive_min = 0.0
    f_drive_max = veh["f_drive_max"] / f_drive_s
    f_brake_min = -veh["f_brake_max"] / f_brake_s
    f_brake_max = 0.0
    gamma_y_min = -np.inf
    gamma_y_max = np.inf
    v_min = 1.0 / v_s
    v_max_opt = pars["veh_params"]["v_max"] / v_s
    beta_min = -0.5 * np.pi / beta_s
    beta_max = 0.5 * np.pi / beta_s
    omega_z_min = - 0.5 * np.pi / omega_z_s
    omega_z_max = 0.5 * np.pi / omega_z_s
    xi_min = - 0.5 * np.pi / xi_s
    xi_max = 0.5 * np.pi / xi_s
    v_guess = 20.0 / v_s
    f_dyn = ca.Function('f_dyn', [x, u, kappa], [dx, sf], ['x', 'u', 'kappa'], ['dx', 'sf'])
    f_fx = ca.Function('f_fx', [x, u], [f_x_fl, f_x_fr, f_x_rl, f_x_rr], ['x', 'u'], ['f_x_fl', 'f_x_fr', 'f_x_rl', 'f_x_rr'])
    f_fy = ca.Function('f_fy', [x, u], [f_y_fl, f_y_fr, f_y_rl, f_y_rr], ['x', 'u'], ['f_y_fl', 'f_y_fr', 'f_y_rl', 'f_y_rr'])
    f_fz = ca.Function('f_fz', [x, u], [f_z_fl, f_z_fr, f_z_rl, f_z_rr], ['x', 'u'], ['f_z_fl', 'f_z_fr', 'f_z_rl', 'f_z_rr'])
    f_a = ca.Function('f_a', [x, u], [ax, ay], ['x', 'u'], ['ax', 'ay'])
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        machine.ini_nlp_state(x=x, u=u)
        inverter.ini_nlp_state(x=x, u=u)
        batt.ini_nlp_state(x=x, u=u)
        radiators.ini_nlp_state(x=x, u=u)
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    x_opt = []
    u_opt = []
    dt_opt = []
    tf_opt = []
    ax_opt = []
    ay_opt = []
    ec_opt = []
    delta_p = []
    F_p = []
    Xk = ca.MX.sym('X0', nx)
    w.append(Xk)
    # Correction: Cast interpolated track widths to float to avoid mixed types in lbw/ubw
    n_min = (float(-w_tr_right_interp(0)) + pars["optim_opts"]["width_opt"] / 2) / n_s
    n_max = (float(w_tr_left_interp(0)) - pars["optim_opts"]["width_opt"] / 2) / n_s
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min, machine.temp_min, batt.temp_min, inverter.temp_min, radiators.temp_cool_mi_min, radiators.temp_cool_b_min, batt.soc_min])
        ubw.append([v_max_opt, beta_max, omega_z_max, n_max, xi_max, machine.temp_max, batt.temp_max, inverter.temp_max, radiators.temp_cool_mi_max, radiators.temp_cool_b_max, batt.soc_max])
        w0.append([v_guess, 0.0, 0.0, 0.0, 0.0, machine.temp_guess, batt.temp_guess, inverter.temp_guess, radiators.temp_cool_mi_guess, radiators.temp_cool_b_guess, batt.soc_guess])
        g.append(Xk[5] - pars["pwr_params_mintime"]["T_mot_ini"] / machine.temp_mot_s)
        lbg.append([0])
        ubg.append([0])
        g.append(Xk[6] - pars["pwr_params_mintime"]["T_batt_ini"] / batt.temp_batt_s)
        lbg.append([0])
        ubg.append([0])
        g.append(Xk[7] - pars["pwr_params_mintime"]["T_inv_ini"] / inverter.temp_inv_s)
        lbg.append([0])
        ubg.append([0])
        g.append(Xk[8] - pars["pwr_params_mintime"]["T_cool_mi_ini"] / radiators.temp_cool_mi_s)
        lbg.append([0])
        ubg.append([0])
        g.append(Xk[9] - pars["pwr_params_mintime"]["T_cool_b_ini"] / radiators.temp_cool_b_s)
        lbg.append([0])
        ubg.append([0])
        g.append(Xk[10] - pars["pwr_params_mintime"]["SOC_ini"] / batt.soc_batt_s)
        lbg.append([0])
        ubg.append([0])
    else:
        lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min])
        ubw.append([v_max_opt, beta_max, omega_z_max, n_max, xi_max])
        w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])
    x_opt.append(Xk * x_s)
    h = np.diff(s_opt)
    for k in range(N):
        Uk = ca.MX.sym('U_' + str(k), nu)
        w.append(Uk)
        lbw.append([delta_min, f_drive_min, f_brake_min, gamma_y_min])
        ubw.append([delta_max, f_drive_max, f_brake_max, gamma_y_max])
        w0.append([0.0] * nu)
        Xc = []
        for j in range(d):
            Xkj = ca.MX.sym('X_' + str(k) + '_' + str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-np.inf] * nx)
            ubw.append([np.inf] * nx)
            if pars["pwr_params_mintime"]["pwr_behavior"]:
                w0.append([v_guess, 0.0, 0.0, 0.0, 0.0, machine.temp_guess, batt.temp_guess, inverter.temp_guess, radiators.temp_cool_mi_guess, radiators.temp_cool_b_guess, batt.soc_guess])
            else:
                w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])
        Xk_end = D[0] * Xk
        sf_opt = []
        for j in range(1, d + 1):
            xp = C[0, j] * Xk
            for r in range(d):
                xp = xp + C[r + 1, j] * Xc[r]
            kappa_col = kappa_interp(k + tau[j])
            fj, qj = f_dyn(Xc[j - 1], Uk, kappa_col)
            g.append(h[k] * fj - xp)
            lbg.append([0.0] * nx)
            ubg.append([0.0] * nx)
            Xk_end = Xk_end + D[j] * Xc[j - 1]
            J = J + B[j] * qj * h[k]
            sf_opt.append(B[j] * qj * h[k])
        dt_opt.append(sf_opt[0] + sf_opt[1] + sf_opt[2])
        if pars["pwr_params_mintime"]["pwr_behavior"]:
            ec_opt.append((batt.f_nlp(Xk, Uk)[0] + batt.f_nlp(Xk, Uk)[1]) * 1000 * dt_opt[-1])
        else:
            ec_opt.append(Xk[0] * v_s * Uk[1] * f_drive_s * dt_opt[-1])
        Xk = ca.MX.sym('X_' + str(k + 1), nx)
        w.append(Xk)
        # Correction: Cast interpolated track widths to float here as well
        n_min = (float(-w_tr_right_interp(k + 1)) + pars["optim_opts"]["width_opt"] / 2.0) / n_s
        n_max = (float(w_tr_left_interp(k + 1)) - pars["optim_opts"]["width_opt"] / 2.0) / n_s
        if pars["pwr_params_mintime"]["pwr_behavior"]:
            lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min, machine.temp_min, batt.temp_min, inverter.temp_min, radiators.temp_cool_mi_min, radiators.temp_cool_b_min, batt.soc_min])
            ubw.append([v_max_opt, beta_max, omega_z_max, n_max, xi_max, machine.temp_max, batt.temp_max, inverter.temp_max, radiators.temp_cool_mi_max, radiators.temp_cool_b_max, batt.soc_max])
            w0.append([v_guess, 0.0, 0.0, 0.0, 0.0, machine.temp_guess, batt.temp_guess, inverter.temp_guess, radiators.temp_cool_mi_guess, radiators.temp_cool_mi_guess, batt.soc_guess])
        else:
            lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min])
            ubw.append([v_max_opt, beta_max, omega_z_max, n_max, xi_max])
            w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])
        g.append(Xk_end - Xk)
        lbg.append([0.0] * nx)
        ubg.append([0.0] * nx)
        f_x_flk, f_x_frk, f_x_rlk, f_x_rrk = f_fx(Xk, Uk)
        f_y_flk, f_y_frk, f_y_rlk, f_y_rrk = f_fy(Xk, Uk)
        f_z_flk, f_z_frk, f_z_rlk, f_z_rrk = f_fz(Xk, Uk)
        axk, ayk = f_a(Xk, Uk)
        g.append(Xk[0] * Uk[1])
        lbg.append([-np.inf])
        ubg.append([veh["power_max"] / (f_drive_s * v_s)])
        mue_fl = pars["optim_opts"]["mue"]
        mue_fr = pars["optim_opts"]["mue"]
        mue_rl = pars["optim_opts"]["mue"]
        mue_rr = pars["optim_opts"]["mue"]
        g.append(((f_x_flk / (mue_fl * f_z_flk)) ** 2 + (f_y_flk / (mue_fl * f_z_flk)) ** 2))
        g.append(((f_x_frk / (mue_fr * f_z_frk)) ** 2 + (f_y_frk / (mue_fr * f_z_frk)) ** 2))
        g.append(((f_x_rlk / (mue_rl * f_z_rlk)) ** 2 + (f_y_rlk / (mue_rl * f_z_rlk)) ** 2))
        g.append(((f_x_rrk / (mue_rr * f_z_rrk)) ** 2 + (f_y_rrk / (mue_rr * f_z_rrk)) ** 2))
        lbg.append([0.0] * 4)
        ubg.append([1.0] * 4)
        g.append(((f_y_flk + f_y_frk) * ca.cos(Uk[0] * delta_s) + f_y_rlk + f_y_rrk + (f_x_flk + f_x_frk) * ca.sin(Uk[0] * delta_s)) * veh["cog_z"] / ((veh["track_width_front"] + veh["track_width_rear"]) / 2) - Uk[3] * gamma_y_s)
        lbg.append([0.0])
        ubg.append([0.0])
        g.append(Uk[1] * Uk[2])
        lbg.append([-20000.0 / (f_drive_s * f_brake_s)])
        ubg.append([0.0])
        if k > 0:
            sigma = (1 - kappa_interp(k) * Xk[3] * n_s) / (Xk[0] * v_s)
            g.append((Uk - w[1 + (k - 1) * (nx - nx_pwr)]) / (h[k - 1] * sigma))
            lbg.append([delta_min / (veh["t_delta"]), -np.inf, f_brake_min / (veh["t_brake"]), -np.inf])
            ubg.append([delta_max / (veh["t_delta"]), f_drive_max / (veh["t_drive"]), np.inf, np.inf])
        if pars["optim_opts"]["safe_traj"]:
            g.append((ca.fmax(axk, 0) / pars["optim_opts"]["ax_pos_safe"]) ** 2 + (ayk / pars["optim_opts"]["ay_safe"]) ** 2)
            g.append((ca.fmin(axk, 0) / pars["optim_opts"]["ax_neg_safe"]) ** 2 + (ayk / pars["optim_opts"]["ay_safe"]) ** 2)
            lbg.append([0.0] * 2)
            ubg.append([1.0] * 2)
        delta_p.append(Uk[0] * delta_s)
        F_p.append(Uk[1] * f_drive_s / 10000.0 + Uk[2] * f_brake_s / 10000.0)
        x_opt.append(Xk * x_s)
        u_opt.append(Uk * u_s)
        tf_opt.extend([f_x_flk, f_y_flk, f_z_flk, f_x_frk, f_y_frk, f_z_frk])
        tf_opt.extend([f_x_rlk, f_y_rlk, f_z_rlk, f_x_rrk, f_y_rrk, f_z_rrk])
        ax_opt.append(axk)
        ay_opt.append(ayk)
        if pars["pwr_params_mintime"]["pwr_behavior"]:
            machine.p_losses_opt.extend(machine.f_nlp(Xk, Uk))
            inverter.p_losses_opt.extend(inverter.f_nlp(Xk, Uk))
            batt.p_losses_opt.extend(batt.f_nlp(Xk, Uk))
            radiators.temps_opt.extend(radiators.f_nlp(Xk, Uk))
    g.append(w[0] - Xk)
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        lbg.append([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        ubg.append([0.0, 0.0, 0.0, 0.0, 0.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    else:
        lbg.append([0.0, 0.0, 0.0, 0.0, 0.0])
        ubg.append([0.0, 0.0, 0.0, 0.0, 0.0])
    if pars["optim_opts"]["limit_energy"]:
        g.append(ca.sum1(ca.vertcat(*ec_opt)) / 3600000.0)
        lbg.append([0])
        ubg.append([pars["optim_opts"]["energy_limit"]])
    diff_matrix = np.eye(N)
    for i in range(N - 1):
        diff_matrix[i, i + 1] = -1.0
    diff_matrix[N - 1, 0] = -1.0
    delta_p = ca.vertcat(*delta_p)
    Jp_delta = ca.mtimes(ca.MX(diff_matrix), delta_p)
    Jp_delta = ca.dot(Jp_delta, Jp_delta)
    F_p = ca.vertcat(*F_p)
    Jp_f = ca.mtimes(ca.MX(diff_matrix), F_p)
    Jp_f = ca.dot(Jp_f, Jp_f)
    J = J + pars["optim_opts"]["penalty_F"] * Jp_f + pars["optim_opts"]["penalty_delta"] * Jp_delta
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)
    x_opt = ca.vertcat(*x_opt)
    u_opt = ca.vertcat(*u_opt)
    tf_opt = ca.vertcat(*tf_opt)
    dt_opt = ca.vertcat(*dt_opt)
    ax_opt = ca.vertcat(*ax_opt)
    ay_opt = ca.vertcat(*ay_opt)
    ec_opt = ca.vertcat(*ec_opt)
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        machine.p_losses_opt = ca.vertcat(*machine.p_losses_opt)
        inverter.p_losses_opt = ca.vertcat(*inverter.p_losses_opt)
        batt.p_losses_opt = ca.vertcat(*batt.p_losses_opt)
        radiators.temps_opt = ca.vertcat(*radiators.temps_opt)
    nlp = {'f': J, 'x': w, 'g': g}
    opts = {"expand": True, "verbose": print_debug, "ipopt.max_iter": 2000, "ipopt.tol": 1e-7}
    if pars["optim_opts"]["warm_start"]:
        opts_warm_start = {"ipopt.warm_start_init_point": "yes", "ipopt.warm_start_bound_push": 1e-3, "ipopt.warm_start_mult_bound_push": 1e-3, "ipopt.warm_start_slack_bound_push": 1e-3, "ipopt.mu_init": 1e-3}
        opts.update(opts_warm_start)
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    f_sol = ca.Function('f_sol', [w], [x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt, ec_opt], ['w'], ['x_opt', 'u_opt', 'tf_opt', 'dt_opt', 'ax_opt', 'ay_opt', 'ec_opt'])
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        machine.extract_sol(w=w, sol_states=sol['x'])
        inverter.extract_sol(w=w, sol_states=sol['x'])
        batt.extract_sol(w=w, sol_states=sol['x'])
        radiators.extract_sol(w=w, sol_states=sol['x'])
    x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt, ec_opt = f_sol(sol['x'])
    x_opt = np.reshape(x_opt, (N + 1, nx))
    u_opt = np.reshape(u_opt, (N, nu))
    tf_opt = np.append(tf_opt[-12:], tf_opt[:])
    tf_opt = np.reshape(tf_opt, (N + 1, 12))
    t_opt = np.hstack((0.0, np.cumsum(dt_opt)))
    ax_opt = np.append(ax_opt[-1], ax_opt)
    ay_opt = np.append(ay_opt[-1], ay_opt)
    atot_opt = np.sqrt(np.power(ax_opt, 2) + np.power(ay_opt, 2))
    ec_opt_cum = np.hstack((0.0, np.cumsum(ec_opt))) / 3600.0
    return -x_opt[:-1, 3], x_opt[:-1, 0], reftrack, a_interp, normvectors

def run_optimization(track_file_path, pars, opt_type='mintime', imp_opts=None, mintime_opts=None):
    if imp_opts is None:
        imp_opts = {"flip_imp_track": False, "set_new_start": False, "new_start": np.array([0.0, 0.0]), "min_track_width": None, "num_laps": 1}
    if mintime_opts is None:
        mintime_opts = {"reopt_mintime_solution": False, "recalc_vel_profile_by_tph": False}
    reftrack_imp = import_track(file_path=track_file_path, imp_opts=imp_opts, width_veh=pars["veh_params"]["width"])
    reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = prep_track(reftrack_imp=reftrack_imp, reg_smooth_opts=pars["reg_smooth_opts"], stepsize_opts=pars["stepsize_opts"], debug=False, min_width=imp_opts["min_track_width"])
    if opt_type == 'mincurv':
        alpha_opt = math_utils.opt_min_curv(reftrack=reftrack_interp, normvectors=normvec_normalized_interp, A=a_interp, kappa_bound=pars["veh_params"]["curvlim"], w_veh=pars["optim_opts"]["width_opt"], print_debug=False, plot_debug=False)[0]
    elif opt_type == 'mincurv_iqp':
        alpha_opt, reftrack_interp, normvec_normalized_interp = math_utils.iqp_handler(reftrack=reftrack_interp, normvectors=normvec_normalized_interp, A=a_interp, kappa_bound=pars["veh_params"]["curvlim"], w_veh=pars["optim_opts"]["width_opt"], print_debug=False, plot_debug=False, stepsize_interp=pars["stepsize_opts"]["stepsize_reg"], iters_min=pars["optim_opts"]["iqp_iters_min"], curv_error_allowed=pars["optim_opts"]["iqp_curverror_allowed"])
    elif opt_type == 'shortest_path':
        alpha_opt = math_utils.opt_shortest_path(reftrack=reftrack_interp, normvectors=normvec_normalized_interp, w_veh=pars["optim_opts"]["width_opt"], print_debug=False)
    elif opt_type == 'mintime':
        alpha_opt, v_opt, reftrack_interp, a_interp_tmp, normvec_normalized_interp = opt_mintime(reftrack=reftrack_interp, coeffs_x=coeffs_x_interp, coeffs_y=coeffs_y_interp, normvectors=normvec_normalized_interp, pars=pars, print_debug=False, plot_debug=False)
        if a_interp_tmp is not None:
            a_interp = a_interp_tmp
    if opt_type == 'mintime' and mintime_opts["reopt_mintime_solution"]:
        raceline_mintime = reftrack_interp[:, :2] + np.expand_dims(alpha_opt, 1) * normvec_normalized_interp
        w_tr_right_mintime = reftrack_interp[:, 2] - alpha_opt
        w_tr_left_mintime = reftrack_interp[:, 3] + alpha_opt
        racetrack_mintime = np.column_stack((raceline_mintime, w_tr_right_mintime, w_tr_left_mintime))
        reftrack_interp, normvec_normalized_interp, a_interp = prep_track(reftrack_imp=racetrack_mintime, reg_smooth_opts=pars["reg_smooth_opts"], stepsize_opts=pars["stepsize_opts"], debug=False, min_width=imp_opts["min_track_width"])[:3]
        w_tr_tmp = 0.5 * pars["optim_opts"]["w_tr_reopt"] * np.ones(reftrack_interp.shape[0])
        racetrack_mintime_reopt = np.column_stack((reftrack_interp[:, :2], w_tr_tmp, w_tr_tmp))
        alpha_opt = math_utils.opt_min_curv(reftrack=racetrack_mintime_reopt, normvectors=normvec_normalized_interp, A=a_interp, kappa_bound=pars["veh_params"]["curvlim"], w_veh=pars["optim_opts"]["w_veh_reopt"], print_debug=False, plot_debug=False)[0]
    raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp, spline_lengths_opt, el_lengths_opt_interp = math_utils.create_raceline(refline=reftrack_interp[:, :2], normvectors=normvec_normalized_interp, alpha=alpha_opt, stepsize_interp=pars["stepsize_opts"]["stepsize_interp_after_opt"])
    psi_vel_opt, kappa_opt = math_utils.calc_head_curv_an(coeffs_x=coeffs_x_opt, coeffs_y=coeffs_y_opt, ind_spls=spline_inds_opt_interp, t_spls=t_vals_opt_interp)
    if opt_type == 'mintime' and not mintime_opts["recalc_vel_profile_by_tph"]:
        s_splines = np.cumsum(spline_lengths_opt)
        s_splines = np.insert(s_splines, 0, 0.0)
        vx_profile_opt = np.interp(s_points_opt_interp, s_splines[:-1], v_opt)
    else:
        # -------------------------------------------------------------
        # PROFIL DE VITESSE SIMPLIFIÉ MAIS PHYSIQUE
        # -------------------------------------------------------------
        # On ne dispose pas ici d’un vrai g-g-v (ggv) ni d’ax_max_machines,
        # donc on construit un profil de vitesse basé sur la courbure :
        #
        #   v_max_courbe(s) = sqrt( mu * g / |kappa(s)| )
        #
        # (norme de l’accélération latérale = v^2 * |kappa| <= mu * g)
        #
        # Ensuite on sature par v_max (vitesse max du véhicule),
        # et on lisse avec le filtre moyenne glissante de math_utils.conv_filt.
        # -------------------------------------------------------------
        g = pars["veh_params"]["g"]
        mu = pars["optim_opts"]["mue"]
        v_max = pars["veh_params"]["v_max"]

        # éviter la division par zéro sur les lignes droites
        kappa_abs = np.abs(kappa_opt)
        kappa_abs[kappa_abs < 1e-6] = 1e-6

        # vitesse limite par la courbure
        v_curve = np.sqrt(mu * g / kappa_abs)

        # on sature par la vitesse max véhicule
        vx_profile_opt = np.minimum(v_curve, v_max)

        # remplace les NaN / inf éventuels (par ex. si kappa ~ 0)
        vx_profile_opt = np.nan_to_num(
            vx_profile_opt,
            nan=v_max,
            posinf=v_max,
            neginf=0.0,
        )

        # lissage optionnel
        filt_window = pars["vel_calc_opts"]["vel_profile_conv_filt_window"]
        if filt_window is not None:
            # s’assurer que la fenêtre est impaire pour conv_filt
            if not (filt_window % 2 == 1):
                filt_window += 1
            vx_profile_opt = math_utils.conv_filt(
                signal=vx_profile_opt,
                filt_window=filt_window,
                closed=True,
            )
    vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
    ax_profile_opt = math_utils.calc_ax_profile(vx_profile=vx_profile_opt_cl, el_lengths=el_lengths_opt_interp, eq_length_output=False)
    trajectory_opt = np.column_stack((s_points_opt_interp, raceline_interp, psi_vel_opt, kappa_opt, vx_profile_opt, ax_profile_opt))
    check_traj(reftrack=reftrack_interp, reftrack_normvec_normalized=normvec_normalized_interp, trajectory=trajectory_opt, ggv=None, ax_max_machines=None, v_max=pars["veh_params"]["v_max"], length_veh=pars["veh_params"]["length"], width_veh=pars["veh_params"]["width"], debug=False, dragcoeff=pars["veh_params"]["dragcoeff"], mass_veh=pars["veh_params"]["mass"], curvlim=pars["veh_params"]["curvlim"])
    return trajectory_opt