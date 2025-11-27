import numpy as np
import math
import sys
import time
import quadprog
from scipy import interpolate
from scipy import optimize
from scipy import spatial
from typing import Union

def angle3pt(a, b, c):
    ang = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    if ang >= math.pi:
        ang -= 2 * math.pi
    elif ang < -math.pi:
        ang += 2 * math.pi
    return ang

def calc_ax_profile(vx_profile, el_lengths, eq_length_output=False):
    if vx_profile.size != el_lengths.size + 1:
        raise RuntimeError("Array size of vx_profile should be 1 element bigger than el_lengths!")
    if eq_length_output:
        ax_profile = np.zeros(vx_profile.size)
        ax_profile[:-1] = (np.power(vx_profile[1:], 2) - np.power(vx_profile[:-1], 2)) / (2 * el_lengths)
    else:
        ax_profile = (np.power(vx_profile[1:], 2) - np.power(vx_profile[:-1], 2)) / (2 * el_lengths)
    return ax_profile

def normalize_psi(psi):
    psi_out = np.sign(psi) * np.mod(np.abs(psi), 2 * math.pi)
    if type(psi_out) is np.ndarray:
        psi_out[psi_out >= math.pi] -= 2 * math.pi
        psi_out[psi_out < -math.pi] += 2 * math.pi
    else:
        if psi_out >= math.pi:
            psi_out -= 2 * math.pi
        elif psi_out < -math.pi:
            psi_out += 2 * math.pi
    return psi_out

def calc_head_curv_an(coeffs_x, coeffs_y, ind_spls, t_spls, calc_curv=True, calc_dcurv=False):
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise ValueError("Coefficient matrices must have the same length!")
    if ind_spls.size != t_spls.size:
        raise ValueError("ind_spls and t_spls must have the same length!")
    if not calc_curv and calc_dcurv:
        raise ValueError("dkappa cannot be calculated without kappa!")
    x_d = coeffs_x[ind_spls, 1] \
          + 2 * coeffs_x[ind_spls, 2] * t_spls \
          + 3 * coeffs_x[ind_spls, 3] * np.power(t_spls, 2)
    y_d = coeffs_y[ind_spls, 1] \
          + 2 * coeffs_y[ind_spls, 2] * t_spls \
          + 3 * coeffs_y[ind_spls, 3] * np.power(t_spls, 2)
    psi = np.arctan2(y_d, x_d) - math.pi / 2
    psi = normalize_psi(psi)
    if calc_curv:
        x_dd = 2 * coeffs_x[ind_spls, 2] \
               + 6 * coeffs_x[ind_spls, 3] * t_spls
        y_dd = 2 * coeffs_y[ind_spls, 2] \
               + 6 * coeffs_y[ind_spls, 3] * t_spls
        kappa = (x_d * y_dd - y_d * x_dd) / np.power(np.power(x_d, 2) + np.power(y_d, 2), 1.5)
    else:
        kappa = 0.0
    if calc_dcurv:
        x_ddd = 6 * coeffs_x[ind_spls, 3]
        y_ddd = 6 * coeffs_y[ind_spls, 3]
        dkappa = ((np.power(x_d, 2) + np.power(y_d, 2)) * (x_d * y_ddd - y_d * x_ddd) -
                  3 * (x_d * y_dd - y_d * x_dd) * (x_d * x_dd + y_d * y_dd)) / \
                 np.power(np.power(x_d, 2) + np.power(y_d, 2), 3)
        return psi, kappa, dkappa
    else:
        return psi, kappa

def calc_head_curv_num(path, el_lengths, is_closed, stepsize_psi_preview=1.0, stepsize_psi_review=1.0, stepsize_curv_preview=2.0, stepsize_curv_review=2.0, calc_curv=True):
    if is_closed and path.shape[0] != el_lengths.size:
        raise RuntimeError("path and el_lenghts must have the same length!")
    elif not is_closed and path.shape[0] != el_lengths.size + 1:
        raise RuntimeError("path must have the length of el_lengths + 1!")
    no_points = path.shape[0]
    if is_closed:
        ind_step_preview_psi = round(stepsize_psi_preview / float(np.average(el_lengths)))
        ind_step_review_psi = round(stepsize_psi_review / float(np.average(el_lengths)))
        ind_step_preview_curv = round(stepsize_curv_preview / float(np.average(el_lengths)))
        ind_step_review_curv = round(stepsize_curv_review / float(np.average(el_lengths)))
        ind_step_preview_psi = max(ind_step_preview_psi, 1)
        ind_step_review_psi = max(ind_step_review_psi, 1)
        ind_step_preview_curv = max(ind_step_preview_curv, 1)
        ind_step_review_curv = max(ind_step_review_curv, 1)
        steps_tot_psi = ind_step_preview_psi + ind_step_review_psi
        steps_tot_curv = ind_step_preview_curv + ind_step_review_curv
        path_temp = np.vstack((path[-ind_step_review_psi:], path, path[:ind_step_preview_psi]))
        tangvecs = np.stack((path_temp[steps_tot_psi:, 0] - path_temp[:-steps_tot_psi, 0],
                             path_temp[steps_tot_psi:, 1] - path_temp[:-steps_tot_psi, 1]), axis=1)
        psi = np.arctan2(tangvecs[:, 1], tangvecs[:, 0]) - math.pi / 2
        psi = normalize_psi(psi)
        if calc_curv:
            psi_temp = np.insert(psi, 0, psi[-ind_step_review_curv:])
            psi_temp = np.append(psi_temp, psi[:ind_step_preview_curv])
            delta_psi = np.zeros(no_points)
            for i in range(no_points):
                delta_psi[i] = normalize_psi(psi_temp[i + steps_tot_curv] - psi_temp[i])
            s_points_cl = np.cumsum(el_lengths)
            s_points_cl = np.insert(s_points_cl, 0, 0.0)
            s_points = s_points_cl[:-1]
            s_points_cl_reverse = np.flipud(-np.cumsum(np.flipud(el_lengths)))
            s_points_temp = np.insert(s_points, 0, s_points_cl_reverse[-ind_step_review_curv:])
            s_points_temp = np.append(s_points_temp, s_points_cl[-1] + s_points[:ind_step_preview_curv])
            kappa = delta_psi / (s_points_temp[steps_tot_curv:] - s_points_temp[:-steps_tot_curv])
        else:
            kappa = 0.0
    else:
        tangvecs = np.zeros((no_points, 2))
        tangvecs[0, 0] = path[1, 0] - path[0, 0]
        tangvecs[0, 1] = path[1, 1] - path[0, 1]
        tangvecs[1:-1, 0] = path[2:, 0] - path[:-2, 0]
        tangvecs[1:-1, 1] = path[2:, 1] - path[:-2, 1]
        tangvecs[-1, 0] = path[-1, 0] - path[-2, 0]
        tangvecs[-1, 1] = path[-1, 1] - path[-2, 1]
        psi = np.arctan2(tangvecs[:, 1], tangvecs[:, 0]) - math.pi / 2
        psi = normalize_psi(psi)
        if calc_curv:
            delta_psi = np.zeros(no_points)
            delta_psi[0] = psi[1] - psi[0]
            delta_psi[1:-1] = psi[2:] - psi[:-2]
            delta_psi[-1] = psi[-1] - psi[-2]
            delta_psi = normalize_psi(delta_psi)
            kappa = np.zeros(no_points)
            kappa[0] = delta_psi[0] / el_lengths[0]
            kappa[1:-1] = delta_psi[1:-1] / (el_lengths[1:] + el_lengths[:-1])
            kappa[-1] = delta_psi[-1] / el_lengths[-1]
        else:
            kappa = 0.0
    return psi, kappa

def calc_tangent_vectors(psi):
    psi_ = np.copy(psi)
    psi_ += math.pi / 2
    psi_ = normalize_psi(psi_)
    tangvec_normalized = np.zeros((psi_.size, 2))
    tangvec_normalized[:, 0] = np.cos(psi_)
    tangvec_normalized[:, 1] = np.sin(psi_)
    return tangvec_normalized

def calc_normal_vectors_ahead(psi):
    tangvec_normalized = calc_tangent_vectors(psi=psi)
    normvec_normalized = np.stack((-tangvec_normalized[:, 1], tangvec_normalized[:, 0]), axis=1)
    return normvec_normalized

def calc_normal_vectors(psi):
    normvec_normalized = -calc_normal_vectors_ahead(psi=psi)
    return normvec_normalized

def calc_spline_lengths(coeffs_x, coeffs_y, quickndirty=False, no_interp_points=15):
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise RuntimeError("Coefficient matrices must have the same length!")
    if coeffs_x.size == 4 and coeffs_x.shape[0] == 4:
        coeffs_x = np.expand_dims(coeffs_x, 0)
        coeffs_y = np.expand_dims(coeffs_y, 0)
    no_splines = coeffs_x.shape[0]
    spline_lengths = np.zeros(no_splines)
    if quickndirty:
        for i in range(no_splines):
            spline_lengths[i] = math.sqrt(math.pow(np.sum(coeffs_x[i]) - coeffs_x[i, 0], 2)
                                          + math.pow(np.sum(coeffs_y[i]) - coeffs_y[i, 0], 2))
    else:
        t_steps = np.linspace(0.0, 1.0, no_interp_points)
        spl_coords = np.zeros((no_interp_points, 2))
        for i in range(no_splines):
            spl_coords[:, 0] = coeffs_x[i, 0] \
                               + coeffs_x[i, 1] * t_steps \
                               + coeffs_x[i, 2] * np.power(t_steps, 2) \
                               + coeffs_x[i, 3] * np.power(t_steps, 3)
            spl_coords[:, 1] = coeffs_y[i, 0] \
                               + coeffs_y[i, 1] * t_steps \
                               + coeffs_y[i, 2] * np.power(t_steps, 2) \
                               + coeffs_y[i, 3] * np.power(t_steps, 3)
            spline_lengths[i] = np.sum(np.sqrt(np.sum(np.power(np.diff(spl_coords, axis=0), 2), axis=1)))
    return spline_lengths

def calc_splines(path, el_lengths=None, psi_s=None, psi_e=None, use_dist_scaling=True):
    if np.all(np.isclose(path[0], path[-1])) and psi_s is None:
        closed = True
    else:
        closed = False
    if not closed and (psi_s is None or psi_e is None):
        raise RuntimeError("Headings must be provided for unclosed spline calculation!")
    if el_lengths is not None and path.shape[0] != el_lengths.size + 1:
        raise RuntimeError("el_lengths input must be one element smaller than path input!")
    if use_dist_scaling and el_lengths is None:
        el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))
    elif el_lengths is not None:
        el_lengths = np.copy(el_lengths)
    if use_dist_scaling and closed:
        el_lengths = np.append(el_lengths, el_lengths[0])
    no_splines = path.shape[0] - 1
    if use_dist_scaling:
        scaling = el_lengths[:-1] / el_lengths[1:]
    else:
        scaling = np.ones(no_splines - 1)
    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))
    template_M = np.array(
                [[1,  0,  0,  0,  0,  0,  0,  0],
                 [1,  1,  1,  1,  0,  0,  0,  0],
                 [0,  1,  2,  3,  0, -1,  0,  0],
                 [0,  0,  2,  6,  0,  0, -2,  0]])
    for i in range(no_splines):
        j = i * 4
        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M
            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)
        else:
            M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],
                                     [1,  1,  1,  1]]
        b_x[j: j + 2] = [[path[i,     0]],
                         [path[i + 1, 0]]]
        b_y[j: j + 2] = [[path[i,     1]],
                         [path[i + 1, 1]]]
    if not closed:
        M[-2, 1] = 1
        if el_lengths is None:
            el_length_s = 1.0
        else:
            el_length_s = el_lengths[0]
        b_x[-2] = math.cos(psi_s + math.pi / 2) * el_length_s
        b_y[-2] = math.sin(psi_s + math.pi / 2) * el_length_s
        M[-1, -4:] = [0, 1, 2, 3]
        if el_lengths is None:
            el_length_e = 1.0
        else:
            el_length_e = el_lengths[-1]
        b_x[-1] = math.cos(psi_e + math.pi / 2) * el_length_e
        b_y[-1] = math.sin(psi_e + math.pi / 2) * el_length_e
    else:
        M[-2, 1] = scaling[-1]
        M[-2, -3:] = [-1, -2, -3]
        M[-1, 2] = 2 * math.pow(scaling[-1], 2)
        M[-1, -2:] = [-2, -6]
    x_les = np.squeeze(np.linalg.solve(M, b_x))
    y_les = np.squeeze(np.linalg.solve(M, b_y))
    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    normvec_normalized = np.expand_dims(norm_factors, axis=1) * normvec
    return coeffs_x, coeffs_y, M, normvec_normalized

def calc_t_profile(vx_profile, el_lengths, t_start=0.0, ax_profile=None):
    if vx_profile.size < el_lengths.size:
        raise RuntimeError("vx_profile and el_lenghts must have at least the same length!")
    if ax_profile is not None and ax_profile.size < el_lengths.size:
        raise RuntimeError("ax_profile and el_lenghts must have at least the same length!")
    if ax_profile is None:
        ax_profile = calc_ax_profile(vx_profile=vx_profile,
                                     el_lengths=el_lengths,
                                     eq_length_output=False)
    no_points = el_lengths.size
    t_steps = np.zeros(no_points)
    for i in range(no_points):
        if not math.isclose(ax_profile[i], 0.0):
            t_steps[i] = (-vx_profile[i] + math.sqrt((math.pow(vx_profile[i], 2) + 2 * ax_profile[i] * el_lengths[i])))\
                         / ax_profile[i]
        else:
            t_steps[i] = el_lengths[i] / vx_profile[i]
    t_profile = np.insert(np.cumsum(t_steps), 0, 0.0) + t_start
    return t_profile

def conv_filt(signal, filt_window, closed):
    if not filt_window % 2 == 1:
        raise RuntimeError("Window width of moving average filter must be odd!")
    w_window_half = int((filt_window - 1) / 2)
    if closed:
        signal_tmp = np.concatenate((signal[-w_window_half:], signal, signal[:w_window_half]), axis=0)
        signal_filt = np.convolve(signal_tmp,
                                  np.ones(filt_window) / float(filt_window),
                                  mode="same")[w_window_half:-w_window_half]
    else:
        signal_filt = np.copy(signal)
        signal_filt[w_window_half:-w_window_half] = np.convolve(signal,
                                                                np.ones(filt_window) / float(filt_window),
                                                                mode="same")[w_window_half:-w_window_half]
    return signal_filt

def calc_ax_poss(vx_start, radius, ggv, mu, dyn_model_exp, drag_coeff, m_veh, ax_max_machines=None, mode='accel_forw'):
    if mode not in ['accel_forw', 'decel_forw', 'decel_backw']:
        raise RuntimeError("Unknown operation mode for calc_ax_poss!")
    if mode == 'accel_forw' and ax_max_machines is None:
        raise RuntimeError("ax_max_machines is required if operation mode is accel_forw!")
    if ggv.ndim != 2 or ggv.shape[1] != 3:
        raise RuntimeError("ggv must have two dimensions and three columns [vx, ax_max, ay_max]!")
    ax_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 1])
    ay_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 2])
    ay_used = math.pow(vx_start, 2) / radius
    if mode in ['accel_forw', 'decel_backw'] and ax_max_tires < 0.0:
        # print("WARNING: Inverting sign of ax_max_tires because it should be positive but was negative!")
        ax_max_tires *= -1.0
    elif mode == 'decel_forw' and ax_max_tires > 0.0:
        # print("WARNING: Inverting sign of ax_max_tires because it should be negative but was positve!")
        ax_max_tires *= -1.0
    radicand = 1.0 - math.pow(ay_used / ay_max_tires, dyn_model_exp)
    if radicand > 0.0:
        ax_avail_tires = ax_max_tires * math.pow(radicand, 1.0 / dyn_model_exp)
    else:
        ax_avail_tires = 0.0
    if mode == 'accel_forw':
        ax_max_machines_tmp = np.interp(vx_start, ax_max_machines[:, 0], ax_max_machines[:, 1])
        ax_avail_vehicle = min(ax_avail_tires, ax_max_machines_tmp)
    else:
        ax_avail_vehicle = ax_avail_tires
    ax_drag = -math.pow(vx_start, 2) * drag_coeff / m_veh
    if mode in ['accel_forw', 'decel_forw']:
        ax_final = ax_avail_vehicle + ax_drag
    else:
        ax_final = ax_avail_vehicle - ax_drag
    return ax_final

def __solver_fb_acc_profile(p_ggv, ax_max_machines, v_max, radii, el_lengths, mu, vx_profile, drag_coeff, m_veh, dyn_model_exp=1.0, backwards=False):
    no_points = vx_profile.size
    if backwards:
        radii_mod = np.flipud(radii)
        el_lengths_mod = np.flipud(el_lengths)
        mu_mod = np.flipud(mu)
        vx_profile = np.flipud(vx_profile)
        mode = 'decel_backw'
    else:
        radii_mod = radii
        el_lengths_mod = el_lengths
        mu_mod = mu
        mode = 'accel_forw'
    vx_diffs = np.diff(vx_profile)
    acc_inds = np.where(vx_diffs > 0.0)[0]
    if acc_inds.size != 0:
        acc_inds_diffs = np.diff(acc_inds)
        acc_inds_diffs = np.insert(acc_inds_diffs, 0, 2)
        acc_inds_rel = acc_inds[acc_inds_diffs > 1]
    else:
        acc_inds_rel = []
    acc_inds_rel = list(acc_inds_rel)
    while acc_inds_rel:
        i = acc_inds_rel.pop(0)
        while i < no_points - 1:
            ax_possible_cur = calc_ax_poss(vx_start=vx_profile[i],
                                           radius=radii_mod[i],
                                           ggv=p_ggv[i],
                                           ax_max_machines=ax_max_machines,
                                           mu=mu_mod[i],
                                           mode=mode,
                                           dyn_model_exp=dyn_model_exp,
                                           drag_coeff=drag_coeff,
                                           m_veh=m_veh)
            vx_possible_next = math.sqrt(math.pow(vx_profile[i], 2) + 2 * ax_possible_cur * el_lengths_mod[i])
            if backwards:
                for j in range(1):
                    ax_possible_next = calc_ax_poss(vx_start=vx_possible_next,
                                                    radius=radii_mod[i + 1],
                                                    ggv=p_ggv[i + 1],
                                                    ax_max_machines=ax_max_machines,
                                                    mu=mu_mod[i + 1],
                                                    mode=mode,
                                                    dyn_model_exp=dyn_model_exp,
                                                    drag_coeff=drag_coeff,
                                                    m_veh=m_veh)
                    vx_tmp = math.sqrt(math.pow(vx_profile[i], 2) + 2 * ax_possible_next * el_lengths_mod[i])
                    if vx_tmp < vx_possible_next:
                        vx_possible_next = vx_tmp
                    else:
                        break
            if vx_possible_next < vx_profile[i + 1]:
                vx_profile[i + 1] = vx_possible_next
            i += 1
            if vx_possible_next > v_max or (acc_inds_rel and i >= acc_inds_rel[0]):
                break
    if backwards:
        vx_profile = np.flipud(vx_profile)
    return vx_profile

def __solver_fb_unclosed(p_ggv, ax_max_machines, v_max, radii, el_lengths, v_start, drag_coeff, m_veh, op_mode, mu=None, v_end=None, dyn_model_exp=1.0):
    if mu is None:
        mu = np.ones(radii.size)
        mu_mean = 1.0
    else:
        mu_mean = np.mean(mu)
    if op_mode == 'ggv':
        ay_max_global = mu_mean * np.amin(p_ggv[0, :, 2])
        vx_profile = np.sqrt(ay_max_global * radii)
        ay_max_curr = mu * np.interp(vx_profile, p_ggv[0, :, 0], p_ggv[0, :, 2])
        vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))
    else:
        vx_profile = np.sqrt(p_ggv[:, 0, 2] * radii)
    vx_profile[vx_profile > v_max] = v_max
    if vx_profile[0] > v_start:
        vx_profile[0] = v_start
    vx_profile = __solver_fb_acc_profile(p_ggv=p_ggv,
                                         ax_max_machines=ax_max_machines,
                                         v_max=v_max,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=False,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh)
    if v_end is not None and vx_profile[-1] > v_end:
        vx_profile[-1] = v_end
    vx_profile = __solver_fb_acc_profile(p_ggv=p_ggv,
                                         ax_max_machines=ax_max_machines,
                                         v_max=v_max,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=True,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh)
    return vx_profile

def __solver_fb_closed(p_ggv, ax_max_machines, v_max, radii, el_lengths, drag_coeff, m_veh, op_mode, mu=None, dyn_model_exp=1.0):
    no_points = radii.size
    if mu is None:
        mu = np.ones(no_points)
        mu_mean = 1.0
    else:
        mu_mean = np.mean(mu)
    if op_mode == 'ggv':
        ay_max_global = mu_mean * np.amin(p_ggv[0, :, 2])
        vx_profile = np.sqrt(ay_max_global * radii)
        converged = False
        for i in range(100):
            vx_profile_prev_iteration = vx_profile
            ay_max_curr = mu * np.interp(vx_profile, p_ggv[0, :, 0], p_ggv[0, :, 2])
            vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))
            if np.max(np.abs(vx_profile / vx_profile_prev_iteration - 1.0)) < 0.005:
                converged = True
                break
        if not converged:
            print("The initial vx profile did not converge after 100 iterations, please check radii and ggv!")
    else:
        vx_profile = np.sqrt(p_ggv[:, 0, 2] * radii)
    vx_profile[vx_profile > v_max] = v_max
    vx_profile_double = np.concatenate((vx_profile, vx_profile), axis=0)
    radii_double = np.concatenate((radii, radii), axis=0)
    el_lengths_double = np.concatenate((el_lengths, el_lengths), axis=0)
    mu_double = np.concatenate((mu, mu), axis=0)
    p_ggv_double = np.concatenate((p_ggv, p_ggv), axis=0)
    vx_profile_double = __solver_fb_acc_profile(p_ggv=p_ggv_double,
                                                ax_max_machines=ax_max_machines,
                                                v_max=v_max,
                                                radii=radii_double,
                                                el_lengths=el_lengths_double,
                                                mu=mu_double,
                                                vx_profile=vx_profile_double,
                                                backwards=False,
                                                dyn_model_exp=dyn_model_exp,
                                                drag_coeff=drag_coeff,
                                                m_veh=m_veh)
    vx_profile_double = np.concatenate((vx_profile_double[no_points:], vx_profile_double[no_points:]), axis=0)
    vx_profile_double = __solver_fb_acc_profile(p_ggv=p_ggv_double,
                                                ax_max_machines=ax_max_machines,
                                                v_max=v_max,
                                                radii=radii_double,
                                                el_lengths=el_lengths_double,
                                                mu=mu_double,
                                                vx_profile=vx_profile_double,
                                                backwards=True,
                                                dyn_model_exp=dyn_model_exp,
                                                drag_coeff=drag_coeff,
                                                m_veh=m_veh)
    vx_profile = vx_profile_double[no_points:]
    return vx_profile

def calc_vel_profile(ax_max_machines, kappa, el_lengths, closed, drag_coeff, m_veh, ggv=None, loc_gg=None, v_max=None, dyn_model_exp=1.0, mu=None, v_start=None, v_end=None, filt_window=None):
    if (ggv is not None or mu is not None) and loc_gg is not None:
        raise RuntimeError("Either ggv and optionally mu OR loc_gg must be supplied, not both (or all) of them!")
    if ggv is None and loc_gg is None:
        raise RuntimeError("Either ggv or loc_gg must be supplied!")
    if loc_gg is not None:
        if loc_gg.ndim != 2:
            raise RuntimeError("loc_gg must have two dimensions!")
        if loc_gg.shape[0] != kappa.size:
            raise RuntimeError("Length of loc_gg and kappa must be equal!")
        if loc_gg.shape[1] != 2:
            raise RuntimeError("loc_gg must consist of two columns: [ax_max, ay_max]!")
    if ggv is not None and ggv.shape[1] != 3:
        raise RuntimeError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")
    if mu is not None and kappa.size != mu.size:
        raise RuntimeError("kappa and mu must have the same length!")
    if closed and kappa.size != el_lengths.size:
        raise RuntimeError("kappa and el_lengths must have the same length if closed!")
    elif not closed and kappa.size != el_lengths.size + 1:
        raise RuntimeError("kappa must have the length of el_lengths + 1 if unclosed!")
    if not closed and v_start is None:
        raise RuntimeError("v_start must be provided for the unclosed case!")
    if v_start is not None and v_start < 0.0:
        v_start = 0.0
        print('WARNING: Input v_start was < 0.0. Using v_start = 0.0 instead!')
    if v_end is not None and v_end < 0.0:
        v_end = 0.0
        print('WARNING: Input v_end was < 0.0. Using v_end = 0.0 instead!')
    if not 1.0 <= dyn_model_exp <= 2.0:
        print('WARNING: Exponent for the vehicle dynamics model should be in the range [1.0, 2.0]!')
    if ax_max_machines.shape[1] != 2:
        raise RuntimeError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")
    if v_max is None:
        if ggv is None:
            raise RuntimeError("v_max must be supplied if ggv is None!")
        else:
            v_max = min(ggv[-1, 0], ax_max_machines[-1, 0])
    else:
        if ggv is not None and ggv[-1, 0] < v_max:
            raise RuntimeError("ggv has to cover the entire velocity range of the car (i.e. >= v_max)!")
        if ax_max_machines[-1, 0] < v_max:
            raise RuntimeError("ax_max_machines has to cover the entire velocity range of the car (i.e. >= v_max)!")
    if ggv is not None:
        p_ggv = np.repeat(np.expand_dims(ggv, axis=0), kappa.size, axis=0)
        op_mode = 'ggv'
    else:
        p_ggv = np.expand_dims(np.column_stack((np.ones(loc_gg.shape[0]) * 10.0, loc_gg)), axis=1)
        op_mode = 'loc_gg'
    radii = np.abs(np.divide(1.0, kappa, out=np.full(kappa.size, np.inf), where=kappa != 0.0))
    if not closed:
        vx_profile = __solver_fb_unclosed(p_ggv=p_ggv,
                                          ax_max_machines=ax_max_machines,
                                          v_max=v_max,
                                          radii=radii,
                                          el_lengths=el_lengths,
                                          v_start=v_start,
                                          drag_coeff=drag_coeff,
                                          m_veh=m_veh,
                                          op_mode=op_mode,
                                          mu=mu,
                                          v_end=v_end,
                                          dyn_model_exp=dyn_model_exp)
    else:
        vx_profile = __solver_fb_closed(p_ggv=p_ggv,
                                        ax_max_machines=ax_max_machines,
                                        v_max=v_max,
                                        radii=radii,
                                        el_lengths=el_lengths,
                                        drag_coeff=drag_coeff,
                                        m_veh=m_veh,
                                        op_mode=op_mode,
                                        mu=mu,
                                        dyn_model_exp=dyn_model_exp)
    if filt_window is not None:
        vx_profile = conv_filt(signal=vx_profile,
                               filt_window=filt_window,
                               closed=closed)
    return vx_profile

def calc_vel_profile_brake(kappa, el_lengths, v_start, drag_coeff, m_veh, ggv=None, loc_gg=None, dyn_model_exp=1.0, mu=None, decel_max=None):
    if decel_max is not None and not decel_max < 0.0:
        raise RuntimeError("Deceleration input must be negative!")
    if (ggv is not None or mu is not None) and loc_gg is not None:
        raise RuntimeError("Either ggv and optionally mu OR loc_gg must be supplied, not both (or all) of them!")
    if ggv is None and loc_gg is None:
        raise RuntimeError("Either ggv or loc_gg must be supplied!")
    if loc_gg is not None:
        if loc_gg.ndim != 2:
            raise RuntimeError("loc_gg must have two dimensions!")
        if loc_gg.shape[0] != kappa.size:
            raise RuntimeError("Length of loc_gg and kappa must be equal!")
        if loc_gg.shape[1] != 2:
            raise RuntimeError("loc_gg must consist of two columns: [ax_max, ay_max]!")
    if ggv is not None and ggv.shape[1] != 3:
        raise RuntimeError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")
    if mu is not None and kappa.size != mu.size:
        raise RuntimeError("kappa and mu must have the same length!")
    if kappa.size != el_lengths.size + 1:
        raise RuntimeError("kappa must have the length of el_lengths + 1!")
    if v_start < 0.0:
        v_start = 0.0
        print('WARNING: Input v_start was < 0.0. Using v_start = 0.0 instead!')
    if not 1.0 <= dyn_model_exp <= 2.0:
        print('WARNING: Exponent for the vehicle dynamics model should be in the range [1.0, 2.0]!')
    if ggv is not None and ggv[-1, 0] < v_start:
        raise RuntimeError("ggv has to cover the entire velocity range of the car (i.e. >= v_start)!")
    if ggv is not None:
        p_ggv = np.repeat(np.expand_dims(ggv, axis=0), kappa.size, axis=0)
    else:
        p_ggv = np.expand_dims(np.column_stack((np.ones(loc_gg.shape[0]) * 10.0, loc_gg)), axis=1)
    no_points = kappa.size
    vx_profile = np.zeros(no_points)
    vx_profile[0] = v_start
    radii = np.abs(np.divide(1, kappa, out=np.full(kappa.size, np.inf), where=kappa != 0))
    if mu is None:
        mu = np.ones(no_points)
    for i in range(no_points - 1):
        ggv_mod = np.copy(p_ggv[i])
        ggv_mod[:, 1] *= -1.0
        ax_final = calc_ax_poss(vx_start=vx_profile[i],
                                radius=radii[i],
                                ggv=ggv_mod,
                                mu=mu[i],
                                dyn_model_exp=dyn_model_exp,
                                drag_coeff=drag_coeff,
                                m_veh=m_veh,
                                ax_max_machines=None,
                                mode='decel_forw')
        ax_drag = -math.pow(vx_profile[i], 2) * drag_coeff / m_veh
        if decel_max is not None and ax_final < decel_max:
            if ax_drag < decel_max:
                ax_final = ax_drag
            else:
                ax_final = decel_max
        radicand = math.pow(vx_profile[i], 2) + 2 * ax_final * el_lengths[i]
        if radicand < 0.0:
            break
        else:
            vx_profile[i + 1] = math.sqrt(radicand)
    return vx_profile

def check_normals_crossing(track, normvec_normalized, horizon=10):
    no_points = track.shape[0]
    if horizon >= no_points:
        raise RuntimeError("Horizon of %i points is too large for a track with %i points, reduce horizon!"
                           % (horizon, no_points))
    elif horizon >= no_points / 2:
        print("WARNING: Horizon of %i points makes no sense for a track with %i points, reduce horizon!"
              % (horizon, no_points))
    les_mat = np.zeros((2, 2))
    idx_list = list(range(0, no_points))
    idx_list = idx_list[-horizon:] + idx_list + idx_list[:horizon]
    for idx in range(no_points):
        idx_neighbours = idx_list[idx:idx + 2 * horizon + 1]
        del idx_neighbours[horizon]
        idx_neighbours = np.array(idx_neighbours)
        is_collinear_b = np.isclose(np.cross(normvec_normalized[idx], normvec_normalized[idx_neighbours]), 0.0)
        idx_neighbours_rel = idx_neighbours[np.nonzero(np.invert(is_collinear_b))[0]]
        for idx_comp in list(idx_neighbours_rel):
            const = track[idx_comp, :2] - track[idx, :2]
            les_mat[:, 0] = normvec_normalized[idx]
            les_mat[:, 1] = -normvec_normalized[idx_comp]
            lambdas = np.linalg.solve(les_mat, const)
            if -track[idx, 3] <= lambdas[0] <= track[idx, 2] \
                    and -track[idx_comp, 3] <= lambdas[1] <= track[idx_comp, 2]:
                return True
    return False

def get_rel_path_part(path_cl, s_pos, s_dist_back=20.0, s_dist_forw=20.0, bound_right_cl=None, bound_left_cl=None):
    s_tot = path_cl[-1, 0]
    if s_dist_back + s_dist_forw >= s_tot:
        raise RuntimeError('Summed distance inputs are greater or equal to the total distance of the given path!')
    if bound_right_cl is not None and bound_right_cl.shape[0] != path_cl.shape[0]:
        raise RuntimeError('Inserted right boundary does not have the same number of points as the path!')
    if bound_left_cl is not None and bound_left_cl.shape[0] != path_cl.shape[0]:
        raise RuntimeError('Inserted left boundary does not have the same number of points as the path!')
    if s_pos >= s_tot:
        s_pos -= s_tot
    s_min = s_pos - s_dist_back
    s_max = s_pos + s_dist_forw
    if s_min < 0.0:
        s_min += s_tot
    if s_max > s_tot:
        s_max -= s_tot
    idx_start = np.searchsorted(path_cl[:, 0], s_min, side="right") - 1
    idx_stop = np.searchsorted(path_cl[:, 0], s_max, side="left") + 1
    if idx_start < idx_stop:
        path_rel = path_cl[idx_start:idx_stop]
        if bound_right_cl is not None:
            bound_right_rel = bound_right_cl[idx_start:idx_stop]
        else:
            bound_right_rel = None
        if bound_left_cl is not None:
            bound_left_rel = bound_left_cl[idx_start:idx_stop]
        else:
            bound_left_rel = None
    else:
        path_rel_part2 = np.copy(path_cl[:idx_stop])
        path_rel_part2[:, 0] += s_tot
        path_rel = np.vstack((path_cl[idx_start:-1], path_rel_part2))
        if bound_right_cl is not None:
            bound_right_rel = np.vstack((bound_right_cl[idx_start:-1], bound_right_cl[:idx_stop]))
        else:
            bound_right_rel = None
        if bound_left_cl is not None:
            bound_left_rel = np.vstack((bound_left_cl[idx_start:-1], bound_left_cl[:idx_stop]))
        else:
            bound_left_rel = None
    return path_rel, bound_right_rel, bound_left_rel

def import_veh_dyn_info(ggv_import_path=None, ax_max_machines_import_path=None):
    if ggv_import_path is not None:
        with open(ggv_import_path, "rb") as fh:
            ggv = np.loadtxt(fh, comments='#', delimiter=",")
        if ggv.ndim == 1:
            ggv = np.expand_dims(ggv, 0)
        if ggv.shape[1] != 3:
            raise RuntimeError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")
        invalid_1 = ggv[:, 0] < 0.0
        invalid_2 = ggv[:, 1:] > 50.0
        invalid_3 = ggv[:, 1] < 0.0
        invalid_4 = ggv[:, 2] < 0.0
        if np.any(invalid_1) or np.any(invalid_2) or np.any(invalid_3) or np.any(invalid_4):
            raise RuntimeError("ggv seems unreasonable!")
    else:
        ggv = None
    if ax_max_machines_import_path is not None:
        with open(ax_max_machines_import_path, "rb") as fh:
            ax_max_machines = np.loadtxt(fh, comments='#',  delimiter=",")
        if ax_max_machines.ndim == 1:
            ax_max_machines = np.expand_dims(ax_max_machines, 0)
        if ax_max_machines.shape[1] != 2:
            raise RuntimeError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")
        invalid_1 = ax_max_machines[:, 0] < 0.0
        invalid_2 = ax_max_machines[:, 1] > 20.0
        invalid_3 = ax_max_machines[:, 1] < 0.0
        if np.any(invalid_1) or np.any(invalid_2) or np.any(invalid_3):
            raise RuntimeError("ax_max_machines seems unreasonable!")
    else:
        ax_max_machines = None
    return ggv, ax_max_machines

def import_veh_dyn_info_2(filepath2localgg=""):
    if not filepath2localgg:
        raise RuntimeError('Missing path to file which contains vehicle acceleration limits!')
    with open(filepath2localgg, 'rb') as fh:
        data_localggfile = np.loadtxt(fh, comments='#', delimiter=',')
    if data_localggfile.ndim == 1:
        if data_localggfile.size != 5:
            raise RuntimeError('TPA MapInterface: wrong shape of localgg file data -> five columns required!')
        tpamap = np.hstack((np.zeros(3), data_localggfile[3:5]))[np.newaxis, :]
    elif data_localggfile.ndim == 2:
        if data_localggfile.shape[1] != 5:
            raise RuntimeError('TPA MapInterface: wrong shape of localgg file data -> five columns required!')
        tpamap = data_localggfile
        if np.any(tpamap[:, 0] < 0.0):
            raise RuntimeError('TPA MapInterface: one or more s-coordinate values are smaller than zero!')
        if np.any(np.diff(tpamap[:, 0]) <= 0.0):
            raise RuntimeError('TPA MapInterface: s-coordinates are not strictly monotone increasing!')
        if not np.isclose(np.hypot(tpamap[0, 1] - tpamap[-1, 1], tpamap[0, 2] - tpamap[-1, 2]), 0.0):
            raise RuntimeError('TPA MapInterface: s-coordinates representing the race track are not closed; '
                               'first and last point are not equal!')
    else:
        raise RuntimeError("Localgg file must provide one or two dimensions!")
    if np.any(tpamap[:, 3:] > 20.0):
        raise RuntimeError('TPA MapInterface: max. acceleration limit in localgg file exceeds 20 m/s^2!')
    if np.any(tpamap[:, 3:] < 1.0):
        raise RuntimeError('TPA MapInterface: min. acceleration limit in localgg file is below 1 m/s^2!')
    return tpamap

def interp_splines(coeffs_x, coeffs_y, spline_lengths=None, incl_last_point=False, stepsize_approx=None, stepnum_fixed=None):
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise RuntimeError("Coefficient matrices must have the same length!")
    if spline_lengths is not None and coeffs_x.shape[0] != spline_lengths.size:
        raise RuntimeError("coeffs_x/y and spline_lengths must have the same length!")
    if not (coeffs_x.ndim == 2 and coeffs_y.ndim == 2):
        raise RuntimeError("Coefficient matrices do not have two dimensions!")
    if (stepsize_approx is None and stepnum_fixed is None) \
            or (stepsize_approx is not None and stepnum_fixed is not None):
        raise RuntimeError("Provide one of 'stepsize_approx' and 'stepnum_fixed' and set the other to 'None'!")
    if stepnum_fixed is not None and len(stepnum_fixed) != coeffs_x.shape[0]:
        raise RuntimeError("The provided list 'stepnum_fixed' must hold an entry for every spline!")
    if stepsize_approx is not None:
        if spline_lengths is None:
            spline_lengths = calc_spline_lengths(coeffs_x=coeffs_x,
                                                 coeffs_y=coeffs_y,
                                                 quickndirty=False)
        dists_cum = np.cumsum(spline_lengths)
        no_interp_points = math.ceil(dists_cum[-1] / stepsize_approx) + 1
        dists_interp = np.linspace(0.0, dists_cum[-1], no_interp_points)
    else:
        no_interp_points = sum(stepnum_fixed) - (len(stepnum_fixed) - 1)
        dists_interp = None
    path_interp = np.zeros((no_interp_points, 2))
    spline_inds = np.zeros(no_interp_points, dtype=int)
    t_values = np.zeros(no_interp_points)
    if stepsize_approx is not None:
        for i in range(no_interp_points - 1):
            j = np.argmax(dists_interp[i] < dists_cum)
            spline_inds[i] = j
            if j > 0:
                t_values[i] = (dists_interp[i] - dists_cum[j - 1]) / spline_lengths[j]
            else:
                if spline_lengths.ndim == 0:
                    t_values[i] = dists_interp[i] / spline_lengths
                else:
                    t_values[i] = dists_interp[i] / spline_lengths[0]
            path_interp[i, 0] = coeffs_x[j, 0] \
                                + coeffs_x[j, 1] * t_values[i]\
                                + coeffs_x[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_x[j, 3] * math.pow(t_values[i], 3)
            path_interp[i, 1] = coeffs_y[j, 0]\
                                + coeffs_y[j, 1] * t_values[i]\
                                + coeffs_y[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_y[j, 3] * math.pow(t_values[i], 3)
    else:
        j = 0
        for i in range(len(stepnum_fixed)):
            if i < len(stepnum_fixed) - 1:
                t_values[j:(j + stepnum_fixed[i] - 1)] = np.linspace(0, 1, stepnum_fixed[i])[:-1]
                spline_inds[j:(j + stepnum_fixed[i] - 1)] = i
                j += stepnum_fixed[i] - 1
            else:
                t_values[j:(j + stepnum_fixed[i])] = np.linspace(0, 1, stepnum_fixed[i])
                spline_inds[j:(j + stepnum_fixed[i])] = i
                j += stepnum_fixed[i]
        t_set = np.column_stack((np.ones(no_interp_points), t_values, np.power(t_values, 2), np.power(t_values, 3)))
        n_samples = np.array(stepnum_fixed)
        n_samples[:-1] -= 1
        path_interp[:, 0] = np.sum(np.multiply(np.repeat(coeffs_x, n_samples, axis=0), t_set), axis=1)
        path_interp[:, 1] = np.sum(np.multiply(np.repeat(coeffs_y, n_samples, axis=0), t_set), axis=1)
    if incl_last_point:
        path_interp[-1, 0] = np.sum(coeffs_x[-1])
        path_interp[-1, 1] = np.sum(coeffs_y[-1])
        spline_inds[-1] = coeffs_x.shape[0] - 1
        t_values[-1] = 1.0
    else:
        path_interp = path_interp[:-1]
        spline_inds = spline_inds[:-1]
        t_values = t_values[:-1]
        if dists_interp is not None:
            dists_interp = dists_interp[:-1]
    return path_interp, spline_inds, t_values, dists_interp

def interp_track(track, stepsize):
    track_cl = np.vstack((track, track[0]))
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
    dists_cum_cl = np.cumsum(el_lengths_cl)
    dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)
    no_points_interp_cl = math.ceil(dists_cum_cl[-1] / stepsize) + 1
    dists_interp_cl = np.linspace(0.0, dists_cum_cl[-1], no_points_interp_cl)
    track_interp_cl = np.zeros((no_points_interp_cl, track_cl.shape[1]))
    track_interp_cl[:, 0] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 0])
    track_interp_cl[:, 1] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 1])
    track_interp_cl[:, 2] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 2])
    track_interp_cl[:, 3] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 3])
    if track_cl.shape[1] == 5:
        track_interp_cl[:, 4] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 4])
    return track_interp_cl[:-1]

def interp_track_widths(w_track, spline_inds, t_values, incl_last_point=False):
    w_track_cl = np.vstack((w_track, w_track[0]))
    no_interp_points = t_values.size
    if incl_last_point:
        w_track_interp = np.zeros((no_interp_points + 1, w_track.shape[1]))
        w_track_interp[-1] = w_track_cl[-1]
    else:
        w_track_interp = np.zeros((no_interp_points, w_track.shape[1]))
    for i in range(no_interp_points):
        ind_spl = spline_inds[i]
        w_track_interp[i, 0] = np.interp(t_values[i], (0.0, 1.0), w_track_cl[ind_spl:ind_spl + 2, 0])
        w_track_interp[i, 1] = np.interp(t_values[i], (0.0, 1.0), w_track_cl[ind_spl:ind_spl + 2, 1])
        if w_track.shape[1] == 3:
            w_track_interp[i, 2] = np.interp(t_values[i], (0.0, 1.0), w_track_cl[ind_spl:ind_spl + 2, 2])
    return w_track_interp

def create_raceline(refline, normvectors, alpha, stepsize_interp):
    raceline = refline + np.expand_dims(alpha, 1) * normvectors
    raceline_cl = np.vstack((raceline, raceline[0]))
    coeffs_x_raceline, coeffs_y_raceline, A_raceline, normvectors_raceline = calc_splines(path=raceline_cl,
                     use_dist_scaling=False)
    spline_lengths_raceline = calc_spline_lengths(coeffs_x=coeffs_x_raceline,
                            coeffs_y=coeffs_y_raceline)
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = \
        interp_splines(spline_lengths=spline_lengths_raceline,
                                      coeffs_x=coeffs_x_raceline,
                                      coeffs_y=coeffs_y_raceline,
                                      incl_last_point=False,
                                      stepsize_approx=stepsize_interp)
    s_tot_raceline = float(np.sum(spline_lengths_raceline))
    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])
    return raceline_interp, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, \
           t_values_raceline_interp, s_raceline_interp, spline_lengths_raceline, el_lengths_raceline_interp_cl

def nonreg_sampling(track, eps_kappa=1e-3, step_non_reg=0):
    if step_non_reg == 0:
        return track, np.arange(0, track.shape[0])
    path_cl = np.vstack((track[:, :2], track[0, :2]))
    coeffs_x, coeffs_y = calc_splines(path=path_cl)[:2]
    kappa_path = calc_head_curv_an(coeffs_x=coeffs_x,
                                                         coeffs_y=coeffs_y,
                                                         ind_spls=np.arange(0, coeffs_x.shape[0]),
                                                         t_spls=np.zeros(coeffs_x.shape[0]))[1]
    idx_latest = step_non_reg + 1
    sample_idxs = [0]
    for idx in range(1, len(kappa_path)):
        if np.abs(kappa_path[idx]) >= eps_kappa or idx >= idx_latest:
            sample_idxs.append(idx)
            idx_latest = idx + step_non_reg + 1
    return track[sample_idxs], np.array(sample_idxs)

def opt_min_curv(reftrack, normvectors, A, kappa_bound, w_veh, print_debug=False, plot_debug=False, closed=True, psi_s=None, psi_e=None, fix_s=False, fix_e=False):
    no_points = reftrack.shape[0]
    no_splines = no_points
    if not closed:
        no_splines -= 1
    if no_points != normvectors.shape[0]:
        raise RuntimeError("Array size of reftrack should be the same as normvectors!")
    if (no_points * 4 != A.shape[0] and closed) or (no_splines * 4 != A.shape[0] and not closed)\
            or A.shape[0] != A.shape[1]:
        raise RuntimeError("Spline equation system matrix A has wrong dimensions!")
    A_ex_b = np.zeros((no_points, no_splines * 4), dtype=int)
    for i in range(no_splines):
        A_ex_b[i, i * 4 + 1] = 1
    if not closed:
        A_ex_b[-1, -4:] = np.array([0, 1, 2, 3])
    A_ex_c = np.zeros((no_points, no_splines * 4), dtype=int)
    for i in range(no_splines):
        A_ex_c[i, i * 4 + 2] = 2
    if not closed:
        A_ex_c[-1, -4:] = np.array([0, 0, 2, 6])
    A_inv = np.linalg.inv(A)
    T_c = np.matmul(A_ex_c, A_inv)
    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))
    for i in range(no_splines):
        j = i * 4
        if i < no_points - 1:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, i + 1] = normvectors[i + 1, 0]
            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, i + 1] = normvectors[i + 1, 1]
        else:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, 0] = normvectors[0, 0]
            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, 0] = normvectors[0, 1]
    q_x = np.zeros((no_splines * 4, 1))
    q_y = np.zeros((no_splines * 4, 1))
    for i in range(no_splines):
        j = i * 4
        if i < no_points - 1:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[i + 1, 0]
            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[i + 1, 1]
        else:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[0, 0]
            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[0, 1]
    if not closed:
        q_x[-2, 0] = math.cos(psi_s + math.pi / 2)
        q_y[-2, 0] = math.sin(psi_s + math.pi / 2)
        q_x[-1, 0] = math.cos(psi_e + math.pi / 2)
        q_y[-1, 0] = math.sin(psi_e + math.pi / 2)
    x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x)
    y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)
    x_prime_sq = np.power(x_prime, 2)
    y_prime_sq = np.power(y_prime, 2)
    x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)
    curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)
    curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                          where=curv_den != 0)
    curv_part_sq = np.power(curv_part, 2)
    P_xx = np.matmul(curv_part_sq, y_prime_sq)
    P_yy = np.matmul(curv_part_sq, x_prime_sq)
    P_xy = np.matmul(curv_part_sq, x_prime_y_prime)
    T_nx = np.matmul(T_c, M_x)
    T_ny = np.matmul(T_c, M_y)
    H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
    H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
    H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
    H = H_x + H_xy + H_y
    H = (H + H.T) / 2
    f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
    f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
           + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
    f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
    f = f_x + f_xy + f_y
    f = np.squeeze(f)
    Q_x = np.matmul(curv_part, y_prime)
    Q_y = np.matmul(curv_part, x_prime)
    E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)
    k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))
    con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
    con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)
    con_stack = np.append(con_ge, con_le)
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2
    if not closed and fix_s:
        dev_max_left[0] = 0.05
        dev_max_right[0] = 0.05
    if not closed and fix_e:
        dev_max_left[-1] = 0.05
        dev_max_right[-1] = 0.05
    if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
        raise RuntimeError("Problem not solvable, track might be too small to run with current safety distance!")
    G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
    h = np.append(dev_max_right, dev_max_left)
    h = np.append(h, con_stack)
    alpha_mincurv = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]
    q_x_tmp = q_x + np.matmul(M_x, np.expand_dims(alpha_mincurv, 1))
    q_y_tmp = q_y + np.matmul(M_y, np.expand_dims(alpha_mincurv, 1))
    x_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x_tmp)
    y_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y_tmp)
    x_prime_prime = np.squeeze(np.matmul(T_c, q_x) + np.matmul(T_nx, np.expand_dims(alpha_mincurv, 1)))
    y_prime_prime = np.squeeze(np.matmul(T_c, q_y) + np.matmul(T_ny, np.expand_dims(alpha_mincurv, 1)))
    curv_orig_lin = np.zeros(no_points)
    curv_sol_lin = np.zeros(no_points)
    for i in range(no_points):
        curv_orig_lin[i] = (x_prime[i, i] * y_prime_prime[i] - y_prime[i, i] * x_prime_prime[i]) \
                          / math.pow(math.pow(x_prime[i, i], 2) + math.pow(y_prime[i, i], 2), 1.5)
        curv_sol_lin[i] = (x_prime_tmp[i, i] * y_prime_prime[i] - y_prime_tmp[i, i] * x_prime_prime[i]) \
                           / math.pow(math.pow(x_prime_tmp[i, i], 2) + math.pow(y_prime_tmp[i, i], 2), 1.5)
    curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))
    return alpha_mincurv, curv_error_max

def iqp_handler(reftrack, normvectors, A, spline_len, psi, kappa, dkappa, kappa_bound, w_veh, print_debug, plot_debug, stepsize_interp, iters_min=3, curv_error_allowed=0.01):
    reftrack_tmp = reftrack
    normvectors_tmp = normvectors
    A_tmp = A
    spline_len_tmp = spline_len
    psi_reftrack_tmp = psi
    kappa_reftrack_tmp = kappa
    dkappa_reftrack_tmp = dkappa
    iter_cur = 0
    while True:
        iter_cur += 1
        alpha_mincurv_tmp, curv_error_max_tmp = opt_min_curv(reftrack=reftrack_tmp,
                         normvectors=normvectors_tmp,
                         A=A_tmp,
                         kappa_bound=kappa_bound,
                         w_veh=w_veh,
                         print_debug=print_debug,
                         plot_debug=plot_debug)
        if iter_cur < iters_min:
            alpha_mincurv_tmp *= iter_cur * 1.0 / iters_min
        if iter_cur >= iters_min and curv_error_max_tmp <= curv_error_allowed:
            break
        refline_tmp, _, _, _, spline_inds_tmp, t_values_tmp = create_raceline(refline=reftrack_tmp[:, :2],
                            normvectors=normvectors_tmp,
                            alpha=alpha_mincurv_tmp,
                            stepsize_interp=stepsize_interp)[:6]
        reftrack_tmp[:, 2] -= alpha_mincurv_tmp
        reftrack_tmp[:, 3] += alpha_mincurv_tmp
        ws_track_tmp = interp_track_widths(w_track=reftrack_tmp[:, 2:],
                                                                   spline_inds=spline_inds_tmp,
                                                                   t_values=t_values_tmp,
                                                                   incl_last_point=False)
        reftrack_tmp = np.column_stack((refline_tmp, ws_track_tmp))
        refline_tmp_cl = np.vstack((reftrack_tmp[:, :2], reftrack_tmp[0, :2]))
        coeffs_x_tmp, coeffs_y_tmp, A_tmp, normvectors_tmp = calc_splines(path=refline_tmp_cl,
                         use_dist_scaling=False)
        spline_len_tmp = calc_spline_lengths(coeffs_x=coeffs_x_tmp, coeffs_y=coeffs_y_tmp)
        psi_reftrack_tmp, kappa_reftrack_tmp, dkappa_reftrack_tmp = calc_head_curv_an(
            coeffs_x=coeffs_x_tmp,
            coeffs_y=coeffs_y_tmp,
            ind_spls=np.arange(reftrack_tmp.shape[0]),
            t_spls=np.zeros(reftrack_tmp.shape[0]),
            calc_dcurv=True
        )
    return alpha_mincurv_tmp, reftrack_tmp, normvectors_tmp, spline_len_tmp, psi_reftrack_tmp, kappa_reftrack_tmp,\
           dkappa_reftrack_tmp

def opt_shortest_path(reftrack, normvectors, w_veh, print_debug=False):
    no_points = reftrack.shape[0]
    if no_points != normvectors.shape[0]:
        raise RuntimeError("Array size of reftrack should be the same as normvectors!")
    H = np.zeros((no_points, no_points))
    f = np.zeros(no_points)
    for i in range(no_points):
        if i < no_points - 1:
            H[i, i] += 2 * (math.pow(normvectors[i, 0], 2) + math.pow(normvectors[i, 1], 2))
            H[i, i + 1] = 0.5 * 2 * (-2 * normvectors[i, 0] * normvectors[i + 1, 0]
                                     - 2 * normvectors[i, 1] * normvectors[i + 1, 1])
            H[i + 1, i] = H[i, i + 1]
            H[i + 1, i + 1] = 2 * (math.pow(normvectors[i + 1, 0], 2) + math.pow(normvectors[i + 1, 1], 2))
            f[i] += 2 * normvectors[i, 0] * reftrack[i, 0] - 2 * normvectors[i, 0] * reftrack[i + 1, 0] \
                    + 2 * normvectors[i, 1] * reftrack[i, 1] - 2 * normvectors[i, 1] * reftrack[i + 1, 1]
            f[i + 1] = -2 * normvectors[i + 1, 0] * reftrack[i, 0] \
                       - 2 * normvectors[i + 1, 1] * reftrack[i, 1] \
                       + 2 * normvectors[i + 1, 0] * reftrack[i + 1, 0] \
                       + 2 * normvectors[i + 1, 1] * reftrack[i + 1, 1]
        else:
            H[i, i] += 2 * (math.pow(normvectors[i, 0], 2) + math.pow(normvectors[i, 1], 2))
            H[i, 0] = 0.5 * 2 * (-2 * normvectors[i, 0] * normvectors[0, 0] - 2 * normvectors[i, 1] * normvectors[0, 1])
            H[0, i] = H[i, 0]
            H[0, 0] += 2 * (math.pow(normvectors[0, 0], 2) + math.pow(normvectors[0, 1], 2))
            f[i] += 2 * normvectors[i, 0] * reftrack[i, 0] - 2 * normvectors[i, 0] * reftrack[0, 0] \
                    + 2 * normvectors[i, 1] * reftrack[i, 1] - 2 * normvectors[i, 1] * reftrack[0, 1]
            f[0] += -2 * normvectors[0, 0] * reftrack[i, 0] - 2 * normvectors[0, 1] * reftrack[i, 1] \
                    + 2 * normvectors[0, 0] * reftrack[0, 0] + 2 * normvectors[0, 1] * reftrack[0, 1]
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2
    dev_max_right[dev_max_right < 0.001] = 0.001
    dev_max_left[dev_max_left < 0.001] = 0.001
    G = np.vstack((np.eye(no_points), -np.eye(no_points)))
    h = np.ones(2 * no_points) * np.append(dev_max_right, dev_max_left)
    alpha_shpath = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]
    return alpha_shpath

def path_matching_local(path, ego_position, consider_as_closed=False, s_tot=None, no_interp_values=11):
    if consider_as_closed and s_tot is None:
        s_tot = path[-1, 0] + path[1, 0] - path[0, 0]
    dists_to_cg = np.hypot(path[:, 1] - ego_position[0], path[:, 2] - ego_position[1])
    ind_min = np.argpartition(dists_to_cg, 1)[0]
    if consider_as_closed:
        if ind_min == 0:
            ind_prev = dists_to_cg.shape[0] - 1
            ind_follow = 1
        elif ind_min == dists_to_cg.shape[0] - 1:
            ind_prev = ind_min - 1
            ind_follow = 0
        else:
            ind_prev = ind_min - 1
            ind_follow = ind_min + 1
    else:
        ind_prev = max(ind_min - 1, 0)
        ind_follow = min(ind_min + 1, dists_to_cg.shape[0] - 1)
    ang_prev = np.abs(angle3pt(path[ind_min, 1:3],
                                                                    ego_position,
                                                                    path[ind_prev, 1:3]))
    ang_follow = np.abs(angle3pt(path[ind_min, 1:3],
                                                                      ego_position,
                                                                      path[ind_follow, 1:3]))
    if ang_prev > ang_follow:
        a_pos = path[ind_prev, 1:3]
        b_pos = path[ind_min, 1:3]
        s_curs = np.append(path[ind_prev, 0], path[ind_min, 0])
    else:
        a_pos = path[ind_min, 1:3]
        b_pos = path[ind_follow, 1:3]
        s_curs = np.append(path[ind_min, 0], path[ind_follow, 0])
    if consider_as_closed:
        if ind_min == 0 and ang_prev > ang_follow:
            s_curs[1] = s_tot
        elif ind_min == dists_to_cg.shape[0] - 1 and ang_prev <= ang_follow:
            s_curs[1] = s_tot
    t_lin = np.linspace(0.0, 1.0, no_interp_values)
    x_cg_interp = np.linspace(a_pos[0], b_pos[0], no_interp_values)
    y_cg_interp = np.linspace(a_pos[1], b_pos[1], no_interp_values)
    dists_to_cg = np.hypot(x_cg_interp - ego_position[0], y_cg_interp - ego_position[1])
    ind_min_interp = np.argpartition(dists_to_cg, 1)[0]
    t_lin_used = t_lin[ind_min_interp]
    s_interp = np.interp(t_lin_used, (0.0, 1.0), s_curs)
    d_displ = dists_to_cg[ind_min_interp]
    return s_interp, d_displ

def path_matching_global(path_cl, ego_position, s_expected=None, s_range=20.0, no_interp_values=11):
    s_tot = path_cl[-1, 0]
    if s_expected is not None:
        path_rel = get_rel_path_part(path_cl=path_cl,
                                                                                   s_pos=s_expected,
                                                                                   s_dist_back=s_range,
                                                                                   s_dist_forw=s_range)[0]
        consider_as_closed = False
    else:
        path_rel = path_cl[:-1]
        consider_as_closed = True
    s_interp, d_displ = path_matching_local(path=path_rel,
                            ego_position=ego_position,
                            consider_as_closed=consider_as_closed,
                            s_tot=s_tot,
                            no_interp_values=no_interp_values)
    if s_interp >= s_tot:
        s_interp -= s_tot
    return s_interp, d_displ

def progressbar(i, i_total, prefix='', suffix='', decimals=1, length=50):
    pass

def side_of_line(a, b, z):
    side = np.sign((b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0]))
    return side

def dist_to_p(t_glob, path, p):
    s = interpolate.splev(t_glob, path)
    s = np.array(s).flatten()
    return spatial.distance.euclidean(p, s)

def spline_approximation(track, k_reg=3, s_reg=10, stepsize_prep=1.0, stepsize_reg=3.0, debug=False):
    track_interp = interp_track(track=track,
                                                 stepsize=stepsize_prep)
    track_interp_cl = np.vstack((track_interp, track_interp[0]))
    track_cl = np.vstack((track, track[0]))
    no_points_track_cl = track_cl.shape[0]
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
    dists_cum_cl = np.cumsum(el_lengths_cl)
    dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)
    tck_cl, t_glob_cl = interpolate.splprep([track_interp_cl[:, 0], track_interp_cl[:, 1]],
                                            k=k_reg,
                                            s=s_reg,
                                            per=1)[:2]
    no_points_lencalc_cl = math.ceil(dists_cum_cl[-1]) * 4
    path_smoothed_tmp = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_lencalc_cl), tck_cl)).T
    len_path_smoothed_tmp = np.sum(np.sqrt(np.sum(np.power(np.diff(path_smoothed_tmp, axis=0), 2), axis=1)))
    no_points_reg_cl = math.ceil(len_path_smoothed_tmp / stepsize_reg) + 1
    path_smoothed = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_reg_cl), tck_cl)).T[:-1]
    dists_cl = np.zeros(no_points_track_cl)
    closest_point_cl = np.zeros((no_points_track_cl, 2))
    closest_t_glob_cl = np.zeros(no_points_track_cl)
    t_glob_guess_cl = dists_cum_cl / dists_cum_cl[-1]
    for i in range(no_points_track_cl):
        closest_t_glob_cl[i] = optimize.fmin(dist_to_p,
                                             x0=t_glob_guess_cl[i],
                                             args=(tck_cl, track_cl[i, :2]),
                                             disp=False)
        closest_point_cl[i] = interpolate.splev(closest_t_glob_cl[i], tck_cl)
        dists_cl[i] = math.sqrt(math.pow(closest_point_cl[i, 0] - track_cl[i, 0], 2)
                                + math.pow(closest_point_cl[i, 1] - track_cl[i, 1], 2))
    sides = np.zeros(no_points_track_cl - 1)
    for i in range(no_points_track_cl - 1):
        sides[i] = side_of_line(a=track_cl[i, :2],
                                                 b=track_cl[i+1, :2],
                                                 z=closest_point_cl[i])
    sides_cl = np.hstack((sides, sides[0]))
    w_tr_right_new_cl = track_cl[:, 2] + sides_cl * dists_cl
    w_tr_left_new_cl = track_cl[:, 3] - sides_cl * dists_cl
    w_tr_right_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, w_tr_right_new_cl)
    w_tr_left_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, w_tr_left_new_cl)
    track_reg = np.column_stack((path_smoothed, w_tr_right_smoothed_cl[:-1], w_tr_left_smoothed_cl[:-1]))
    if track_cl.shape[1] == 5:
        banking_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, track_cl[:, 4])
        track_reg = np.column_stack((track_reg, banking_smoothed_cl[:-1]))
    return track_reg