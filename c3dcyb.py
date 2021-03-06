import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from utility.C3D import C3DServer
from utility.Geoms import Line, PointCloud
from utility.emg_proc import envelope
import json
import multiprocessing
from functools import partial
from scipy.signal import medfilt, hanning, find_peaks
import re

global sl_fr


def norm(a: np.ndarray, **kwargs):
    if type(a) is not np.ndarray or a.ndim is not 2:
        raise Exception('Incorrect array in user defined function norm')
    return a / np.linalg.norm(a, axis=1, **kwargs)[:, None] if a.shape[0] > a.shape[1] \
        else a / np.linalg.norm(a, axis=0, **kwargs)[None, :]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def proc_input(list_in):
    list_out = list(list_in)
    options_out = list()
    for el in list_in:
        if "-" in el:
            options_out.append(el)
            list_out.remove(el)
    return list_out, options_out


def calc_floating_angles(mat_a, mat_b, side='R', in_deg=True):
    if type(mat_a) is not np.ndarray: return False
    if type(mat_b) is not np.ndarray: return False
    a_x = mat_a[:, 0]
    a_y = mat_a[:, 1]
    a_z = mat_a[:, 2]
    b_x = mat_b[:, 0]
    b_y = mat_b[:, 1]
    b_z = mat_b[:, 2]
    axis = np.cross(b_z, a_x) / np.linalg.norm(np.cross(b_z, a_x))

    sin_alpha = -np.dot(axis, a_z)
    cos_alpha = np.dot(a_y, axis)
    cos_beta = np.dot(a_x, b_z)
    if side == 'R':
        sin_gamma = -np.dot(axis, b_x)
        cos_gamma = np.dot(b_y, axis)
    else:
        sin_gamma = np.dot(axis, b_x)
        cos_gamma = np.dot(b_y, axis)

    alpha = np.arctan2(sin_alpha, cos_alpha)
    beta = np.arccos(cos_beta)
    gamma = np.arctan(sin_gamma / cos_gamma)

    flexion = alpha
    external_rotation = gamma
    if side == 'R':
        adduction = beta - np.pi
    else:
        adduction = np.pi - beta

    angles = np.array([flexion, adduction, external_rotation], dtype=np.float32)
    if in_deg:
        return np.degrees(angles)
    else:
        return angles


def parallel_proc(fname, pdiff, ptarget_dir, pdir_in, pcf, penv, pz, pp):
    print('Processing file ' + fname + '...')
    joint_angles, emg_data, direc = c3d_proc(os.path.join(pdir_in, fname),
                                             diff=pdiff, emg_lowpass=pcf, env=penv, z_dir=pz, param=pp)
    name = os.path.splitext(fname)[0] if not pdiff else os.path.splitext(fname)[0] + '_w'

    list_vals = [ar.tolist() for ar in joint_angles.values()]
    keys = joint_angles.keys()
    save_angles = {key: li for key, li in zip(keys, list_vals)}
    save_angles["EMG"] = [emg.tolist() for emg in emg_data]
    save_angles["low-pass cf"] = pcf if penv else None
    save_angles["data mode"] = "velocity" if pdiff else "angles"
    prefix = ""
    if pp:
        prefix += "Param"
    if direc is not None:
        prefix += direc
    if direc is not None or pp:
        name = re.sub(r'(_)', r'\1' + prefix, name)
    if penv:
        name += "_Env"
    with open(os.path.join(ptarget_dir, name) + '.json', 'w') as fp:
        json.dump(save_angles, fp, indent=4)


def dir_proc(diff=False, env=False):
    dir_names, options = proc_input(sys.argv)
    for dir in dir_names[1:]:
        if not os.path.isdir(dir):
            print("No such directory found:")
            print(dir)
            help_msg()
            return
    if len(dir_names) >= 2:
        dir_in = dir_names[1]
        target_dir = dir_names[1]
        print('\nProcessing directory:')
        print(dir_in + '\n')
    else:
        print('Please enter directory path')
        help_msg()
        return
    if len(dir_names) >= 3:
        target_dir = dir_names[2]

    cf = [int(re.search(r'(\d+)$', str(arg)).group(0))
          for arg in options if "-cf" in arg or "--cutoff" in arg]
    z_dir = '-z' in options or '--zdir' in options
    param = '-p' in options or '-param' in options
    if not cf:
        cf = [5]
    f = partial(parallel_proc,
                pdiff=diff, pdir_in=dir_in, ptarget_dir=target_dir, pcf=cf[0], penv=env, pz=z_dir, pp=param)
    fnames = [f for f in os.listdir(dir_in) if f.endswith('.c3d')]
    nProcess = multiprocessing.cpu_count()
    import time
    t1 = time.perf_counter()
    with multiprocessing.Pool(nProcess) as pool:
        pool.map(f, fnames)
    print("Elapsed time: {}".format(time.perf_counter() - t1))
    return


def v_markers(dict_mkr_coords, asis_breadth=None):
    # region Create virtual markers
    dict_mkr_coords['LVAN'] = (dict_mkr_coords['LANK'] + dict_mkr_coords['LANM']) / 2
    dict_mkr_coords['RVAN'] = (dict_mkr_coords['RANK'] + dict_mkr_coords['RANM']) / 2
    dict_mkr_coords['LVKN'] = (dict_mkr_coords['LKNE'] + dict_mkr_coords['LKNM']) / 2
    dict_mkr_coords['RVKN'] = (dict_mkr_coords['RKNE'] + dict_mkr_coords['RKNM']) / 2

    # Hip joint center based on Vaughan, Davis and Connor: Dynamics of Human Gait
    v_hip = dict_mkr_coords['LASI'] - dict_mkr_coords['RASI']
    v_norm = np.linalg.norm(v_hip, axis=1)
    if asis_breadth is None:
        asis_breadth = np.mean(v_norm)
    v_hip = v_hip / v_norm[:, None]  # Broadcasting division for each row
    temp_hip = dict_mkr_coords['LASI'] - dict_mkr_coords['SACR']
    w_hip = np.cross(temp_hip, v_hip)
    w_hip = w_hip / np.linalg.norm(w_hip, axis=1)[:, None]
    u_hip = np.cross(v_hip, w_hip)
    dict_mkr_coords['LVHI'] = dict_mkr_coords['SACR'] + asis_breadth * (0.598 * u_hip + 0.344 * v_hip - 0.29 * w_hip)
    dict_mkr_coords['RVHI'] = dict_mkr_coords['SACR'] + asis_breadth * (0.598 * u_hip - 0.344 * v_hip - 0.29 * w_hip)
    # endregion


def jcs_angle_est(dict_mkr_coords, asis_breadth=None):
    v_markers(dict_mkr_coords, asis_breadth=asis_breadth)
    
    return


def default_feet_sys(reference_sys: dict, dict_mkr: dict):
    # region Left Foot
    k_vect = norm(dict_mkr['LHEE']-dict_mkr['LTOE'])
    i_vect = norm(np.cross(dict_mkr['LVAN'] - dict_mkr['LHEE'], k_vect, axis=1))
    j_vect = np.cross(k_vect, i_vect, axis=1)
    reference_sys['LFoot'] = np.dstack((i_vect, j_vect, k_vect))
    # endregion

    # region Right Foot
    # endregion
    return


def coord_sys_angle(reference_sys: dict, diff=False, segment_names=('Thigh', 'Shank', 'Foot'),
                    ang_method=calc_floating_angles):
    cnt_frames = reference_sys['Pelvis'].shape[0]
    joint_names = ['Hip', 'Knee', 'Ankle'] if not diff else ['HipW', 'KneeW', 'AnkleW']
    joint_angles = {}
    for cur_side in ['L', 'R']:
        # Hip joint
        joint_angles[cur_side + joint_names[0]] = np.zeros((cnt_frames, 3), dtype=np.float32)
        for i in range(cnt_frames):
            joint_angles[cur_side + joint_names[0]][i] = ang_method(reference_sys['Pelvis'][i],
                                                                    reference_sys[cur_side + 'Thigh'][i],
                                                                    in_deg=False)
        # Rest of the joints
        for seg in range(len(segment_names) - 1):
            joint_angles[cur_side + joint_names[seg + 1]] = np.zeros((cnt_frames, 3), dtype=np.float32)
            for i in range(cnt_frames):
                joint_angles[cur_side + joint_names[seg + 1]][i] = \
                    ang_method(reference_sys[cur_side + segment_names[seg]][i],
                               reference_sys[cur_side + segment_names[seg + 1]][i], side=cur_side,
                               in_deg=False)

    for name in joint_angles.keys():
        for ang in range(3):
            joint_angles[name][:, ang] = np.unwrap(joint_angles[name][:, ang])
            if diff:
                joint_angles[name][:, ang] = np.gradient(joint_angles[name][:, ang])

    # endregion
    return joint_angles


def angle_est(dict_mkr_coords, asis_breadth=None):
    v_markers(dict_mkr_coords, asis_breadth=asis_breadth)

    # region Calculation of local coordinate systems
    # Access frame from rs = reference_sys['Name'] using rs[i]
    # Access all frames of a given vector using rs[:, :, n]
    reference_sys = {}
    segment_names = ['Thigh', 'Shank', 'Foot']
    ref_markers = [['VHI', 'VKN', 'THI'], ['VKN', 'VAN', 'TIB'], ['HEE', 'TOE', 'VAN']]
    for side in ['L', 'R']:
        for seg_enum, cur_seg in enumerate(segment_names):
            k_vect = norm((dict_mkr_coords[side + ref_markers[seg_enum][0]] - dict_mkr_coords[
                side + ref_markers[seg_enum][1]]))
            j_vect = (-1 if side == 'L' else 1) * norm(np.cross(k_vect,
                              dict_mkr_coords[side + ref_markers[seg_enum][2]] -
                              dict_mkr_coords[side + ref_markers[seg_enum][0]], axis=1))
            i_vect = np.cross(j_vect, k_vect)
            reference_sys[side + cur_seg] = np.dstack((i_vect, j_vect, k_vect))

    # Pelvis local coordinate system
    i_vect = (dict_mkr_coords['RASI'] - dict_mkr_coords['LASI'])
    i_vect = i_vect / np.linalg.norm(i_vect, axis=1)[:, None]
    k_vect = np.cross(dict_mkr_coords['RASI'] - dict_mkr_coords['SACR'],
                      dict_mkr_coords['LASI'] - dict_mkr_coords['SACR'], axis=1)
    k_vect = k_vect / np.linalg.norm(k_vect, axis=1)[:, None]
    j_vect = np.cross(k_vect, i_vect)
    reference_sys['Pelvis'] = np.dstack((i_vect, j_vect, k_vect))
    # endregion

    from jcs import Segment, JCS
    pelv = Segment(reference_sys['Pelvis'][:,:,0], reference_sys['Pelvis'][:,:,1], reference_sys['Pelvis'][:,:,2])
    lth = Segment(reference_sys['LThigh'][:,:,0], reference_sys['LThigh'][:,:,1], reference_sys['LThigh'][:,:,2])
    lhip = JCS(pelv, lth, side='L')

    return coord_sys_angle(reference_sys=reference_sys), dict_mkr_coords


def param_est(dict_mkr_coords, asis_breadth=None):
    v_markers(dict_mkr_coords, asis_breadth=asis_breadth)

    i_vect = (dict_mkr_coords['RASI'] - dict_mkr_coords['LASI'])
    i_vect = i_vect / np.linalg.norm(i_vect, axis=1)[:, None]
    k_vect = np.cross(dict_mkr_coords['RASI'] - dict_mkr_coords['SACR'],
                      dict_mkr_coords['LASI'] - dict_mkr_coords['SACR'], axis=1)
    k_vect = k_vect / np.linalg.norm(k_vect, axis=1)[:, None]
    j_vect = np.cross(k_vect, i_vect)

    med = partial(medfilt, kernel_size=51)

    def sm_med(a):
        return med(np.convolve(a, hanning(11), 'same'))

    la_speed = np.linalg.norm(np.apply_along_axis(sm_med, 0, np.gradient(dict_mkr_coords['LHEE'], axis=0)), axis=1)
    ra_speed = np.linalg.norm(np.apply_along_axis(sm_med, 0, np.gradient(dict_mkr_coords['RHEE'], axis=0)), axis=1)

    d_speed = ra_speed - la_speed

    direc = j_vect
    direc[:, 2] = 0
    dist = dict_mkr_coords['RHEE'] - dict_mkr_coords['LHEE']
    ap_dist = np.empty(dist.shape[0])
    sacr_speed = np.apply_along_axis(med, 0, np.gradient(dict_mkr_coords['SACR'], axis=0))
    for i in range(dist.shape[0]):
        ap_dist[i] = np.dot(dist[i, :], direc[i, :])
        sacr_speed[i] = np.dot(sacr_speed[i, :], direc[i, :])

    height_dist = dict_mkr_coords['RHEE'][:, 2] - dict_mkr_coords['LHEE'][:, 2]

    step_class = np.ones(dist.shape[0])
    step_class[d_speed > 0.001] = 0
    step_class[d_speed < -0.001] = 2
    mode_w = 51
    step_window = rolling_window(step_class, mode_w)

    def mode_filter(a, classes=(0, 1, 2)):
        scores = []
        if classes is None:
            classes = np.unique(a)
        for c in classes:
            scores.append(np.sum(a == c))
        return classes[int(np.argmax(scores))]

    step_class = np.apply_along_axis(mode_filter, 1, step_window)
    step_class = np.append(step_class, np.ones(int((mode_w - 1) / 2)) * step_class[-1])
    step_class = np.append(np.ones(int((mode_w - 1) / 2)) * step_class[0], step_class)

    step_durations = []
    step_starts = []
    rs = 1  # running sum
    cc = step_class[0]  # current class
    step_starts.append(0)
    for i in range(1, dist.shape[0]):
        if step_class[i] == cc:
            rs += 1
        else:
            step_starts.append(i)
            step_durations.append(rs)
            rs = 1
            cc = step_class[i]
    step_durations.append(rs)

    def get_span(a, dist):
        t0 = step_starts[int(np.sum(a > np.array(step_starts)) - 1)]
        tend = t0 + step_durations[int(np.sum(a > np.array(step_starts)) - 1)]
        if int(np.sum(a > np.array(step_starts)) - 1) == 0:
            return max(np.max(dist[t0:tend]), 0) - min(np.min(dist[t0:tend]), 0)
        if int(np.sum(a > np.array(step_starts)) - 1) == len(step_starts) - 1:
            return max(np.max(dist[t0:tend]), 0) - min(np.min(dist[t0:tend]), 0)
        return np.max(dist[t0:tend]) - np.min(dist[t0:tend])

    v_get_stride_length = np.vectorize(partial(get_span, dist=ap_dist))
    v_get_step_height = np.vectorize(partial(get_span, dist=height_dist))
    stride_lengths = np.zeros(dist.shape[0])
    step_heights = np.zeros(dist.shape[0])

    for i, step_start in enumerate(step_starts):
        if step_class[step_start] == 1:
            continue
        stride_lengths[step_start:step_start + step_durations[i]] = \
            v_get_stride_length(np.arange(step_start, step_start + step_durations[i]))
        stride_lengths[step_start:step_start + step_durations[i]] = \
            mode_filter(stride_lengths[step_start:step_start + step_durations[i]], classes=None)
        step_heights[step_start:step_start + step_durations[i]] = \
            v_get_step_height(np.arange(step_start, step_start + step_durations[i]))
        step_heights[step_start:step_start + step_durations[i]] = \
            mode_filter(step_heights[step_start:step_start + step_durations[i]], classes=None)

    def get_speed(a):
        return stride_lengths[a] / step_durations[int(np.sum(a >= np.array(step_starts)) - 1)]

    step_speed = np.vectorize(get_speed)(np.arange(stride_lengths.shape[0]))

    dict_out = {
        'step_class': step_class,
        'step_heights': step_heights,
        'stride_lengths': stride_lengths,
        'la_speed': la_speed,
        'ra_speed': ra_speed,
        'sacr_speed': sacr_speed,
        'step_speed': step_speed
    }
    return dict_out


def c3d_proc(c3d_name, asis_breadth=None, emg_lowpass=10, diff=False, env=False, z_dir=False, param=False):
    with C3DServer() as c3d:
        # region Read a C3D file and extract all necessary info
        c3d.open_c3d(c3d_name)
        mkr_names = ['SACR', 'RASI', 'LASI', 'LTHI', 'LKNE', 'LKNM', 'RTHI', 'RKNE', 'RKNM',
                     'LTIB', 'LANK', 'LANM', 'RTIB', 'RANK', 'RANM', 'LHEE', 'LTOE', 'RHEE', 'RTOE']
        dict_mkr_idx = {k: c3d.get_marker_index(k) for k in mkr_names}
        mkr_scale_factor = 0.001 if c3d.get_marker_unit() == 'mm' else 1.0
        mkr_pts = c3d.get_all_marker_coords_arr3d(mkr_scale_factor)
        dict_mkr_coords = {k: mkr_pts[:, dict_mkr_idx[k], :] for k in mkr_names}
        emg_data = [None] * 8
        emg_freq = c3d.get_analog_frame_rate()
        for i in range(8):
            emg_data[i] = c3d.get_analog_data("Sensor {}.EMG{}".format(i + 1, i + 1))
            if env:
                emg_data[i] = envelope(emg_data[i], sfreq=emg_freq, low_pass=emg_lowpass)
        c3d.close_c3d()
        # endregion

        # region crop out regions with missing
        bool_idx = np.all(np.any(list(dict_mkr_coords.values()), axis=2), axis=0)
        for key in dict_mkr_coords.keys():
            dict_mkr_coords[key] = dict_mkr_coords[key][bool_idx, :]
        emg_data = [np.array([emg[i] for i in np.argwhere(np.repeat(bool_idx, 20)).flatten().tolist()])
                    for emg in emg_data]
        # endregion

        if param:
            joint_angles = param_est(dict_mkr_coords, asis_breadth)
        else:
            joint_angles, _ = angle_est(dict_mkr_coords, asis_breadth, diff=diff)

        direction = None
        if z_dir:
            if dict_mkr_coords['SACR'][-1, 2] > dict_mkr_coords['SACR'][0, 2] * 1.1:
                direction = 'Up'
            elif dict_mkr_coords['SACR'][-1, 2] < dict_mkr_coords['SACR'][0, 2] * 0.9:
                direction = 'Down'

    return joint_angles, emg_data, direction


def visu(diff=False, env=False):
    # region Globals and Commandline arguements
    global sl_fr
    dir_names, _ = proc_input(sys.argv)

    c3d_f_path = dir_names[1]
    print(c3d_f_path)
    # endregion

    # region Read a C3D file and extract all necessary info
    c3d = C3DServer()
    c3d.open_c3d(c3d_f_path)
    start_frame = c3d.get_start_frame_number()
    end_frame = c3d.get_end_frame_number()
    cnt_frames = c3d.get_total_frame_counts()
    arr_frames = np.linspace(start_frame, end_frame, cnt_frames, dtype=int)
    mkr_names = ['SACR', 'RASI', 'LASI', 'LTHI', 'LKNE', 'LKNM', 'RTHI', 'RKNE', 'RKNM',
                 'LTIB', 'LANK', 'LANM', 'RTIB', 'RANK', 'RANM', 'LHEE', 'LTOE', 'RHEE', 'RTOE']
    dict_mkr_idx = {k: c3d.get_marker_index(k) for k in mkr_names}
    mkr_scale_factor = 0.001 if c3d.get_marker_unit() == 'mm' else 1.0
    mkr_pts = c3d.get_all_marker_coords_arr3d(mkr_scale_factor)
    dim_mkr_pts = mkr_pts.shape
    mkr_pts_reshape = np.reshape(mkr_pts, (dim_mkr_pts[0] * dim_mkr_pts[1], dim_mkr_pts[2]))
    dict_mkr_coords = {k: mkr_pts[:, dict_mkr_idx[k], :] for k in mkr_names}
    emg_data = [None] * 8
    cf = [int(re.search(r'(\d+)$', str(arg)).group(0))
          for arg in sys.argv if "-cf" in arg or "--cutoff" in arg]
    emg_freq = int(c3d.get_analog_frame_rate())
    if not cf:
        cf = [5]
    for i in range(8):
        emg_data[i] = c3d.get_analog_data("Sensor {}.EMG{}".format(i + 1, i + 1))
        if env:
            emg_data[i] = envelope(emg_data[i], low_pass=cf[0], sfreq=emg_freq)
    c3d.close_c3d()
    # endregion

    # region crop out regions with missing
    bool_idx = np.all(np.any(list(dict_mkr_coords.values()), axis=2), axis=0)
    for key in dict_mkr_coords.keys():
        dict_mkr_coords[key] = dict_mkr_coords[key][bool_idx, :]
    mkr_pts = mkr_pts[bool_idx, :, :]
    emg_data = [np.array([emg[i] for i in np.argwhere(np.repeat(bool_idx, 20)).flatten().tolist()]) for emg in emg_data]
    arr_frames = arr_frames[bool_idx]
    start_frame = arr_frames[0]
    end_frame = arr_frames[-1]
    # endregion

    joint_angles, dict_mkr_coords = angle_est(dict_mkr_coords, None)
    new_mkr_pts = np.stack((dict_mkr_coords['LVHI'], dict_mkr_coords['LVKN'], dict_mkr_coords['LVAN'],
                            dict_mkr_coords['RVHI'], dict_mkr_coords['RVKN'], dict_mkr_coords['RVAN']))
    new_mkr_pts = np.rollaxis(new_mkr_pts, 1)
    mkr_pts = np.concatenate((mkr_pts, new_mkr_pts), axis=1)
    # region Visualisation axes initialisation
    # Create a figure using matplotlib
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(4, 5, height_ratios=[6, 6, 6, 1])
    # Add an axes for 3D graph for markers
    from matplotlib.ticker import MultipleLocator
    ax = fig.add_subplot(gs[:3, :3], projection='3d', proj_type='persp')
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel('X (m)', fontsize=16)
    ax.set_ylabel('Y (m)', fontsize=16)
    ax.set_zlabel('Z (m)', fontsize=16)
    ax.view_init(elev=20, azim=-135)
    title = ax.set_title('C3D viewer, frame={0:05d}'.format(start_frame))

    # Calculate the proper range of x y z limits
    pts_min = np.min(mkr_pts_reshape, axis=0)
    pts_max = np.max(mkr_pts_reshape, axis=0)
    pts_range = pts_max - pts_min
    pts_mid = (pts_min + pts_max) * 0.5
    pts_range_max = np.max(pts_range)
    ax.set_xlim(pts_mid[0] - pts_range_max * 0.5, pts_mid[0] + pts_range_max * 0.5)
    ax.set_ylim(pts_mid[1] - pts_range_max * 0.5, pts_mid[1] + pts_range_max * 0.5)
    ax.set_zlim(pts_mid[2] - pts_range_max * 0.5, pts_mid[2] + pts_range_max * 0.5)

    # A list to contain geometric class instances
    geom_objs = []

    # Create a PointCloud class instance in order to display the markers and virtual markers
    pts_obj = PointCloud('[MARKERS]', ax)
    pts_obj.set_point_names(mkr_names)
    pts_obj.set_from_points(mkr_pts[:, :len(mkr_names), :])
    pts_obj.set_marker_style(3, 'tab:blue', 'tab:blue')
    pts_obj.draw_vis_objs(fr=0)
    geom_objs.append(pts_obj)

    virtual_names = ['LVHI', 'LVKN', 'LVAN', 'RVHI', 'RVKN', 'RVAN']
    vpts_obj = PointCloud('[VMARKERS]', ax)
    vpts_obj.set_point_names(virtual_names)
    vpts_obj.set_from_points(mkr_pts[:, -len(virtual_names):, :])
    vpts_obj.set_marker_style(8, 'tab:orange', 'none')
    vpts_obj.draw_vis_objs(fr=0)
    geom_objs.append(vpts_obj)

    # region Create several Line class instances in order to display the lines in the leg
    line_obj = Line(name='RThigh', axes=ax)
    line_obj.set_point_names(['RVHI', 'RVKN'])
    line_obj.set_from_2points(dict_mkr_coords['RVHI'], dict_mkr_coords['RVKN'])
    line_obj.set_color((0.05, 0.5, 0.1, 0.5))
    line_obj.draw_vis_objs(fr=0)
    geom_objs.append(line_obj)

    line_obj = Line(name='RShank', axes=ax)
    line_obj.set_point_names(['RVKN', 'RVAN'])
    line_obj.set_from_2points(dict_mkr_coords['RVKN'], dict_mkr_coords['RVAN'])
    line_obj.set_color((0.05, 0.5, 0.1, 0.5))
    line_obj.draw_vis_objs(fr=0)
    geom_objs.append(line_obj)

    line_obj = Line(name='LThigh', axes=ax)
    line_obj.set_point_names(['LVHI', 'LVKN'])
    line_obj.set_from_2points(dict_mkr_coords['LVHI'], dict_mkr_coords['LVKN'])
    line_obj.set_color((0.5, 0.05, 0.05, 0.5))
    line_obj.draw_vis_objs(fr=0)
    geom_objs.append(line_obj)

    line_obj = Line(name='LShank', axes=ax)
    line_obj.set_point_names(['LVKN', 'LVAN'])
    line_obj.set_from_2points(dict_mkr_coords['LVKN'], dict_mkr_coords['LVAN'])
    line_obj.set_color((0.5, 0.05, 0.05, 0.5))
    line_obj.draw_vis_objs(fr=0)
    geom_objs.append(line_obj)

    line_obj = Line(name='LFoot', axes=ax)
    line_obj.set_point_names(['LVAN', 'LTOE'])
    line_obj.set_from_2points(dict_mkr_coords['LVAN'], dict_mkr_coords['LTOE'])
    line_obj.set_color((0.5, 0.05, 0.05, 0.5))
    line_obj.draw_vis_objs(fr=0)
    geom_objs.append(line_obj)

    line_obj = Line(name='RFoot', axes=ax)
    line_obj.set_point_names(['RVAN', 'RTOE'])
    line_obj.set_from_2points(dict_mkr_coords['RVAN'], dict_mkr_coords['RTOE'])
    line_obj.set_color((0.05, 0.5, 0.1, 0.5))
    line_obj.draw_vis_objs(fr=0)
    geom_objs.append(line_obj)
    # endregion
    # endregion
    import seaborn as sns
    sns.set_style("darkgrid")
    sns.set_context('paper')
    # region Plotting angles
    joint_names = ['Hip', 'Knee', 'Ankle'] if not diff else ['HipW', 'KneeW', 'AnkleW']
    vlines = []
    axes_list = list()
    for row in range(3):
        for col, side in enumerate(['L', 'R']):
            cur_ax = fig.add_subplot(gs[row, 3 + col])
            cur_ax.set_xlim(start_frame / 100, end_frame / 100)
            # cur_ax.set_ylim(-2, 2)
            vlines.append(
                cur_ax.axvline(x=start_frame / 100, ymin=0, ymax=1, color=(0, 1, 0), linewidth=2.0, linestyle='--'))
            # cur_ax.set_title(side + joint_names[row])
            cur_ax.plot(arr_frames / 100, joint_angles[side + joint_names[row]][:, 0] / np.pi * 180,
                        color='tab:blue', lw=2, label=side + joint_names[row] + ' Flexion-Extension')
            cur_ax.set_title(side + joint_names[row] + ' Flexion-Extension', fontsize=14)
            cur_ax.tick_params(axis='both', which='major', labelsize=12)
            if row == 2:
                cur_ax.set_xlabel("Time (s)", fontsize=14)
            else:
                cur_ax.set_xticklabels([])
            if col == 0:
                cur_ax.set_ylabel("Degrees", verticalalignment='top', labelpad=20, fontsize=14)
            else:
                cur_ax.set_yticklabels([])
            axes_list.append(cur_ax)
    fig.align_ylabels()
    for i, cur_ax in enumerate(axes_list):
        cur_ax.get_shared_x_axes().join(cur_ax, axes_list[i % 2])
        cur_ax.get_shared_y_axes().join(cur_ax, axes_list[int(i / 2) * 2])
    axes_list[0].set_ylim((-10 + np.min(joint_angles['RHip'][:, 0] / np.pi * 180),
                           10 + np.max(joint_angles['LHip'][:, 0] / np.pi * 180)))
    axes_list[2].set_ylim((-10 + np.min(joint_angles['RKnee'][:, 0] / np.pi * 180),
                           10 + np.max(joint_angles['RKnee'][:, 0] / np.pi * 180)))
    axes_list[4].set_ylim((-10 + np.min(joint_angles['RAnkle'][:, 0] / np.pi * 180),
                           10 + np.max(joint_angles['RAnkle'][:, 0] / np.pi * 180)))

    ax_fr = fig.add_subplot(gs[3, :], facecolor='lightgoldenrodyellow')
    sl_fr = Slider(ax_fr, 'Frame', start_frame, end_frame, valinit=start_frame, valstep=1, valfmt='%05d')

    def update(val):
        fr_no = int(sl_fr.val)
        fr_idx = fr_no - start_frame
        title.set_text('C3D viewer, frame={0:05d}'.format(fr_no))
        for geom in geom_objs:
            geom.update_vis_objs(fr_idx)
        for vl in vlines:
            vl.set_xdata(fr_no / 100)
        fig.canvas.draw()
        fig.canvas.flush_events()

    sl_fr.on_changed(update)
    import seaborn
    with seaborn.axes_style("dark"):
        seaborn.set_context('poster')
        sns.set_style("dark", {"axes.facecolor": ".95"})
        f, axes = plt.subplots(8, 1, sharex='col', figsize=(12, 8))

        axes[7].get_shared_y_axes().remove(axes[7])
        if env:
            axes[0].set_title("EMG envelopes, low-pass = {0}".format(cf[0]))
        else:
            axes[0].set_title("Subject 4 Walking EMG signals, $F_s={}$".format(emg_freq))
        emg_names = ["LIO", "RIO", "LEO", "REO", "LD", "MT", "ES", "ECG"]
        x = np.arange(len(emg_data[0])) / emg_freq

        def sc(axis):
            l = axis.get_majorticklocs()
            return len(l) > 1 and (l[1] - l[0])

        for row, ax in enumerate(axes):
            ax.plot(x, emg_data[row], linewidth=0.8, c=seaborn.color_palette()[0])
            d = sc(ax.yaxis)
            ax.set_ylabel(emg_names[row], fontsize=14)
            if row is not 7:
                seaborn.despine(ax=ax, bottom=True)
        ax.spines["top"].set_linewidth(2)
        f.subplots_adjust(hspace=-0.01)

        from utility.scalebars import add_scalebar
        add_scalebar(ax, sizex=0, matchx=False, sizey=1 * d, matchy=False, hidex=False, hidey=False,
                     labely='\n{0:.3f} mV'.format(d * 1000), borderpad=0.2)
        ax = axes[0]
        d = sc(ax.yaxis)
        add_scalebar(ax, sizex=0, matchx=False, sizey=d, matchy=False, hidex=False, hidey=False,
                     labely='\n{0:.3f} mV'.format(d * 1000), borderpad=0.2)
        for ax in axes:
            ax.get_yaxis().set_ticks([])
        ax.set_xlabel("Time (s)")
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig("emg.png", transparent=True)
        # f, axes = plt.subplots(8, 1, sharex='col', sharey='col')
        #
        # axes[0].set_title("Normalised EMG input frame to NN".format(cf[0]))
        # emg_names = ["LIO", "RIO", "LEO", "REO", "LD", "MT", "ES", "ECG"]
        # x = np.arange(20)
        #
        # for i, emg in enumerate(emg_data):
        #     emg = np.array(emg_data[i])
        #     emg_data[i] = (emg - np.mean(emg)) / np.std(emg)
        # emg_data = np.array(emg_data)[:, 350:370]
        # vmin = np.min(emg_data)
        # vmax = np.max(emg_data)
        # cbar_ax = f.add_axes([.91, .11, .03, .77])
        # for row, ax in enumerate(axes):
        #     emg = np.array(emg_data[row])
        #     emg = (emg - np.mean(emg)) / np.std(emg)
        #     a = np.expand_dims(emg, axis=0)
        #     hm = seaborn.heatmap(a, ax=ax, vmin=vmin, vmax=vmax, cbar=row == 0, cbar_ax=None if row else cbar_ax)
        #     ax.set_ylabel(emg_names[row])
        #     ax.get_yaxis().set_ticks([])
        # # f.tight_layout(rect=[0, 0, .9, 1])
        # cbar_ax.margins(x=0.1)
        # plt.savefig("20.png", transparent=True)
    plt.show()

    return sl_fr
    # endregion


def help_msg():
    print('Joint Angle Estimation using Joint-Coordinate Systems and EMG pre-processing\n'
          'Usage:\n'
          'python c3dcyb.py [options] source_path [target_path]\n'
          '\t[options]:\n'
          '\t\tnone\t\tInput source_path to directory of .c3d files to process\n'
          '\t\t-d --diff\tExtract joint angular velocity instead of angles\n'
          '\t\t-cfN --cutoffN\tSubstitute N with cutoff frequency (Hz) for EMG low-pass\n'
          '\t\t-e --enve\tCalculate envelope of EMG using supplied cf\n'
          '\t\t\t\tOptionally specify target_path if target directory is different\n'
          '\t\t-h --help\tDisplay this message\n'
          '\t\t-v --visu\tProcess a single .c3d file, and display interactive plots\n'
          '\t\t-z --zdir\tIndicate vertical travel in name (Up or Down)\n'
          '\t\t-p --para\tExtract gait parameters instead of angles\n'
          '\t\t\t\tNo target_path needed\n')


def main():
    if len(proc_input(sys.argv)) >= 2:
        if '-h' in sys.argv or '--help' in sys.argv:
            help_msg()
            return
        if '-v' in sys.argv or '--visu' in sys.argv:
            visu(diff='-d' in sys.argv or '--diff' in sys.argv, env='-e' in sys.argv or '--enve' in sys.argv)
            return
        else:
            dir_proc(diff='-d' in sys.argv or '--diff' in sys.argv, env='-e' in sys.argv or '--enve' in sys.argv)
    else:
        print('Please use at least 1 argument\n')
        help_msg()


if __name__ == "__main__":
    main()
