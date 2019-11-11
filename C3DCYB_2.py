import os
import sys
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from C3D import C3DServer
from Geoms import Line, PointCloud, CoordSys
from emg_proc import envelope
import json

dflt_c3d_f_name = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB005\005_Stair\005_Stair11.c3d'
c3d_f_path = ''
global sl_fr


def calc_floating_angles(mat_a, mat_b, side='R', in_deg=True):
    if type(mat_a) is not np.ndarray: return False
    if type(mat_b) is not np.ndarray: return False
    a_x = mat_a[:,0]
    a_y = mat_a[:,1]
    a_z = mat_a[:,2]
    b_x = mat_b[:,0]
    b_y = mat_b[:,1]
    b_z = mat_b[:,2]
    axis = np.cross(b_z, a_x)/np.linalg.norm(np.cross(b_z, a_x))

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
    gamma = np.arctan(sin_gamma/cos_gamma)

    flexion = alpha
    external_rotation = gamma
    if side == 'R':
        adduction = beta-np.pi
    else:
        adduction = np.pi-beta

    angles = np.array([flexion, adduction, external_rotation], dtype=np.float32)
    if in_deg:
        return np.degrees(angles)
    else:
        return angles


def dir_proc(dir_in):
    for file in [f for f in os.listdir(dir_in) if f.endswith('.c3d')]:
        joint_angles, emg_data = c3d_proc(dir_in+'\\'+file)
        name = os.path.splitext(file)[0]
        list_vals = [ar.tolist() for ar in joint_angles.values()]
        keys = joint_angles.keys()
        save_angles = {key : li for key, li in zip(keys, list_vals)}
        with open(dir_in + '\\' + name + '.json', 'w') as fp:
            json.dump(save_angles, fp, indent=4)
        np.savetxt(dir_in + '\\' + name + '_emg.csv',  emg_data, delimiter=",")
    return


def angle_est(dict_mkr_coords, asis_breadth=None):
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

    # region Calculation of local coordinate systems
    # Access frame from rs = reference_sys['Name'] using rs[i]
    # Access all frames of a given vector using rs[:, :, n]
    reference_sys = {}
    segment_names = ['Thigh', 'Shank', 'Foot']
    ref_markers = [['VHI', 'VKN', 'THI'], ['VKN', 'VAN', 'TIB'], ['HEE', 'TOE', 'VAN']]
    for side in ['L', 'R']:
        for seg_enum, cur_seg in enumerate(segment_names):
            k_vect = (dict_mkr_coords[side + ref_markers[seg_enum][0]] - dict_mkr_coords[
                side + ref_markers[seg_enum][1]])
            k_vect = k_vect / np.linalg.norm(k_vect, axis=1)[:, None]
            j_vect = np.cross(k_vect,
                              dict_mkr_coords[side + ref_markers[seg_enum][2]] -
                              dict_mkr_coords[side + ref_markers[seg_enum][0]], axis=1)
            j_vect = (-1 if side == 'L' else 1) * j_vect / np.linalg.norm(j_vect, axis=1)[:, None]
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

    # region Angle estimation
    cnt_frames = len(dict_mkr_coords['SACR'])
    joint_names = ['Hip', 'Knee', 'Ankle']
    joint_angles = {}
    for cur_side in ['L', 'R']:
        joint_angles[cur_side + 'Hip'] = np.zeros((cnt_frames, 3), dtype=np.float32)
        for i in range(cnt_frames):
            joint_angles[cur_side + 'Hip'][i] = calc_floating_angles(reference_sys['Pelvis'][i],
                                                                     reference_sys[cur_side + 'Thigh'][i],
                                                                     in_deg=False)
        for seg in range(len(segment_names) - 1):
            joint_angles[cur_side + joint_names[seg + 1]] = np.zeros((cnt_frames, 3), dtype=np.float32)
            for i in range(cnt_frames):
                joint_angles[cur_side + joint_names[seg + 1]][i] = \
                    calc_floating_angles(reference_sys[cur_side + segment_names[seg]][i],
                                         reference_sys[cur_side + segment_names[seg + 1]][i], side=cur_side,
                                         in_deg=False)

    for name in joint_angles.keys():
        for ang in range(3):
            joint_angles[name][:, ang] = np.unwrap(joint_angles[name][:, ang])
    # endregion
    return joint_angles, dict_mkr_coords


def c3d_proc(c3d_name, asis_breadth=None, emg_lowpass=50):
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
        for i in range(8):
            emg_data[i] = c3d.get_analog_data("Sensor {}.EMG{}".format(i + 1, i + 1))
            emg_data[i] = envelope(emg_data[i], low_pass=emg_lowpass)
        c3d.close_c3d()
        # endregion

        # region crop out regions with missing
        bool_idx = np.all(np.any(list(dict_mkr_coords.values()), axis=2), axis=0)
        for key in dict_mkr_coords.keys():
            dict_mkr_coords[key] = dict_mkr_coords[key][bool_idx, :]
        emg_data = [np.array([emg[i] for i in np.argwhere(np.repeat(bool_idx, 20)).flatten().tolist()])
                    for emg in emg_data]
        # endregion

        joint_angles, _ = angle_est(dict_mkr_coords, asis_breadth)

    return joint_angles, emg_data



def main():

    # region Globals and Commandline arguements
    global sl_fr
    asis_breadth = None

    if len(sys.argv) >= 2:
        c3d_f_path = sys.argv[1]
    else:
        c3d_f_path = os.path.join(os.getcwd(), dflt_c3d_f_name)
    if len(sys.argv) >= 3:
        asis_breadth = sys.argv[3]
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
    dict_mkr_idx = {k:c3d.get_marker_index(k) for k in mkr_names}
    mkr_scale_factor = 0.001 if c3d.get_marker_unit() == 'mm' else 1.0
    mkr_pts = c3d.get_all_marker_coords_arr3d(mkr_scale_factor)
    dim_mkr_pts = mkr_pts.shape
    mkr_pts_reshape = np.reshape(mkr_pts, (dim_mkr_pts[0]*dim_mkr_pts[1], dim_mkr_pts[2]))
    dict_mkr_coords = {k:mkr_pts[:,dict_mkr_idx[k],:] for k in mkr_names}
    emg_data = [None]*8
    for i in range(8):
        emg_data[i] = c3d.get_analog_data("Sensor {}.EMG{}".format(i+1, i+1))
        emg_data[i] = envelope(emg_data[i], low_pass=50)
    c3d.close_c3d()
    # endregion

    # region crop out regions with missing
    bool_idx = np.all(np.any(list(dict_mkr_coords.values()), axis=2), axis=0)
    for key in dict_mkr_coords.keys():
        dict_mkr_coords[key] = dict_mkr_coords[key][bool_idx, :]
    mkr_pts = mkr_pts[bool_idx, :, :]
    emg_data = [np.array([emg[i] for i in np.argwhere(np.repeat(bool_idx, 20)).flatten().tolist()]) for emg in emg_data]
    cnt_frames = len(dict_mkr_coords['SACR'])
    arr_frames = arr_frames[bool_idx]
    start_frame = arr_frames[0]
    end_frame = arr_frames[-1]
    # endregion


    # region Create virtual markers
    virtual_names = ['LVHI', 'LVKN', 'LVAN', 'RVHI', 'RVKN', 'RVAN']
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
    ax = fig.add_subplot(gs[:3, :3], projection='3d', proj_type='persp')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-135)
    title = ax.set_title('C3D viewer, frame={0:05d}'.format(start_frame))

    # Calculate the proper range of x y z limits
    pts_min = np.min(mkr_pts_reshape, axis=0)
    pts_max = np.max(mkr_pts_reshape, axis=0)
    pts_range = pts_max-pts_min
    pts_mid = (pts_min+pts_max)*0.5
    pts_range_max = np.max(pts_range)
    ax.set_xlim(pts_mid[0]-pts_range_max*0.5, pts_mid[0]+pts_range_max*0.5)
    ax.set_ylim(pts_mid[1]-pts_range_max*0.5, pts_mid[1]+pts_range_max*0.5)
    ax.set_zlim(pts_mid[2]-pts_range_max*0.5, pts_mid[2]+pts_range_max*0.5)

    # A list to contain geometric class instances
    geom_objs = []

    # Create a PointCloud class instance in order to display the markers and virtual markers
    pts_obj = PointCloud('[MARKERS]', ax)
    pts_obj.set_point_names(mkr_names)
    pts_obj.set_from_points(mkr_pts[:, :len(mkr_names), :])
    pts_obj.set_marker_style(2, 'tab:blue', 'tab:blue')
    pts_obj.draw_vis_objs(fr=0)
    geom_objs.append(pts_obj)

    vpts_obj = PointCloud('[VMARKERS]', ax)
    vpts_obj.set_point_names(virtual_names)
    vpts_obj.set_from_points(mkr_pts[:, -len(virtual_names):, :])
    vpts_obj.set_marker_style(8, 'tab:orange', 'none', )
    vpts_obj.draw_vis_objs(fr=0)
    geom_objs.append(vpts_obj)

    # region Create several Line class instances in order to display the lines in the leg
    line_obj = Line(name='RThigh' , axes=ax)
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
    joint_names = ['Hip', 'Knee', 'Ankle']
    # region Plotting angles
    vlines = []
    for row in range(3):
        for col, side in enumerate(['L', 'R']):
            cur_ax = fig.add_subplot(gs[row, 3+col])
            cur_ax.set_xlim(start_frame, end_frame)
            cur_ax.set_ylim(-2, 2)
            vlines.append(cur_ax.axvline(x=start_frame, ymin=0, ymax=1, color=(0, 1, 0), linewidth=1.0, linestyle='--'))
            cur_ax.set_title(side + joint_names[row])
            cur_ax.plot(arr_frames, joint_angles[side + joint_names[row]][:, 0],
                        linewidth=1.0, color='tab:blue', label='Flexion')
            cur_ax.legend(loc='upper right', fontsize='small')

    ax_fr = fig.add_subplot(gs[3, :], facecolor='lightgoldenrodyellow')
    sl_fr = Slider(ax_fr, 'Frame', start_frame, end_frame, valinit=start_frame, valstep=1, valfmt='%05d')


    def update(val):
        fr_no = int(sl_fr.val)
        fr_idx = fr_no-start_frame
        title.set_text('C3D viewer, frame={0:05d}'.format(fr_no))
        for geom in geom_objs:
            geom.update_vis_objs(fr_idx)
        for vl in vlines:
            vl.set_xdata(fr_no)
        fig.canvas.draw()
        fig.canvas.flush_events()

    sl_fr.on_changed(update)

    plt.show()

    return sl_fr
    # endregion


if __name__ == "__main__":
    main()
