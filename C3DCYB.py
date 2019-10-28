#%%
import os
import sys
import itertools as it
import json

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

from C3D import C3DServer
from Geoms import Line, PointCloud, CoordSys
import Transforms as tf
from plt_axis import set_axes_equal
#%%
def calc_floating_angles(mat_a, mat_b, side='right', in_deg=True):
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
    if side == 'right':
        sin_gamma = -np.dot(axis, b_x)
        cos_gamma = np.dot(b_y, axis)
    else:
        sin_gamma = np.dot(axis, b_x)
        cos_gamma = np.dot(b_y, axis)
        
    alpha = np.arctan(sin_alpha/cos_alpha)
    beta = np.arccos(cos_beta)
    gamma = np.arctan(sin_gamma/cos_gamma)
    
    flexion = alpha
    external_rotation = gamma
    if side == 'right':
        adduction = beta-np.pi
    else:
        adduction = np.pi-beta
        
    angles = np.array([flexion, adduction, external_rotation], dtype=np.float32)
    if in_deg:
        return np.degrees(angles)
    else:
        return angles


# %%
dflt_c3d_f_name = '005_Stair11.c3d'
c3d_f_path = ''
dflt_json_f_name = 'Cluster.json'
json_f_path = ''
asis_breadth = None
#%%
global sl_fr

def main():
    global sl_fr
    global asis_breadth

    if len(sys.argv) >= 2:
        c3d_f_path = sys.argv[1]
    else:
        c3d_f_path = os.path.join(os.getcwd(), dflt_c3d_f_name)
    
    if len(sys.argv) >= 3:
        json_f_path = sys.argv[2]
    else:
        json_f_path = os.path.join(os.getcwd(), dflt_json_f_name)
    
    # Read a C3D file and extract all necessary info
    c3d = C3DServer()
    c3d.open_c3d(c3d_f_path)
    start_frame = c3d.get_start_frame_number()
    end_frame = c3d.get_end_frame_number()
    cnt_frames = c3d.get_total_frame_counts()
    arr_frames = np.linspace(start_frame, end_frame, cnt_frames, dtype=int)
    mkr_names = c3d.get_marker_names()
    dict_mkr_idx = {k:c3d.get_marker_index(k) for k in mkr_names}
    mkr_scale_factor = 0.001 if c3d.get_marker_unit() == 'mm' else 1.0
    mkr_pts = c3d.get_all_marker_coords_arr3d(mkr_scale_factor)
    dim_mkr_pts = mkr_pts.shape 
    mkr_pts_reshape = np.reshape(mkr_pts, (dim_mkr_pts[0]*dim_mkr_pts[1], dim_mkr_pts[2]))
    dict_mkr_coords = {k:mkr_pts[:,dict_mkr_idx[k],:] for k in mkr_names}
    dict_cl_mkr_names = json.load(open(json_f_path))
    dict_cl_mkr_idx = {k:[dict_mkr_idx[mkr] for mkr in v] for k, v in dict_cl_mkr_names.items()}
    dict_cl_line_idx = {k:list(it.combinations(v, 2)) for k, v in dict_cl_mkr_idx.items()}
    c3d.close_c3d()

    # Create virtual markers
    virtual_names = ['LVHI', 'LVKN', 'LVAN', 'RVHI', 'RVKN', 'RVAN']
    indices = [(nam, ind + len(mkr_names)+1) for ind, nam in enumerate(virtual_names)]
    mkr_names.extend(virtual_names)
    dict_mkr_idx.update(indices)
    dict_mkr_coords['LVAN'] = (dict_mkr_coords['LANK'] + dict_mkr_coords['LANM']) / 2
    dict_mkr_coords['RVAN'] = (dict_mkr_coords['RANK'] + dict_mkr_coords['RANM']) / 2
    dict_mkr_coords['LVKN'] = (dict_mkr_coords['LKNE'] + dict_mkr_coords['LKNM']) / 2
    dict_mkr_coords['RVKN'] = (dict_mkr_coords['RKNE'] + dict_mkr_coords['RKNM']) / 2

    # Hip joint center based on Vaughan, Davis and Connor: Dynamics of Human Gait
    v_hip = dict_mkr_coords['LASI'] - dict_mkr_coords['RASI']
    v_norm = np.linalg.norm(v_hip, axis=1)

    if len(sys.argv) >= 4:
        asis_breadth = sys.argv[3]
    elif asis_breadth == None:
        asis_breadth = np.mean(v_norm)

    v_hip = v_hip / v_norm[:, None] # Broadcasting division for each row
    temp_hip = dict_mkr_coords['LASI'] - dict_mkr_coords['SACR']
    w_hip = np.cross(temp_hip, v_hip)
    w_hip = w_hip / np.linalg.norm(w_hip, axis=1)[:, None]
    u_hip = np.cross(v_hip, w_hip)
    dict_mkr_coords['LVHI'] = dict_mkr_coords['SACR'] + asis_breadth*(0.598 * u_hip + 0.344 * v_hip - 0.29 * w_hip)
    dict_mkr_coords['RVHI'] = dict_mkr_coords['SACR'] + asis_breadth*(0.598 * u_hip - 0.344 * v_hip - 0.29 * w_hip)
    new_mkr_pts = np.stack((dict_mkr_coords['LVHI'], dict_mkr_coords['LVKN'], dict_mkr_coords['LVAN'],
                              dict_mkr_coords['RVHI'], dict_mkr_coords['RVKN'], dict_mkr_coords['RVAN']))
    new_mkr_pts = np.rollaxis(new_mkr_pts, 1)
    mkr_pts = np.concatenate((mkr_pts, new_mkr_pts), axis=1)

    # Create a figure using matplotlib
    fig = plt.figure(figsize=(14, 7))
    # Add an axes for 3D graph for markers
    ax = fig.add_axes([0.05, 0.2, 0.45, 0.7], projection='3d', proj_type='persp')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-135)
    title = ax.set_title('C3D viewer, frame={0:05d}'.format(start_frame))

    # Calcuate the proper range of x y z limits
    pts_min = np.min(mkr_pts_reshape, axis=0)
    pts_max = np.max(mkr_pts_reshape, axis=0)
    pts_range = pts_max-pts_min
    pts_mid = (pts_min+pts_max)*0.5
    pts_range_max = np.max(pts_range)
    ax.set_xlim(pts_mid[0]-pts_range_max*0.5, pts_mid[0]+pts_range_max*0.5)
    ax.set_ylim(pts_mid[1]-pts_range_max*0.5, pts_mid[1]+pts_range_max*0.5)
    ax.set_zlim(pts_mid[2]-pts_range_max*0.5, pts_mid[2]+pts_range_max*0.5)
    #set_axes_equal(ax)
    
    # A list to contain geometric class instances
    geom_objs = []
    
    # Create a PointCloud class instance in order to display the markers
    pts_obj = PointCloud('[MARKERS]', ax)
    pts_obj.set_point_names(mkr_names[:19])
    pts_obj.set_from_points(mkr_pts[:, :19, :])
    pts_obj.set_marker_style(3, 'b', 'b')
    pts_obj.draw_vis_objs(fr=0)
    geom_objs.append(pts_obj)

    vpts_obj = PointCloud('[VMARKERS]', ax)
    vpts_obj.set_point_names(mkr_names[19:])
    vpts_obj.set_from_points(mkr_pts[:, 19:, :])
    vpts_obj.set_marker_style(8, 'b', 'none')
    vpts_obj.draw_vis_objs(fr=0)
    geom_objs.append(vpts_obj)

    
    # Create several Line class instances in order to display the lines inside the clusters
    line_cnt=0
    for mkr_name, idx_list in dict_cl_line_idx.items():
        for mkr_idx in idx_list:
            line_obj = Line(name='Line'+str(line_cnt), axes=ax)
            line_obj.set_point_names([mkr_names[mkr_idx[0]], mkr_names[mkr_idx[1]]])
            line_obj.set_from_2points(mkr_pts[:,mkr_idx[0],:], mkr_pts[:,mkr_idx[1],:])
            line_obj.draw_vis_objs(fr=0)
            geom_objs.append(line_obj)
            line_cnt = line_cnt+1
    
    # Create a CoordSys class instance for the right thigh
    csys_r_thigh = CoordSys('Right_Thigh', ax)
    csys_r_thigh.set_from_3points(dict_mkr_coords["RKNE"], dict_mkr_coords["RKNM"], dict_mkr_coords["RTHI"])
    csys_r_thigh_rot_mat = Rotation.from_euler('XYZ', [-90, 0, 180], degrees=True).as_dcm()
    csys_r_thigh.apply_intrinsic_rotation(csys_r_thigh_rot_mat)
    csys_r_thigh.update_axes_pos()
    csys_r_thigh.draw_vis_objs(fr=0)
    geom_objs.append(csys_r_thigh)
    # Create a CoordSys class instance for the right shank
    csys_r_shank = CoordSys('Right_Shank', ax)
    csys_r_shank.set_from_3points(dict_mkr_coords["RANK"], dict_mkr_coords["RANM"], dict_mkr_coords["RTIB"])
    csys_r_shank_rot_mat = Rotation.from_euler('XYZ', [-90, 0, 180], degrees=True).as_dcm()
    csys_r_shank.apply_intrinsic_rotation(csys_r_shank_rot_mat)
    csys_r_shank.update_axes_pos()
    csys_r_shank.draw_vis_objs(fr=0)
    geom_objs.append(csys_r_shank)
    # Create a CoordSys class instace for the left thigh
    csys_l_thigh = CoordSys('Left_Thigh', ax)
    csys_l_thigh.set_from_3points(dict_mkr_coords["LKNE"], dict_mkr_coords["LKNM"], dict_mkr_coords["LTHI"])
    csys_l_thigh_rot_mat = Rotation.from_euler('XYZ', [-90, 0, 0], degrees=True).as_dcm()
    csys_l_thigh.apply_intrinsic_rotation(csys_l_thigh_rot_mat)
    csys_l_thigh.update_axes_pos()
    csys_l_thigh.draw_vis_objs(fr=0)
    geom_objs.append(csys_l_thigh)
    # Create a CoordSys class instance for the left shank
    csys_l_shank = CoordSys('Left_Shank', ax)
    csys_l_shank.set_from_3points(dict_mkr_coords["LANK"], dict_mkr_coords["LANM"], dict_mkr_coords["LTIB"])
    csys_l_shank_rot_mat = Rotation.from_euler('XYZ', [-90, 0, 0], degrees=True).as_dcm()
    csys_l_shank.apply_intrinsic_rotation(csys_l_shank_rot_mat)
    csys_l_shank.update_axes_pos()
    csys_l_shank.draw_vis_objs(fr=0)
    geom_objs.append(csys_l_shank)    
    
    # Calcuation of the right knee flexion angles in two different ways
    r_knee_ang_floating = np.zeros((cnt_frames, 3), dtype=np.float32)
    r_knee_ang_euler = np.zeros((cnt_frames, 3), dtype=np.float32)
    for i in range(cnt_frames):
        r_knee_ang_floating[i] = calc_floating_angles(csys_r_thigh.arr_rot[i], csys_r_shank.arr_rot[i], side='right')
        r_knee_ang_euler[i] = tf.calc_rot_angles_euler(csys_r_thigh.arr_rot[i], csys_r_shank.arr_rot[i])

    r_knee_flex_floating = r_knee_ang_floating[:,0]
    r_knee_flex_euler = -r_knee_ang_euler[:,0]   
    r_knee_add_floating = r_knee_ang_floating[:,1]
    r_knee_add_euler = r_knee_ang_euler[:,1]
    r_knee_ext_rot_floating = r_knee_ang_floating[:,2]
    r_knee_ext_rot_euler = -r_knee_ang_euler[:,2]
    
    # Calculation of the left knee flexion angle in two different ways
    l_knee_ang_floating = np.zeros((cnt_frames, 3), dtype=np.float32)
    l_knee_ang_euler = np.zeros((cnt_frames, 3), dtype=np.float32)
    for i in range(cnt_frames):
        l_knee_ang_floating[i] = calc_floating_angles(csys_l_thigh.arr_rot[i], csys_l_shank.arr_rot[i], side='left')
        l_knee_ang_euler[i] = tf.calc_rot_angles_euler(csys_l_thigh.arr_rot[i], csys_l_shank.arr_rot[i])
    l_knee_flex_floating = l_knee_ang_floating[:,0]
    l_knee_flex_euler = -l_knee_ang_euler[:,0]   
    l_knee_add_floating = l_knee_ang_floating[:,1]
    l_knee_add_euler = l_knee_ang_euler[:,1]
    l_knee_ext_rot_floating = l_knee_ang_floating[:,2]
    l_knee_ext_rot_euler = -l_knee_ang_euler[:,2]    
    
    
    ax_2d_0 = fig.add_axes([0.55, 0.55, 0.4, 0.3])
    ax_2d_0.set_xlim(start_frame, end_frame)
    ax_2d_0.set_ylim(-10, 100)
    vline_2d_0 = ax_2d_0.axvline(x=start_frame, ymin=0, ymax=1, color=(0,1,0), linewidth=1.0, linestyle='--')    
    title_2d_0 = ax_2d_0.set_title('Right Knee Flexion-Extension')
    ax_2d_0.plot(arr_frames, r_knee_flex_floating, linewidth=1.0, color='r', label='Floating', linestyle='--')
    ax_2d_0.plot(arr_frames, r_knee_flex_euler, linewidth=1.0, color='b', label='Euler', linestyle='-.')
    ax_2d_0.legend(loc='upper right', fontsize='small')

    ax_2d_1 = fig.add_axes([0.55, 0.15, 0.4, 0.3])
    ax_2d_1.set_xlim(start_frame, end_frame)
    ax_2d_1.set_ylim(-10, 100)
    vline_2d_1 = ax_2d_1.axvline(x=start_frame, ymin=0, ymax=1, color=(0,1,0), linewidth=1.0, linestyle='--')    
    title_2d_1 = ax_2d_1.set_title('Left Knee Flexion-Extension')
    ax_2d_1.plot(arr_frames, l_knee_flex_floating, linewidth=1.0, color='r', label='Floating', linestyle='--')
    ax_2d_1.plot(arr_frames, l_knee_flex_euler, linewidth=1.0, color='b', label='Euler', linestyle='-.')
    ax_2d_1.legend(loc='upper right', fontsize='small')

    ax_fr = fig.add_axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    sl_fr = Slider(ax_fr, 'Frame', start_frame, end_frame, valinit=start_frame, valstep=1, valfmt='%05d')

    def update(val):
        fr_no = int(sl_fr.val)
        fr_idx = fr_no-start_frame
        title.set_text('C3D viewer, frame={0:05d}'.format(fr_no))
        for geom in geom_objs:
            geom.update_vis_objs(fr_idx)
        #vline_2d.set_xdata(fr_idx)
        vline_2d_0.set_xdata(fr_no)
        vline_2d_1.set_xdata(fr_no)
        fig.canvas.draw()
        fig.canvas.flush_events()

    sl_fr.on_changed(update)
    
    plt.show()
    
    return sl_fr

if __name__ == "__main__":
    main()