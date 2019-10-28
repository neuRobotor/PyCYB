#%%
import numpy as np
from scipy.spatial.transform import Rotation as R
#%%
def proj_vec_onto_plane(v, n):
    return v-np.dot(v, n)*n

def calc_vec_angle(a, b, ref):
    angle = np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    sign = np.sign(np.cross(a, b).dot(ref))
    return sign*angle

def calc_rot_angles_euler(mat_a, mat_b, seq='XYZ', in_deg=True):
    if type(mat_a) is not np.ndarray: return False
    if type(mat_b) is not np.ndarray: return False
    '''
    Intrinsic rotation: B = A*R --> R = inv(A)*B
    '''
    rot_mat = np.dot(mat_a.T, mat_b)
    r = R.from_dcm(rot_mat)
    arr_angles = r.as_euler(seq, degrees=in_deg)
    return arr_angles
    
def calc_rot_angle_proj(mat_a, mat_b, rot_axis='X', proj_axis='Y', in_deg=True):
    if type(mat_a) is not np.ndarray: return False
    if type(mat_b) is not np.ndarray: return False
    if rot_axis == proj_axis: return False
    dict_axis_idx = {'X':0, 'Y':1, 'Z':2}
    rot_axis_idx = dict_axis_idx[rot_axis]
    proj_axis_idx = dict_axis_idx[proj_axis]  
    v_a_normal = mat_a[:,rot_axis_idx]
    v_a_ref = mat_a[:,proj_axis_idx]
    v_b_ref = mat_b[:,proj_axis_idx]
    v_b_proj = proj_vec_onto_plane(v_b_ref, v_a_normal)
    angle = calc_vec_angle(v_a_ref, v_b_proj, v_a_normal)
    if in_deg:
        return np.degrees(angle)
    else:
        return angle
    
def create_coord_sys_v2(axis_v):
    dict_cross_v = {}
    for i in range(3):
        v = np.array([1.0 if x==i else 0.0 for x in range(3)], dtype=float)
        cross_v = np.cross(axis_v, v)
        dict_cross_v.update({np.linalg.norm(cross_v): cross_v})
    dict_cross_v_ordered = dict(sorted(dict_cross_v.items(), reverse=True))
    cross_v = list(dict_cross_v_ordered.values())[0]
    cross_v_unit = cross_v/np.linalg.norm(cross_v)
    
    axis_new = np.cross(axis_v, cross_v_unit)/np.linalg.norm(np.cross(axis_v, cross_v_unit))
    axes = np.zeros((3, 3), dtype=float)
    axes[2,:] = axis_v
    dot_prod_x = np.dot(np.array([1.0, 0.0, 0.0], dtype=float), axis_new.T)
    dot_prod_y = np.dot(np.array([0.0, 1.0, 0.0], dtype=float), axis_new.T)
    if np.absolute(dot_prod_x) >= np.absolute(dot_prod_y):
        axes[0,:] = np.sign(dot_prod_x)*axis_new
        axes[1,:] = np.cross(axes[2,:], axes[0,:])
    else:
        axes[1,:] = np.sign(dot_prod_y)*axis_new
        axes[0,:] = np.cross(axes[1,:], axes[2,:])
    rot_mat = axes.T
    return rot_mat

def create_coord_sys(axis_v):
    dict_cross_v = {}
    for i in range(3):
        v = np.array([1.0 if x==i else 0.0 for x in range(3)], dtype=float)
        cross_v = np.cross(axis_v, v)
        dict_cross_v.update({np.linalg.norm(cross_v): cross_v})
    dict_cross_v_ordered = dict(sorted(dict_cross_v.items(), reverse=True))
    cross_v = list(dict_cross_v_ordered.values())[0]
    cross_v_unit = cross_v/np.linalg.norm(cross_v)
    axes = np.zeros((3, 3), dtype=float)
    axes[2,:] = axis_v
    axis_new = np.cross(axis_v, cross_v_unit)/np.linalg.norm(np.cross(axis_v, cross_v_unit))    
    dot_prod_axis_new = np.dot(np.array([1.0, 0.0, 0.0], dtype=float), axis_new.T)
    axes[0,:] = np.sign(dot_prod_axis_new)*axis_new
    axes[1,:] = np.cross(axes[2,:], axes[0,:])
    rot_mat = axes.T
    return rot_mat
