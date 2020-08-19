import numpy as np
import scipy.io
import os
import json
import re
from utility.C3D import C3DServer
from c3dcyb import calc_floating_angles
from functools import partial
from jcs import Segment, JCS
import multiprocessing


def extend_angle_est(dict_mkr_coords, segment_dict: dict):
    # region Calculation of local coordinate systems
    # Access frame from rs = reference_sys['Name'] using rs[i]
    # Access all frames of a given vector using rs[:, :, n]
    reference_sys = {}

    def norm(x):
        return x / np.linalg.norm(x, axis=1)[:, None]

    for segment, markers in segment_dict.items():
        if segment == 'Pelvis':
            k_vect = norm(np.cross(dict_mkr_coords[markers[1]], dict_mkr_coords[2]))
            i_vect = norm(dict_mkr_coords[markers[1]]-dict_mkr_coords[markers[2]])
            j_vect = np.cross(k_vect, i_vect)
        else:
            k_vect = norm(dict_mkr_coords[markers[0]] - dict_mkr_coords[markers[1]])
            j_vect = (-1 if 'L' in markers[0] else 1) * norm(np.cross(k_vect,
                                                                      dict_mkr_coords[markers[2]] -
                                                                      dict_mkr_coords[markers[0]], axis=1))
            i_vect = np.cross(j_vect, k_vect)
        reference_sys[segment] = np.dstack((i_vect, j_vect, k_vect))

    # region Angle estimation
    cnt_frames = len(dict_mkr_coords['SACR'])
    segment_names = ['Thigh', 'Shank', 'Foot']
    joint_names = ['Hip', 'Knee', 'Ankle']
    joint_angles = {}
    for cur_side in ['L', 'R']:
        # Hip joint
        joint_angles[cur_side + joint_names[0]] = np.zeros((cnt_frames, 3), dtype=np.float32)
        for i in range(cnt_frames):
            joint_angles[cur_side + joint_names[0]][i] = calc_floating_angles(reference_sys['Pelvis'][i],
                                                                              reference_sys[cur_side + 'Thigh'][i],
                                                                              in_deg=False)
        # Rest of the joints
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


def extend_proc(c3d_name, z_dir=False, param=False):
    with C3DServer() as c3d:
        # region Read a C3D file and extract all necessary info
        c3d.open_c3d(c3d_name)
        dict_mkr_idx = {k: c3d.get_marker_index(k) for k in c3d.get_marker_names()}
        mkr_scale_factor = 0.001 if c3d.get_marker_unit() == 'mm' else 1.0
        mkr_pts = c3d.get_all_marker_coords_arr3d(mkr_scale_factor)
        dict_mkr_coords = {k: mkr_pts[:, dict_mkr_idx[k], :] for k in c3d.get_marker_names()}
        c3d.close_c3d()
        # endregion


        seg_dict = {
            'Pelvis': ('SACR', 'RASI', 'LASI'),
            'LThigh': ('VLHJC', 'VLKJC', 'LPFE'),
            'RThigh': ('VRHJC', 'VRKJC', 'RPFE'),
            'LShank': ('VLKJC', 'VLAJC', 'LPTI'),
            'RShank': ('VRKJC', 'VRAJC', 'RPTI'),
            'LFoot': ('VLHEE', 'VLTOE', 'VLAJC'),
            'RFoot': ('VRHEE', 'VRTOE', 'VRAJC'),
        }
        joint_angles, _ = extend_angle_est(dict_mkr_coords, seg_dict)

        direction = None
        if z_dir:
            if dict_mkr_coords['SACR'][-1, 2] > dict_mkr_coords['SACR'][0, 2] * 1.1:
                direction = 'Up'
            elif dict_mkr_coords['SACR'][-1, 2] < dict_mkr_coords['SACR'][0, 2] * 0.9:
                direction = 'Down'

    return joint_angles, direction


def f_para(f_path, ptarget_dir, pz, pp):
    f_name = os.path.basename(os.path.normpath(f_path))
    print('Processing file ' + f_name + '...')
    joint_angles, direc = extend_proc(f_path, z_dir=pz, param=pp)
    name = os.path.splitext(f_name)[0]
    list_vals = [ar.tolist() for ar in joint_angles.values()]
    keys = joint_angles.keys()
    save_angles = {key: li for key, li in zip(keys, list_vals)}
    with open(ptarget_dir + '\\' + name + '.json', 'w') as fp:
        json.dump(save_angles, fp, indent=4)


def files_proc(f_paths, target_dir):
    f = partial(f_para, ptarget_dir=target_dir, pz=False, pp=False)
    nProcess = multiprocessing.cpu_count()
    with multiprocessing.Pool(nProcess) as pool:
        pool.map(f, f_paths)


def main():
    file_paths = list()
    for (dirpath, dirnames, filenames) in os.walk('.'):
        file_paths += [os.path.join(dirpath, file) for file in filenames if file.endswith('.c3d')]
    files_proc(file_paths, '.')
    return


if __name__ == '__main__':
    main()
