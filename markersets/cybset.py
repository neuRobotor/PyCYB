from jcs import *
from functools import partial as p
import sys


class CybSet(LegSet):

    def __init__(self, c3d_file, emg_file=None):
        super(CybSet, self).__init__(c3d_file=c3d_file, emg_file=emg_file)
        self.list_marker = ['SACR',
                            'RASI', 'LASI',
                            'RTHI', 'LTHI',
                            'RKNE', 'LKNE',
                            'RKNM', 'LKNM',
                            'RTIB', 'LTIB',
                            'RANK', 'LANK',
                            'RANM', 'LANM',
                            'RHEE', 'LHEE',
                            'RTOE', 'LTOE']

        self.list_emg = ["Sensor {}.EMG{}".format(i, i) for i in range(1, 9)]

    def marker_preproc(self):
        # Define virtual markers
        self.dict_marker['LVAN'] = (self.dict_marker['LANK'] + self.dict_marker['LANM']) / 2
        self.dict_marker['RVAN'] = (self.dict_marker['RANK'] + self.dict_marker['RANM']) / 2
        self.dict_marker['LVKN'] = (self.dict_marker['LKNE'] + self.dict_marker['LKNM']) / 2
        self.dict_marker['RVKN'] = (self.dict_marker['RKNE'] + self.dict_marker['RKNM']) / 2

        # Hip joint center based on Vaughan, Davis and Connor: Dynamics of Human Gait
        v_hip = self.dict_marker['LASI'] - self.dict_marker['RASI']
        v_norm = np.linalg.norm(v_hip, axis=1)
        asis_breadth = np.mean(v_norm)
        v_hip = v_hip / v_norm[:, None]  # Broadcasting division for each row

        temp_hip = self.dict_marker['LASI'] - self.dict_marker['SACR']
        w_hip = norm(np.cross(temp_hip, v_hip))
        u_hip = np.cross(v_hip, w_hip)
        self.dict_marker['LVHI'] = self.dict_marker['SACR'] + asis_breadth * (
                0.598 * u_hip + 0.344 * v_hip - 0.29 * w_hip)
        self.dict_marker['RVHI'] = self.dict_marker['SACR'] + asis_breadth * (
                0.598 * u_hip - 0.344 * v_hip - 0.29 * w_hip)

    # region Segment definitions
    def pelvis_seg(self):
        i_vect = (self.dict_marker['RASI'] - self.dict_marker['LASI'])
        k_vect = np.cross(self.dict_marker['RASI'] - self.dict_marker['SACR'],
                          self.dict_marker['LASI'] - self.dict_marker['SACR'], axis=1)
        j_vect = np.cross(k_vect, i_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name='Pelvis')

    def thigh_seg(self, s='R'):
        k_vect = self.dict_marker[s + 'VHI'] - self.dict_marker[s + 'VKN']
        j_vect = (-1 if s == 'L' else 1) * \
                 np.cross(k_vect, self.dict_marker[s + 'THI'] - self.dict_marker[s + 'VHI'], axis=1)
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Thigh')

    def shank_seg(self, s='R'):
        k_vect = self.dict_marker[s + 'VKN'] - self.dict_marker[s + 'VAN']
        j_vect = (-1 if s == 'L' else 1) * \
                 np.cross(k_vect, self.dict_marker[s + 'TIB'] - self.dict_marker[s + 'VKN'], axis=1)
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Shank')

    def foot_seg(self, s='R'):
        k_vect = self.dict_marker[s + 'HEE'] - self.dict_marker[s + 'TOE']
        j_vect = (-1 if s == 'L' else 1) * \
                 np.cross(k_vect, self.dict_marker[s + 'ANK'] - self.dict_marker[s + 'ANM'], axis=1)
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Foot')
    # endregion

    def get_emg_data(self, c3d=None):
        if self.dict_emg and self.emg_freq:
            return self.dict_emg, self.emg_freq
        if c3d is None:
            with C3DServer() as c3d:
                c3d.open_c3d(self.c3d_file)
                return c3d.get_analog_dict(self.list_emg), c3d.get_analog_frame_rate()
        return c3d.get_analog_dict(self.list_emg), c3d.get_analog_frame_rate()


if __name__ == '__main__':
    dir_proc(CybSet, sys.argv[1],
             dir_path=sys.argv[2])
