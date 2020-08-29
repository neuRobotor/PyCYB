from jcs import *
from functools import partial as p
import sys
import scipy.io

class ExtendSet(LegSet):

    def __init__(self, c3d_file, emg_file=None):
        super(ExtendSet, self).__init__(c3d_file=c3d_file, emg_file=emg_file)
        self.list_marker = ['VSACR',
                            'RASI',  'LASI',
                            'VRHJC', 'VLHJC',
                            'RPFE',  'LPFE',
                            'RAFE',  'LAFE',
                            'VRKJC', 'VLKJC',
                            'RPTI',  'LPTI',
                            'RATI',  'LATI',
                            'VRAJC', 'VLAJC',
                            'VRTOE', 'VLTOE',
                            'VRHEE', 'VLHEE',
                            'RLCA',  'LLCA']

    def marker_preproc(self):
        return

    # region Segment definitions
    def pelvis_seg(self):
        i_vect = (self.dict_marker['RASI'] - self.dict_marker['LASI'])
        k_vect = np.cross(self.dict_marker['RASI'] - self.dict_marker['VSACR'],
                          self.dict_marker['LASI'] - self.dict_marker['VSACR'], axis=1)
        j_vect = np.cross(k_vect, i_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name='Pelvis')

    def thigh_seg(self, s='R'):
        k_vect = self.dict_marker[s + 'VHJC'] - self.dict_marker[s + 'VKJC']
        j_vect = self.dict_marker[s + 'AFE'] - self.dict_marker[s + 'PFE']
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Thigh')

    def shank_seg(self, s='R'):
        k_vect = self.dict_marker[s + 'VKJC'] - self.dict_marker[s + 'VAJC']
        j_vect = np.cross(k_vect, self.dict_marker[s + 'ATI'] - self.dict_marker[s + 'PTI'])
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Shank')

    def foot_seg(self, s='R'):
        k_vect = self.dict_marker[s + 'VHEE'] - self.dict_marker[s + 'VTOE']
        j_vect = (-1 if s == 'L' else 1) * \
                 np.cross(k_vect, self.dict_marker[s + 'VHEE'] - self.dict_marker[s + 'LCA'])
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Foot')
    # endregion

    def get_emg_data(self):
        if not self.dict_emg:
            mat = scipy.io.loadmat(self.emg_file)
            self.dict_emg = {desc: emg for desc, emg in zip(mat['Description'].T, mat['Data'].T)}
        return self.dict_emg


if __name__ == '__main__':
    dir_proc(ExtendSet, sys.argv[1],
             dir_path=sys.argv[2])
