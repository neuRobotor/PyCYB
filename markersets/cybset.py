from jcs import *
from functools import partial as p

class CybSet(MarkerSet):

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

        self.dict_segment = {'Pelvis': self.pelvis_seg,
                             'RThigh': p(self.thigh_seg, s='R'),    'LThigh': p(self.thigh_seg, s='L'),
                             'RShank': p(self.shank_seg, s='R'),    'LShank': p(self.shank_seg, s='L'),
                             'RFoot': p(self.foot_seg, s='R'),      'LFoot': p(self.foot_seg, s='L')}

        self.dict_joint = {'RHip': ('Pelvis', 'RThigh', 'R'),   'LHip': ('Pelvis', 'LThigh', 'L'),
                           'RKnee': ('RThigh', 'RShank', 'R'),  'LKnee': ('LThigh', 'LShank', 'L'),
                           'RFoot': ('RShank', 'RFoot', 'R'),   'LFoot': ('LShank', 'LFoot', 'L')}

    def pelvis_seg(self):
        return

    def thigh_seg(self, s='R'):
        return

    def shank_seg(self, s='R'):
        return

    def foot_seg(self, s='R'):
        return
