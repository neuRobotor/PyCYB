from jcs import *
from functools import partial as p
import sys
import scipy.io
from scipy.signal import resample

def get_file_pairs(subject_dir):
    c3d_list = list()
    mat_list = list()
    for root, dirs, files in os.walk(subject_dir):
        for file in files:
            if '.c3d' in file:
                c3d_list.append(os.path.join(root, file))
            elif '.mat' in file:
                mat_list.append(os.path.join(root, file))

    trials = [os.path.splitext(os.path.basename(p))[0]+'.' for p in c3d_list]

    pairs = [(p_c3d, p_mat) for tr in trials for p_c3d in c3d_list if tr in p_c3d
             for p_mat in mat_list if tr in p_mat]
    return pairs


class ExtendSet(LegSet):

    def __init__(self, c3d_file, emg_file=None, target_f=4000):
        super(ExtendSet, self).__init__(c3d_file=c3d_file, emg_file=emg_file)
        self.list_marker = ['VSACR',
                            'VRASI',  'VLASI',
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
        self.target_f = target_f

    def marker_preproc(self):
        for key in self.dict_marker.keys():
            self.dict_marker[key][np.sum(self.dict_marker[key], 1) == 0, :] = np.nan

        return

    # region Segment definitions
    def pelvis_seg(self):
        i_vect = (self.dict_marker['VRASI'] - self.dict_marker['VLASI'])
        k_vect = np.cross(self.dict_marker['VRASI'] - self.dict_marker['VSACR'],
                          self.dict_marker['VLASI'] - self.dict_marker['VSACR'], axis=1)
        j_vect = np.cross(k_vect, i_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name='Pelvis')

    def thigh_seg(self, s='R'):
        k_vect = self.dict_marker['V' + s + 'HJC'] - self.dict_marker['V' + s + 'KJC']
        j_vect = self.dict_marker[s + 'AFE'] - self.dict_marker[s + 'PFE']
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Thigh')

    def shank_seg(self, s='R'):
        k_vect = self.dict_marker['V' + s + 'KJC'] - self.dict_marker['V' + s + 'AJC']
        j_vect = np.cross(k_vect, self.dict_marker[s + 'ATI'] - self.dict_marker[s + 'PTI'])
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Shank')

    def foot_seg(self, s='R'):
        k_vect = self.dict_marker['V' + s + 'HEE'] - self.dict_marker['V' + s + 'TOE']
        j_vect = (-1 if s == 'L' else 1) * \
                 np.cross(k_vect, self.dict_marker['V' + s + 'HEE'] - self.dict_marker[s + 'LCA'])
        i_vect = np.cross(j_vect, k_vect)
        return Segment(lateral=i_vect, frontal=j_vect, longitudinal=k_vect, name=s + 'Foot')
    # endregion

    def get_emg_data(self, c3d=None):
        if not self.dict_emg and not self.emg_freq:
            mat = scipy.io.loadmat(self.emg_file)
            descs = [d[0][0] for d in mat['Description']]

            self.dict_emg = {desc: emg for desc, emg in zip(descs, mat['Data'].T)}
            self.emg_freq = mat['SamplingFrequency'][0][0]

            if self.target_f is not None:
                factor = self.target_f/self.emg_freq
                for key in self.dict_emg.keys():
                    self.dict_emg[key] = resample(self.dict_emg[key], int(len(self.dict_emg[key])*factor))
                self.emg_freq = self.target_f
            return self.dict_emg, self.emg_freq
        return self.dict_emg, self.emg_freq


def parallel_proc(set_class: type, dir_path='.'):
    inp = get_file_pairs(dir_path)
    os.makedirs(os.path.join(dir_path, 'Data'), exist_ok=True)
    w = partial(worker, set_class=set_class, save_dir_path=os.path.join(dir_path, 'Data'))
    import time
    t1 = time.perf_counter()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(w, inp)
    print("Elapsed time: {}".format(time.perf_counter() - t1))


if __name__ == '__main__':
    np.seterr(all='warn')
    parallel_proc(ExtendSet, sys.argv[1])
