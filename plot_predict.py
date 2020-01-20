import matplotlib.pyplot as plt
import seaborn as sns
import json
from keras.models import load_model
from convmemnet import stack_emg, norm_emg
import numpy as np

file_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data\004_Validation20.json'
sns.set()
sns.set_context('paper')
with open(file_path) as json_file:
    dict_data = json.load(json_file)

emg_data = norm_emg(np.array(dict_data["EMG"]))

ws=40
s=1

emg_data = stack_emg(emg_data, window_size=ws, stride=s)
#emg_data = emg_data[:,:,:-1]
model = load_model('env.h5')

y = model.predict(emg_data)

fig, axes = plt.subplots(3, 2)
axes = axes.flatten()

y = model.predict(emg_data)
N = 20
def f(y_in):
    return np.convolve(y_in, np.ones((N,))/N, mode='valid')
y = np.apply_along_axis(f, 0, y)

joint_names = ['LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
Y0 = [[] for _ in range(len(joint_names))]
for i, joint in enumerate(joint_names):
    cur_data = dict_data[joint]
    upsampled = np.array(cur_data)
    xp = np.arange(len(upsampled))*20/s
    x = np.arange(ws, len(upsampled)*20+s)
    x = x[0::s]
    def interp_f(data):
        return np.interp(x, xp, data)
    upsampled = np.apply_along_axis(interp_f, 0, upsampled)
    Y0[i].extend(upsampled.tolist())

Y0 = np.array(Y0)
Y0 = Y0[:, :, 0].transpose()

for i, joint in enumerate(joint_names):
    if "LKnee" not in joint:
        continue
    #cur_data = [row[0] for row in dict_data[joint][1:]]
    axes[i].plot(Y0[:,i])
    axes[i].plot(np.arange(N, len(y[:, 0])+N), y[:, 0])
    axes[i].set_xlabel("Time Sample (0.5 ms)")
    axes[i].set_ylabel("Radians")
    axes[i].set_title(joint)

plt.tight_layout()
plt.show()