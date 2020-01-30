import matplotlib.pyplot as plt
import seaborn as sns
import json
from keras.models import load_model
from convmemnet import stack_emg, norm_emg
import numpy as np
import tensorflow as tf
import keras

file_path = r'C:\Users\win10\Desktop\Projects\CYB\Experiment_Balint\CYB004\Data\004_Validation20.json'
sns.set()
sns.set_context('paper')
with open(file_path) as json_file:
    dict_data = json.load(json_file)

emg_data = norm_emg(np.array(dict_data["EMG"]))
ecg = emg_data[7]

ws=60
s=1

emg_data = stack_emg(emg_data, window_size=ws, stride=s)
#emg_data = emg_data[:,:,:-1]
#emg_data = np.expand_dims(emg_data, 1)
model = load_model('w60_s1.h5')
model.summary()
y = model.predict(emg_data)

fig, axes = plt.subplots(3, 2)
axes = axes.flatten()

# import time
# t1 = time.perf_counter()
# for i in range(2):
#     y = model.predict(emg_data)
# print(time.perf_counter()-t1)
#
# emg_data = np.expand_dims(emg_data, 1)
# t1 = time.perf_counter()
# for cur_data in emg_data:
#     y = model.predict(cur_data, batch_size=1)
# print(time.perf_counter()-t1)
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

ecg = ecg[0:len(Y0)]

for i, joint in enumerate(joint_names):
    # if "LKnee" not in joint:
    #     continue
    #cur_data = [row[0] for row in dict_data[joint][1:]]
    e, = axes[i].plot(np.arange(len(Y0[:, i]))/2000,
                     (ecg-np.mean(ecg))/np.std(ecg)/5 * np.std(Y0[:, i]) + np.mean(Y0[:, i]) + 0.2,
                     alpha=0.25, color='gray', lw=0.7)
    o, = axes[i].plot(np.arange(len(Y0[:, i]))/2000, Y0[:, i])
    est, = axes[i].plot(np.arange(N, len(y[:, i])+N)/2000, y[:, i])
    if i>3 :
        axes[i].set_xlabel("Time (s)")
    else:
        axes[i].set_xticklabels([])
    if i ==1:
        plt.legend((o, est, e), ("Actual Angles", "Predicted Angles", "ECG signal (A.U.)"), bbox_to_anchor=(1.04, 0.5),
                   loc="center left", borderaxespad=0)
    axes[i].set_ylabel("Radians")
    axes[i].set_title(joint)
    axes[i].locator_params(axis='y', nbins=4)


# box = fig.get_position()
# fig.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.tight_layout()
plt.subplots_adjust(right=0.85)
#fig.suptitle("Prediction with separated channels", x=0.5, size=22)

plt.show()
