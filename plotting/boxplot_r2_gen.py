import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sub004 = {

    'Gen': [[0.64296317, 0.65612702, 0.65881451, 0.6712479, 0.56158725, 0.53688033],
            [0.73132138, 0.71918397, 0.76239377, 0.76892186, 0.54202123, 0.56847204],
            [0.7819106, 0.82419151, 0.7382092, 0.86037445, 0.57634928, 0.62991616],
            [0.88054713, 0.87330535, 0.93163479, 0.93904487, 0.78033511, 0.80023925],
            [0.94113822, 0.92456577, 0.94629328, 0.95478431, 0.73919948, 0.75794637]]

}
sub005 = {

    'Gen': [[0.66765921, 0.59395168, 0.56501943, 0.54771185, 0.11569125, -0.09561904],
            [0.63057574, 0.55482087, 0.59810012, 0.56939872, 0.12931674, -0.25176012],
            [0.76426762, 0.63199402, 0.74540176, 0.67012474, 0.46176834, 0.40307208],
            [0.8804782, 0.87328165, 0.90137793, 0.89902827, 0.49798272, 0.58458052],
            [0.86509529, 0.8714404, 0.90778011, 0.8762201, 0.45968218, 0.96883123]]

}
sub101 = {

    'Gen': [[0.6101628, 0.54644081, 0.65533253, 0.69010706, 0.45406872, 0.38296436],
            [0.63939908, 0.63607511, 0.67665872, 0.70857694, 0.40855275, 0.39246102],
            [0.8068815, 0.81514401, 0.88947379, 0.88947521, 0.41104267, 0.42072681],
            [0.79989791, 0.83792631, 0.86266511, 0.87954993, 0.45015558, 0.42255214],
            [0.63983064, 0.63948741, 0.64343862, 0.73590214, 0.41284975, 0.40152274]]

}
sub102 = {

    'Gen': [[0.71753825, 0.74691427, 0.64409407, 0.69178329, 0.22817841, 0.24796641],
            [0.885768, 0.91361834, 0.9013929, 0.90140416, 0.36496352, 0.35449121],
            [0.8579613, 0.88982665, 0.84039202, 0.86493666, 0.32774263, 0.25790619],
            [0.88496176, 0.90975079, 0.88021867, 0.89717714, 0.34648457, 0.26521079],
            [0.90977107, 0.91795542, 0.90142038, 0.89265223, 0.30550149, 0.30356944]]

}

taskdict = {

    'Gen': [],

}


def p(a):
    a = np.array(a)
    b = np.zeros((a.shape[0], 3))
    for it in range(3):
        b[:, it] = (a[:, it * 2] + a[:, it * 2 + 1]) / 2
    return b


ye = []
for i, s in enumerate([sub004, sub005, sub101, sub102]):
    for key in s.keys():
        taskdict[key].append(np.mean(s[key]))
for key in taskdict.keys():
    taskdict[key] = np.vstack(taskdict[key])
for key in taskdict.keys():
    print(np.mean(taskdict[key][~np.isnan(taskdict[key][:, 0]) * ~np.isnan(taskdict[key][:, 2]), :], axis=0))
    ye.append(taskdict[key][~np.isnan(taskdict[key][:, 0]) * ~np.isnan(taskdict[key][:, 2]), :])
print('\n')
print(np.mean(np.vstack(ye), axis=0))
sns.set_style('darkgrid')
sns.set_context('talk')
fig, ax = plt.subplots(1, 5, figsize=(16, 4.5), sharey='all')

for i, s in enumerate([sub003, sub004, sub005, sub101, sub102]):
    sub = np.array([p(cur_task) for cur_task in s.values()]).T
    sub = sub.reshape((-1, 5)).T.flatten()
    dfs = pd.DataFrame(sub, columns=['$R^2$'])
    dfs['Angle'] = np.tile(np.repeat(['Hip', 'Knee', 'Ankle'], 5), 5)
    dfs['task'] = np.repeat(['Walk', 'Stand up', 'Sit Down', 'Ascend', 'Descend'], 15)
    print(dfs)
    ch = sns.barplot(ax=ax[i], data=dfs, y='$R^2$', x='task', hue='Angle')
    ch.set_xticklabels(ch.get_xticklabels(), rotation=60)
    ch.set_title('Subject ' + str(i + 1))
    ax[i].set_xlabel('')
    ax[i].set_ylim([0.2, 1.05])
    ax[i].get_legend().remove()
    if i == 0:
        ch.set_ylabel('Validation $R^2$')
        ax[i].legend()
    else:
        ax[i].set_ylabel('')

plt.tight_layout()
plt.show()
