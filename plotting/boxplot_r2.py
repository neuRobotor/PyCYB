import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sub003 = {

    'Walk': [
        [np.nan, -4.82255145e-04, -1.31513680e-04, -2.15042652e-05, -8.77264044e-02, -2.98141390e-02],
        [-np.nan, -4.49893211e-05, -9.43560317e-03, -5.20743638e-03, -6.35724532e-02, -2.09280331e-02],
        [-1.03615526e-02, -4.36552729e-04, -1.14796676e-03, -2.04844572e-05, -5.70390694e-04, -1.09466100e-03],
        [-6.61050642e-04, -4.16467897e-03, -1.89652102e-04, -4.21991363e-03, -3.49962009e-03, -9.13994469e-03],
        [-4.37699276e-04, -1.05444277e-04, -1.03645139e-04, -1.09655298e-04, -1.58949418e-04,
         -3.27102601e-03]],

    'UpSit': [[np.nan] * 6] * 5,

    'DownSit': [[np.nan] * 6] * 5,

    'UpStair': [[0.61209944, 0.67737202, 0.7370193, 0.77173734, 0.54054953, 0.53642014],
                [0.90979258, 0.94557931, 0.92899891, 0.93252407, 0.62086144, 0.68395681],
                [0.92689601, 0.92223567, 0.92908694, 0.95189351, 0.64529477, 0.58809498],
                [0.90881992, 0.91481714, 0.93517303, 0.91975532, 0.65165601, 0.5694768, ],
                [0.84174847, 0.87182005, 0.89278289, 0.86167207, 0.58343057, 0.67027065]],

    'DownStair': [[0.74256679, 0.72859827, 0.91351468, 0.89551286, 0.87531464, 0.83884831],
                  [0.83545439, 0.8039049, 0.95100523, 0.97012861, 0.90054708, 0.90940302],
                  [0.83957132, 0.82313331, 0.95089508, 0.96880328, 0.92500476, 0.91482343],
                  [0.80393783, 0.82956283, 0.9698037, 0.95363287, 0.93110485, 0.89858647],
                  [0.8444385, 0.80848721, 0.95588378, 0.95908917, 0.91443073, 0.93420351]],

}

sub004 = {

    'Walk': [[0.89081059, 0.82411046, 0.88750994, 0.84109312, 0.74136463, 0.62835599],
             [0.91427358, 0.93048749, 0.95229634, 0.93906215, 0.80459995, 0.75557616],
             [0.88077443, 0.81435565, 0.92933012, 0.89912092, 0.69491472, 0.71079491],
             [0.92548472, 0.9321069, 0.95769004, 0.94876088, 0.7908415, 0.73865044],
             [0.91144584, 0.93527839, 0.95735671, 0.96211434, 0.75930553, 0.75670369]],

    'UpSit': [[0.96991381, 0.97273448, 0.96792806, 0.96502772, 0.80272681, 0.65504492],
              [0.96697595, 0.96321635, 0.99356092, 0.99165617, 0.87203028, 0.80367579],
              [0.99480498, 0.99130353, 0.99777677, 0.99346687, 0.97176635, 0.951412, ],
              [0.98488251, 0.98308393, 0.99569263, 0.99476966, 0.38846276, 0.3726372, ],
              [0.98401239, 0.98830754, 0.99589438, 0.99683799, 0.76634808, 0.75752537]],

    'DownSit': [[0.81906212, 0.83024866, 0.80441822, 0.8295009, 0.60335411, 0.50937544],
                [0.95660538, 0.95330227, 0.9762664, 0.98407925, 0.77678276, 0.63785756],
                [0.98703586, 0.98117273, 0.99449891, 0.99427431, 0.61093743, 0.60422325],
                [0.98525619, 0.98304877, 0.98218407, 0.98384289, 0.23963949, 0.63044547],
                [0.9872424, 0.98943525, 0.99635222, 0.99517909, 0.74678466, 0.77234301]],

    'UpStair': [[0.81821799, 0.91407607, 0.70637654, 0.91250922, 0.68169828, 0.77242842],
                [0.86911171, 0.81863739, 0.85768172, 0.83099637, 0.70475999, 0.76144893],
                [0.8809542, 0.90702589, 0.90743649, 0.92588333, 0.69200602, 0.75839059],
                [0.93849748, 0.94937004, 0.92728763, 0.93383483, 0.71368992, 0.74979835],
                [0.94755263, 0.94721064, 0.95281716, 0.94940385, 0.6931401, 0.80821436]],

    'DownStair': [[0.55230242, 0.61323067, 0.71987588, 0.79971627, 0.62568096, 0.56241436],
                  [0.85975744, 0.81259896, 0.95547607, 0.95893924, 0.89981313, 0.89322016],
                  [0.76704487, 0.77807255, 0.87349926, 0.85866081, 0.7649306, 0.81386576],
                  [0.75213045, 0.83794025, 0.85824953, 0.86054464, 0.77000532, 0.81756773],
                  [0.7946365, 0.83507272, 0.94637767, 0.95607816, 0.90774747, 0.93635369]],

}

sub005 = {

    'Walk': [[0.89259499, 0.81896795, 0.89123924, 0.87776512, 0.62845706, 0.6772764, ],
             [0.93014757, 0.91344589, 0.93268592, 0.95115702, 0.69957635, 0.72807145],
             [0.89748102, 0.90666638, 0.91544596, 0.92453974, 0.62070675, 0.70422877],
             [0.93068635, 0.91289383, 0.94268171, 0.93899044, 0.68163121, 0.69459348],
             [0.94621434, 0.92665974, 0.92420784, 0.94644072, 0.65608485, 0.73920429]],

    'UpSit': [[0.89554442, 0.88545475, 0.91076135, 0.9570268, -2.61824726, 0.64623104],
              [0.99186974, 0.9757221, 0.98887481, 0.99013448, 0.88061958, 0.93958502],
              [0.94467644, 0.97041464, 0.98296015, 0.99308634, 0.50544935, 0.98408501],
              [0.95475458, 0.96438977, 0.94123297, 0.96900776, -13.27197966, 0.72699909],
              [0.97727675, 0.97941691, 0.93434204, 0.93045606, 0.6483886, 0.8852203, ]],

    'DownSit': [[0.88805676, 0.88264058, 0.80261326, 0.79762593, 0.71278326, 0.750499, ],
                [0.79877803, 0.79824864, 0.94570202, 0.94256476, 0.34027996, 0.80525841],
                [0.97198808, 0.98099212, 0.99379389, 0.99302541, 0.61803482, 0.33815972],
                [0.98247956, 0.97733465, 0.99375529, 0.99442353, 0.96237744, 0.93774802],
                [0.82540919, 0.83726342, 0.98107044, 0.97582742, 0.22325552, -0.09930413]],

    'UpStair': [[0.88625798, 0.69733894, 0.85666597, 0.63897247, 0.68040522, 0.70651393],
                [0.76981578, 0.77417287, 0.7449929, 0.78570811, 0.5025896, 0.56368978],
                [0.85847783, 0.84370096, 0.84026938, 0.89135988, 0.74777488, 0.73798737],
                [0.93290441, 0.92916917, 0.94745909, 0.94137471, 0.72216212, 0.75856999],
                [0.92501304, 0.92568391, 0.91843776, 0.91308991, 0.69410951, 0.70308754]],

    'DownStair': [[0.48323371, 0.4290343, 0.71102863, 0.61549584, 0.64018979, 0.57195324],
                  [0.74010077, 0.72396124, 0.83557385, 0.85300326, 0.81805792, 0.7984615, ],
                  [0.37838036, 0.47003694, 0.62624521, 0.72380366, 0.75992019, 0.77866697],
                  [0.52861531, 0.63453413, 0.69644519, 0.83259754, 0.77623947, 0.7673616, ],
                  [0.88144615, 0.75050947, 0.94956443, 0.93129881, 0.89285435, 0.91803538]],

}

sub101 = {

    'Walk': [[0.61580971, 0.62553782, 0.73712404, 0.76648548, 0.50065332, 0.55044165],
             [0.65628428, 0.67117399, 0.78888721, 0.78850769, 0.46939325, 0.59558249],
             [0.73733345, 0.75783979, 0.7825132, 0.85697466, 0.42558618, 0.52588203],
             [0.79108192, 0.82175941, 0.89297341, 0.89981464, 0.6316818, 0.68953382],
             [0.85625939, 0.85603828, 0.91502725, 0.87403701, 0.62548438, 0.62617955]],

    'UpSit': [[0.96189822, 0.96496687, 0.97652292, 0.9700618, 0.78220621, 0.66532233],
              [0.97866767, 0.98072973, 0.98180525, 0.9814143, 0.58038074, 0.5417866, ],
              [0.98258035, 0.98166298, 0.99463939, 0.99471574, 0.79362699, 0.71703207],
              [0.98257271, 0.98053675, 0.99625212, 0.99115044, 0.84584603, 0.64177581],
              [0.98492227, 0.98435216, 0.99244002, 0.99156623, 0.73801916, 0.7831312, ]],

    'DownSit': [[0.86201815, 0.86518272, 0.91733559, 0.9189171, 0.62197969, 0.63726868],
                [0.93076974, 0.93223261, 0.9847633, 0.98432479, 0.41336794, 0.55659214],
                [0.83394308, 0.83466023, 0.91892687, 0.92025902, 0.59815517, 0.71974071],
                [0.95325537, 0.95603181, 0.97968727, 0.98197697, 0.6620182, 0.87143677],
                [0.93254531, 0.9403816, 0.96779768, 0.97306451, 0.45929616, 0.71029355]],

    'UpStair': [[0.67395047, 0.75383207, 0.67083911, 0.83209944, 0.52534207, 0.45375666],
                [0.83947871, 0.75293289, 0.83338271, 0.80356214, 0.69531989, 0.54340689],
                [0.89440083, 0.94670366, 0.92359232, 0.93386586, 0.82014234, 0.7540335, ],
                [0.84782526, 0.9078841, 0.89898482, 0.91259303, 0.7521148, 0.66937417],
                [0.86993232, 0.87579698, 0.86182523, 0.90908339, 0.77498436, 0.63143197]],

    'DownStair': [[0.5164833, 0.51636428, 0.6887909, 0.71544445, 0.59551376, 0.55150037],
                  [0.53977433, 0.51057675, 0.6754149, 0.74021137, 0.56059868, 0.51266907],
                  [0.70610509, 0.70755216, 0.86255879, 0.90627708, 0.76652798, 0.74799864],
                  [0.59230317, 0.59448897, 0.70383639, 0.76993163, 0.64876783, 0.60017849],
                  [0.78382783, 0.70018913, 0.91372542, 0.89686635, 0.82567078, 0.76813492]],

}

sub102 = {

    'Walk': [[0.83435307, 0.83545629, 0.78901412, 0.81329655, 0.51399438, 0.50682275],
             [0.83282939, 0.86836194, 0.79323359, 0.83239414, 0.52497724, 0.58668597],
             [0.84011374, 0.86716823, 0.80461838, 0.83338703, 0.51484167, 0.52848505],
             [0.83947713, 0.8827332, 0.85567095, 0.80149755, 0.50143284, 0.45564649],
             [0.86707992, 0.87081548, 0.87592746, 0.80640939, 0.6212737, 0.53696962]],

    'UpSit': [[0.90142981, 0.90195556, 0.89480245, 0.89051523, 0.79986524, 0.76975324],
              [0.95937133, 0.95992928, 0.98850865, 0.98648492, 0.89745083, 0.88268079],
              [0.95982522, 0.95912152, 0.981687, 0.9804008, 0.90225478, 0.8567832, ],
              [0.97330408, 0.97260278, 0.98627526, 0.98615367, 0.75190638, 0.73735576],
              [0.96113391, 0.9623444, 0.97310524, 0.97548583, 0.8605728, 0.90554596]],

    'DownSit': [[0.84097808, 0.83925271, 0.84604614, 0.84818239, 0.59899289, 0.59591703],
                [0.94532726, 0.94208519, 0.97262421, 0.96831762, 0.79561373, 0.73878953],
                [0.96784774, 0.96780834, 0.98099686, 0.98477692, 0.72975907, 0.60402299],
                [0.96537191, 0.96681957, 0.98704662, 0.98578728, 0.77280398, 0.71467608],
                [0.9701087, 0.97220188, 0.99186594, 0.99347398, 0.86630127, 0.88680545]],

    'UpStair': [[0.71478169, 0.78058888, 0.73607318, 0.7648593, 0.53229821, 0.62050437],
                [0.75014911, 0.81065873, 0.76995441, 0.8103036, 0.5816305, 0.61618828],
                [0.75404123, 0.86605594, 0.83104348, 0.83602275, 0.67174833, 0.75752018],
                [0.94646748, 0.94039334, 0.95573731, 0.94132392, 0.80781991, 0.77816295],
                [0.92758771, 0.94351988, 0.93750839, 0.95373837, 0.80451706, 0.77090662]],

    'DownStair': [[0.4059226, 0.37091703, 0.58113237, 0.45189565, 0.51555883, 0.54806986],
                  [0.84408035, 0.78487346, 0.91692516, 0.90375705, 0.88282831, 0.84713051],
                  [0.82588367, 0.78435916, 0.87710522, 0.92256086, 0.84725129, 0.82826492],
                  [0.87603475, 0.80741299, 0.92168909, 0.91583327, 0.83701601, 0.83091062],
                  [0.85204233, 0.77524621, 0.93360165, 0.88235758, 0.85902819, 0.81872438]],

}

taskdict = {

    'Walk': [],

    'UpSit': [],

    'DownSit': [],

    'UpStair': [],

    'DownStair': []

}


def p(a):
    a = np.array(a)
    b = np.zeros((a.shape[0], 3))
    for it in range(3):
        b[:, it] = (a[:, it * 2] + a[:, it * 2 + 1]) / 2
    return b


ye = []
for i, s in enumerate([sub003, sub004, sub005, sub101, sub102]):
    for key in s.keys():
        taskdict[key].append(np.mean(p(s[key]), axis=0))
for key in taskdict.keys():
    taskdict[key] = np.vstack(taskdict[key])
for key in taskdict.keys():
    print(np.mean(taskdict[key][~np.isnan(taskdict[key][:, 0]) * ~np.isnan(taskdict[key][:, 2]), :], axis=0))
    ye.append(np.mean(taskdict[key][~np.isnan(taskdict[key][:, 0]) * ~np.isnan(taskdict[key][:, 2]), :], axis=0))
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