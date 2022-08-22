import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

epsilons1 = [0.01 * x for x in range(2000)]
epsilons1[0] = 0.0001

epsilons2 = [0.01 * x for x in range(2000)]
epsilons2[0] = 0.0001

epsilons3 = [0.01 * x for x in range(2000)]
epsilons3[0] = 0.0001

epsilons4 = [0.01 * x for x in range(2000)]
epsilons4[0] = 0.0001

epsilons5 = [0.0102 * x for x in range(2000)]
epsilons5[0] = 0.0001

epsilons6 = [0.0105 * x for x in range(2000)]
epsilons6[0] = 0.0001

epsilons7 = [0.011 * x for x in range(2000)]
epsilons7[0] = 0.0001

epsilons8 = [0.015 * x for x in range(2000)]
epsilons8[0] = 0.0001

epsilons9 = [0.02 * x for x in range(2000)]
epsilons9[0] = 0.0001

epsilons10 = [0.1 * x for x in range(250)]
epsilons10[0] = 0.0001

names = ['wha_prob_a', 'wha_prob_b', 'wha_equal_opportunity', 'wha_accuracy_in_all_iterations', 'wha_accuracy_in_all',
         'wha_a_accuracy_in_iterations', 'wha_b_accuracy_in_iterations']

wha_file_name_1_105 = ['wha_1_105_prob_a', 'wha_1_105_prob_b', 'wha_1_105_equal_opportunity',
                       'wha_1_105_accuracy_in_all',
                       'wha_1_105_accuracy_in_all_iterations', 'wha_1_105_accuracy_in_a_iterations',
                       'wha_1_105_accuracy_in_b_iterations']

wha_file_name_1_102 = ['wha_1_102_prob_a', 'wha_1_102_prob_b', 'wha_1_102_equal_opportunity',
                       'wha_1_102_accuracy_in_all',
                       'wha_1_102_accuracy_in_all_iterations', 'wha_1_102_accuracy_in_a_iterations',
                       'wha_1_102_accuracy_in_b_iterations']

wha_file_name_1_1 = ['wha_1_1_prob_a', 'wha_1_1_prob_b', 'wha_1_1_equal_opportunity', 'wha_1_1_accuracy_in_all',
                     'wha_1_1_accuracy_in_all_iterations', 'wha_1_1_accuracy_in_a_iterations',
                     'wha_1_1_accuracy_in_b_iterations']

wha_file_name_102_1 = ['wha_102_1_prob_a', 'wha_102_1_prob_b', 'wha_102_1_equal_opportunity',
                       'wha_102_1_accuracy_in_all',
                       'wha_102_1_accuracy_in_all_iterations', 'wha_102_1_accuracy_in_a_iterations',
                       'wha_102_1_accuracy_in_b_iterations']

wha_file_name_105_1 = ['wha_105_1_prob_a', 'wha_105_1_prob_b', 'wha_105_1_equal_opportunity',
                       'wha_105_1_accuracy_in_all',
                       'wha_105_1_accuracy_in_all_iterations', 'wha_105_1_accuracy_in_a_iterations',
                       'wha_105_1_accuracy_in_b_iterations']

wha_file_name_11_1 = ['wha_11_1_prob_a', 'wha_11_1_prob_b', 'wha_11_1_equal_opportunity', 'wha_11_1_accuracy_in_all',
                      'wha_11_1_accuracy_in_all_iterations', 'wha_11_1_accuracy_in_a_iterations',
                      'wha_11_1_accuracy_in_b_iterations']

wha_file_name_15_1 = ['wha_15_1_prob_a', 'wha_15_1_prob_b', 'wha_15_1_equal_opportunity', 'wha_15_1_accuracy_in_all',
                      'wha_15_1_accuracy_in_all_iterations', 'wha_15_1_accuracy_in_a_iterations',
                      'wha_15_1_accuracy_in_b_iterations']

wha_file_name_2_1 = ['wha_2_1_prob_a', 'wha_2_1_prob_b', 'wha_2_1_equal_opportunity', 'wha_2_1_accuracy_in_all',
                     'wha_2_1_accuracy_in_all_iterations', 'wha_2_1_accuracy_in_a_iterations',
                     'wha_2_1_accuracy_in_b_iterations']

same_noise_wha_file_name = ['wha_same_prob_a', 'wha_same_prob_b', 'wha_same_equal_opportunity',
                            'wha_same_accuracy_in_all',
                            'wha_same_accuracy_in_all_iterations', 'wha_same_accuracy_in_a_iterations',
                            'wha_same_accuracy_in_b_iterations']

debiased = ['wha_prob_a_debiased_multiple_selection', 'wha_prob_b_debiased_multiple_selection',
                            'wha_equal_opportunity_debiased_multiple_selection',
                            'wha_accuracy_debiased_multiple_selection',
                            'wha_accuracy_in_all_iterations_debiased_multiple_selection',
                            'wha_accuracy_in_a_iterations_debiased_multiple_selection',
                            'wha_accuracy_in_b_iterations_debiased_multiple_selection']

expeiments = [same_noise_wha_file_name, wha_file_name_1_105, wha_file_name_1_102, wha_file_name_1_1, wha_file_name_102_1
    , wha_file_name_105_1, wha_file_name_11_1, wha_file_name_15_1, wha_file_name_2_1, debiased]
proportions = [['same'], [1, 1.05], [1, 1.02], [1, 1], [1.02, 1], [1.05, 1], [1.1, 1], [1.5, 1], [2, 1]]
for name_index in range(len(wha_file_name_1_1)):

    data1_1 = pd.read_csv(str(expeiments[0][name_index]) + '_1.csv')
    data1_2 = pd.read_csv(str(expeiments[0][name_index]) + '_2.csv')
    data1_3 = pd.read_csv(str(expeiments[0][name_index]) + '_3.csv')
    data1_4 = pd.read_csv(str(expeiments[0][name_index]) + '_4.csv')
    data2_1 = pd.read_csv(str(expeiments[1][name_index]) + '_1.csv')
    data2_2 = pd.read_csv(str(expeiments[1][name_index]) + '_2.csv')
    data2_3 = pd.read_csv(str(expeiments[1][name_index]) + '_3.csv')
    data2_4 = pd.read_csv(str(expeiments[1][name_index]) + '_4.csv')
    data3_1 = pd.read_csv(str(expeiments[2][name_index]) + '_1.csv')
    data3_2 = pd.read_csv(str(expeiments[2][name_index]) + '_2.csv')
    data3_3 = pd.read_csv(str(expeiments[2][name_index]) + '_3.csv')
    data3_4 = pd.read_csv(str(expeiments[2][name_index]) + '_4.csv')
    data4_1 = pd.read_csv(str(expeiments[3][name_index]) + '_1.csv')
    data4_2 = pd.read_csv(str(expeiments[3][name_index]) + '_2.csv')
    data4_3 = pd.read_csv(str(expeiments[3][name_index]) + '_3.csv')
    data4_4 = pd.read_csv(str(expeiments[3][name_index]) + '_4.csv')
    data5_1 = pd.read_csv(str(expeiments[4][name_index]) + '_1.csv')
    data5_2 = pd.read_csv(str(expeiments[4][name_index]) + '_2.csv')
    data5_3 = pd.read_csv(str(expeiments[4][name_index]) + '_3.csv')
    data5_4 = pd.read_csv(str(expeiments[4][name_index]) + '_4.csv')
    data6_1 = pd.read_csv(str(expeiments[5][name_index]) + '_1.csv')
    data6_2 = pd.read_csv(str(expeiments[5][name_index]) + '_2.csv')
    data6_3 = pd.read_csv(str(expeiments[5][name_index]) + '_3.csv')
    data6_4 = pd.read_csv(str(expeiments[5][name_index]) + '_4.csv')
    data7_1 = pd.read_csv(str(expeiments[6][name_index]) + '_1.csv')
    data7_2 = pd.read_csv(str(expeiments[6][name_index]) + '_2.csv')
    data7_3 = pd.read_csv(str(expeiments[6][name_index]) + '_3.csv')
    data7_4 = pd.read_csv(str(expeiments[6][name_index]) + '_4.csv')
    data8_1 = pd.read_csv(str(expeiments[7][name_index]) + '_1.csv')
    data8_2 = pd.read_csv(str(expeiments[7][name_index]) + '_2.csv')
    data8_3 = pd.read_csv(str(expeiments[7][name_index]) + '_3.csv')
    data8_4 = pd.read_csv(str(expeiments[7][name_index]) + '_4.csv')
    data9_1 = pd.read_csv(str(expeiments[8][name_index]) + '_1.csv')
    data9_2 = pd.read_csv(str(expeiments[8][name_index]) + '_2.csv')
    data9_3 = pd.read_csv(str(expeiments[8][name_index]) + '_3.csv')
    data9_4 = pd.read_csv(str(expeiments[8][name_index]) + '_4.csv')

    data10_1 = pd.read_csv('results/'+str(expeiments[9][name_index]) + '_1.csv')
    data10_2 = pd.read_csv('results/'+str(expeiments[9][name_index]) + '_2.csv')
    data10_3 = pd.read_csv('results/'+str(expeiments[9][name_index]) + '_3.csv')
    data10_4 = pd.read_csv('results/'+str(expeiments[9][name_index]) + '_4.csv')

    data1_1 = data1_1.drop(columns='Unnamed: 0')
    data2_1 = data2_1.drop(columns='Unnamed: 0')
    data3_1 = data3_1.drop(columns='Unnamed: 0')
    data4_1 = data4_1.drop(columns='Unnamed: 0')
    data5_1 = data5_1.drop(columns='Unnamed: 0')
    data6_1 = data6_1.drop(columns='Unnamed: 0')
    data7_1 = data7_1.drop(columns='Unnamed: 0')
    data8_1 = data8_1.drop(columns='Unnamed: 0')
    data9_1 = data9_1.drop(columns='Unnamed: 0')
    data10_1 = data10_1.drop(columns='Unnamed: 0')

    data1_2 = data1_2.drop(columns='Unnamed: 0')
    data2_2 = data2_2.drop(columns='Unnamed: 0')
    data3_2 = data3_2.drop(columns='Unnamed: 0')
    data4_2 = data4_2.drop(columns='Unnamed: 0')
    data5_2 = data5_2.drop(columns='Unnamed: 0')
    data6_2 = data6_2.drop(columns='Unnamed: 0')
    data7_2 = data7_2.drop(columns='Unnamed: 0')
    data8_2 = data8_2.drop(columns='Unnamed: 0')
    data9_2 = data9_2.drop(columns='Unnamed: 0')
    data10_2 = data10_2.drop(columns='Unnamed: 0')

    data1_3 = data1_3.drop(columns='Unnamed: 0')
    data2_3 = data2_3.drop(columns='Unnamed: 0')
    data3_3 = data3_3.drop(columns='Unnamed: 0')
    data4_3 = data4_3.drop(columns='Unnamed: 0')
    data5_3 = data5_3.drop(columns='Unnamed: 0')
    data6_3 = data6_3.drop(columns='Unnamed: 0')
    data7_3 = data7_3.drop(columns='Unnamed: 0')
    data8_3 = data8_3.drop(columns='Unnamed: 0')
    data9_3 = data9_3.drop(columns='Unnamed: 0')
    data10_3 = data10_3.drop(columns='Unnamed: 0')

    data1_4 = data1_4.drop(columns='Unnamed: 0')
    data2_4 = data2_4.drop(columns='Unnamed: 0')
    data3_4 = data3_4.drop(columns='Unnamed: 0')
    data4_4 = data4_4.drop(columns='Unnamed: 0')
    data5_4 = data5_4.drop(columns='Unnamed: 0')
    data6_4 = data6_4.drop(columns='Unnamed: 0')
    data7_4 = data7_4.drop(columns='Unnamed: 0')
    data8_4 = data8_4.drop(columns='Unnamed: 0')
    data9_4 = data9_4.drop(columns='Unnamed: 0')
    data10_4 = data10_4.drop(columns='Unnamed: 0')
    datas = [data1_1, data2_1, data3_1, data4_1, data5_1, data6_1, data7_1, data8_1, data9_1, data10_1,
             data1_2, data2_2, data3_2, data4_2, data5_2, data6_2, data7_2, data8_2, data9_2, data10_2,
             data1_3, data2_3, data3_3, data4_3, data5_3, data6_3, data7_3, data8_3, data9_3, data10_3,
             data1_4, data2_4, data3_4, data4_4, data5_4, data6_4, data7_4, data8_4, data9_4, data10_4]

    for data_index in range(len(datas)):
        data11 = np.array(datas[data_index])
        datas[data_index] = []
        for each in data11:
            datas[data_index].append(each[0])

    data1_1 = datas[0]
    data2_1 = datas[1]
    data3_1 = datas[2]
    data4_1 = datas[3]
    data5_1 = datas[4]
    data6_1 = datas[5]
    data7_1 = datas[6]
    data8_1 = datas[7]
    data9_1 = datas[8]
    data10_1 = datas[9]

    data1_2 = datas[10]
    data2_2 = datas[11]
    data3_2 = datas[12]
    data4_2 = datas[13]
    data5_2 = datas[14]
    data6_2 = datas[15]
    data7_2 = datas[16]
    data8_2 = datas[17]
    data9_2 = datas[18]
    data10_2 = datas[19]

    data1_3 = datas[20]
    data2_3 = datas[21]
    data3_3 = datas[22]
    data4_3 = datas[23]
    data5_3 = datas[24]
    data6_3 = datas[25]
    data7_3 = datas[26]
    data8_3 = datas[27]
    data9_3 = datas[28]
    data10_3 = datas[29]

    data1_4 = datas[30]
    data2_4 = datas[31]
    data3_4 = datas[32]
    data4_4 = datas[33]
    data5_4 = datas[34]
    data6_4 = datas[35]
    data7_4 = datas[36]
    data8_4 = datas[37]
    data9_4 = datas[38]
    data10_4 = datas[39]

    data5_1 = data5_1[:1961]
    data5_2 = data5_2[:1961]
    data5_3 = data5_3[:1961]
    data5_4 = data5_4[:1961]
    epsilons5 = epsilons5[:1961]

    data6_1 = data6_1[:1905]
    data6_2 = data6_2[:1905]
    data6_3 = data6_3[:1905]
    data6_4 = data6_4[:1905]
    epsilons6 = epsilons6[:1905]

    data7_1 = data7_1[:1819]
    data7_2 = data7_2[:1819]
    data7_3 = data7_3[:1819]
    data7_4 = data7_4[:1819]
    epsilons7 = epsilons7[:1819]

    data8_1 = data8_1[:1334]
    data8_2 = data8_2[:1334]
    data8_3 = data8_3[:1334]
    data8_4 = data8_4[:1334]
    epsilons8 = epsilons8[:1334]

    data9_1 = data9_1[:1000]
    data9_2 = data9_2[:1000]
    data9_3 = data9_3[:1000]
    data9_4 = data9_4[:1000]
    epsilons9 = epsilons9[:1000]

    data10_1 = data10_1[:201]
    data10_2 = data10_2[:201]
    data10_3 = data10_3[:201]
    data10_4 = data10_4[:201]
    epsilons10 = epsilons10[:201]

    # data1 = data1[:1000]
    # data2 = data2[:1000]
    # epsilons1 = epsilons1[:1000]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 8))
    # plt.xticks(fontsize=4)
    plt.rcParams["font.family"] = "Times New Roman"

    if name_index == 0:
        # fig.suptitle(r'Proportion of True A candidate selected from qualified A population as a function of $\epsilon_{1}$', size=15)
        # ax1.set_ylabel('Proportion of True A candidate selected from qualified A population', size=8)
        fig.suptitle(
            r'Proportion of qualified A candidate selection to the qualified A population as a function of $\epsilon_{1}$',
            size=10)

        ax1.set_ylabel('Proportion of qualified A selected to qualified A population', size=7)
        ax2.set_ylabel('Proportion of qualified A selected to qualified A population', size=7)
        ax3.set_ylabel('Proportion of qualified A selected to qualified A population', size=7)
        ax4.set_ylabel('Proportion of qualified A selected to qualified A population', size=7)

        ax1.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax2.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax3.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax4.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)

        ax1.set_title(r'One candidate selected', size=10)
        ax2.set_title(r'Two candidate selected', size=10)
        ax3.set_title(r'Three candidate selected', size=10)
        ax4.set_title(r'Four candidate selected', size=10)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

    if name_index == 1:
        # fig.suptitle(r'Proportion of True B candidate selected from qualified B population as a function of $\epsilon_{1}$', size=15)
        # ax1.set_ylabel('Proportion of True B candidate selected from qualified B population', size=8)
        fig.suptitle(
            r'Proportion of qualified B candidate selection to the qualified B population as a function of $\epsilon_{1}$',
            size=10)

        ax1.set_ylabel('Proportion of qualified B selected to qualified B population', size=7)
        ax2.set_ylabel('Proportion of qualified B selected to qualified B population', size=7)
        ax3.set_ylabel('Proportion of qualified B selected to qualified B population', size=7)
        ax4.set_ylabel('Proportion of qualified B selected to qualified B population', size=7)

        ax1.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax2.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax3.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax4.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)

        ax1.set_title(r'One candidate selected', size=10)
        ax2.set_title(r'Two candidate selected', size=10)
        ax3.set_title(r'Three candidate selected', size=10)
        ax4.set_title(r'Four candidate selected', size=10)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

    if name_index == 2:
        flag = True
        fig.suptitle(r'Fairness $(\gamma)$ as a function of $\epsilon_{1}$', size=10)
        # plt.ylabel('Fairness $(\gamma)$', size=10)
        ax1.set_ylabel('Fairness $(\gamma)$', size=7)
        ax2.set_ylabel('Fairness $(\gamma)$', size=7)
        ax3.set_ylabel('Fairness $(\gamma)$', size=7)
        ax4.set_ylabel('Fairness $(\gamma)$', size=7)

        ax1.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax2.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax3.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax4.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)

        ax1.set_title(r'One candidate selected', size=10)
        ax2.set_title(r'Two candidate selected', size=10)
        ax3.set_title(r'Three candidate selected', size=10)
        ax4.set_title(r'Four candidate selected', size=10)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

    if name_index == 3:
        fig.suptitle(r'Proportion of True selection to the qualified population as a function of $\epsilon_{1}$',
                     size=10)
        ax1.set_ylabel('Proportion of True selection', size=7)
        ax2.set_ylabel('Proportion of True selection', size=7)
        ax3.set_ylabel('Proportion of True selection', size=7)
        ax4.set_ylabel('Proportion of True selection', size=7)

        ax1.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax2.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax3.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax4.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)

        ax1.set_title(r'One candidate selected', size=10)
        ax2.set_title(r'Two candidate selected', size=10)
        ax3.set_title(r'Three candidate selected', size=10)
        ax4.set_title(r'Four candidate selected', size=10)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

    if name_index == 4:
        data1_2 = [number / float(2) for number in data1_2]
        data2_2 = [number / float(2) for number in data2_2]
        data3_2 = [number / float(2) for number in data3_2]
        data4_2 = [number / float(2) for number in data4_2]
        data5_2 = [number / float(2) for number in data5_2]
        data6_2 = [number / float(2) for number in data6_2]
        data7_2 = [number / float(2) for number in data7_2]
        data8_2 = [number / float(2) for number in data8_2]
        data9_2 = [number / float(2) for number in data9_2]
        data10_2 = [number / float(2) for number in data10_2]

        data1_3 = [number / float(3) for number in data1_3]
        data2_3 = [number / float(3) for number in data2_3]
        data3_3 = [number / float(3) for number in data3_3]
        data4_3 = [number / float(3) for number in data4_3]
        data5_3 = [number / float(3) for number in data5_3]
        data6_3 = [number / float(3) for number in data6_3]
        data7_3 = [number / float(3) for number in data7_3]
        data8_3 = [number / float(3) for number in data8_3]
        data9_3 = [number / float(3) for number in data9_3]
        data10_3 = [number / float(3) for number in data10_3]

        data1_4 = [number / float(4) for number in data1_4]
        data2_4 = [number / float(4) for number in data2_4]
        data3_4 = [number / float(4) for number in data3_4]
        data4_4 = [number / float(4) for number in data4_4]
        data5_4 = [number / float(4) for number in data5_4]
        data6_4 = [number / float(4) for number in data6_4]
        data7_4 = [number / float(4) for number in data7_4]
        data8_4 = [number / float(4) for number in data8_4]
        data9_4 = [number / float(4) for number in data9_4]
        data10_4 = [number / float(4) for number in data10_4]

        fig.suptitle(r'Accuracy $(\Theta)$ as a function of $\epsilon_{1}$',
                     size=10)
        ax1.set_ylabel(r'Accuracy $(\Theta)$', size=7)
        ax2.set_ylabel(r'Accuracy $(\Theta)$', size=7)
        ax3.set_ylabel(r'Accuracy $(\Theta)$', size=7)
        ax4.set_ylabel(r'Accuracy $(\Theta)$', size=7)

        ax1.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax2.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax3.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax4.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)

        ax1.set_title(r'One candidate selected', size=10)
        ax2.set_title(r'Two candidate selected', size=10)
        ax3.set_title(r'Three candidate selected', size=10)
        ax4.set_title(r'Four candidate selected', size=10)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

    if name_index == 5:
        data1_2 = [number / float(2) for number in data1_2]
        data2_2 = [number / float(2) for number in data2_2]
        data3_2 = [number / float(2) for number in data3_2]
        data4_2 = [number / float(2) for number in data4_2]
        data5_2 = [number / float(2) for number in data5_2]
        data6_2 = [number / float(2) for number in data6_2]
        data7_2 = [number / float(2) for number in data7_2]
        data8_2 = [number / float(2) for number in data8_2]
        data9_2 = [number / float(2) for number in data9_2]
        data10_2 = [number / float(2) for number in data10_2]

        data1_3 = [number / float(3) for number in data1_3]
        data2_3 = [number / float(3) for number in data2_3]
        data3_3 = [number / float(3) for number in data3_3]
        data4_3 = [number / float(3) for number in data4_3]
        data5_3 = [number / float(3) for number in data5_3]
        data6_3 = [number / float(3) for number in data6_3]
        data7_3 = [number / float(3) for number in data7_3]
        data8_3 = [number / float(3) for number in data8_3]
        data9_3 = [number / float(3) for number in data9_3]
        data10_3 = [number / float(3) for number in data10_3]

        data1_4 = [number / float(4) for number in data1_4]
        data2_4 = [number / float(4) for number in data2_4]
        data3_4 = [number / float(4) for number in data3_4]
        data4_4 = [number / float(4) for number in data4_4]
        data5_4 = [number / float(4) for number in data5_4]
        data6_4 = [number / float(4) for number in data6_4]
        data7_4 = [number / float(4) for number in data7_4]
        data8_4 = [number / float(4) for number in data8_4]
        data9_4 = [number / float(4) for number in data9_4]
        data10_4 = [number / float(4) for number in data10_4]

        fig.suptitle(r'Proportion of True selection from A in 10,000 runs as a function of $\epsilon_{1}$',
                     size=10)
        ax1.set_ylabel('Proportion of True candidate from A selection', size=7)
        ax2.set_ylabel('Proportion of True candidate from A selection', size=7)
        ax3.set_ylabel('Proportion of True candidate from A selection', size=7)
        ax4.set_ylabel('Proportion of True candidate from A selection', size=7)

        ax1.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax2.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax3.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax4.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)

        ax1.set_title(r'One candidate selected', size=10)
        ax2.set_title(r'Two candidate selected', size=10)
        ax3.set_title(r'Three candidate selected', size=10)
        ax4.set_title(r'Four candidate selected', size=10)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

    if name_index == 6:
        data1_2 = [number / float(2) for number in data1_2]
        data2_2 = [number / float(2) for number in data2_2]
        data3_2 = [number / float(2) for number in data3_2]
        data4_2 = [number / float(2) for number in data4_2]
        data5_2 = [number / float(2) for number in data5_2]
        data6_2 = [number / float(2) for number in data6_2]
        data7_2 = [number / float(2) for number in data7_2]
        data8_2 = [number / float(2) for number in data8_2]
        data9_2 = [number / float(2) for number in data9_2]
        data10_2 = [number / float(2) for number in data10_2]

        data1_3 = [number / float(3) for number in data1_3]
        data2_3 = [number / float(3) for number in data2_3]
        data3_3 = [number / float(3) for number in data3_3]
        data4_3 = [number / float(3) for number in data4_3]
        data5_3 = [number / float(3) for number in data5_3]
        data6_3 = [number / float(3) for number in data6_3]
        data7_3 = [number / float(3) for number in data7_3]
        data8_3 = [number / float(3) for number in data8_3]
        data9_3 = [number / float(3) for number in data9_3]
        data10_3 = [number / float(3) for number in data10_3]

        data1_4 = [number / float(4) for number in data1_4]
        data2_4 = [number / float(4) for number in data2_4]
        data3_4 = [number / float(4) for number in data3_4]
        data4_4 = [number / float(4) for number in data4_4]
        data5_4 = [number / float(4) for number in data5_4]
        data6_4 = [number / float(4) for number in data6_4]
        data7_4 = [number / float(4) for number in data7_4]
        data8_4 = [number / float(4) for number in data8_4]
        data9_4 = [number / float(4) for number in data9_4]
        data10_4 = [number / float(4) for number in data10_4]

        fig.suptitle(r'Proportion of True selection from B in 10,000 runs as a function of $\epsilon_{1}$',
                     size=10)
        ax1.set_ylabel('Proportion of True candidate from B selection', size=7)
        ax2.set_ylabel('Proportion of True candidate from B selection', size=7)
        ax3.set_ylabel('Proportion of True candidate from B selection', size=7)
        ax4.set_ylabel('Proportion of True candidate from B selection', size=7)

        ax1.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax2.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax3.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)
        ax4.set_xlabel(r'Privacy loss $\epsilon_{1}$', size=7)

        ax1.set_title(r'One candidate selected', size=10)
        ax2.set_title(r'Two candidate selected', size=10)
        ax3.set_title(r'Three candidate selected', size=10)
        ax4.set_title(r'Four candidate selected', size=10)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

    xp1 = np.linspace(0, 20, 2000)
    xp2 = np.linspace(0, 20, 2000)
    xp3 = np.linspace(0, 20, 2000)
    xp4 = np.linspace(0, 20, 2000)
    xp5 = np.linspace(0, 20, 2000)
    xp6 = np.linspace(0, 20, 2000)
    xp7 = np.linspace(0, 20, 2000)
    xp8 = np.linspace(0, 20, 2000)
    xp9 = np.linspace(0, 20, 2000)
    xp10 = np.linspace(0, 20, 200)

    z1_1 = np.polyfit(epsilons1, data1_1, 5)
    p1_1 = np.poly1d(z1_1)
    z2_1 = np.polyfit(epsilons2, data2_1, 5)
    p2_1 = np.poly1d(z2_1)
    z3_1 = np.polyfit(epsilons3, data3_1, 5)
    p3_1 = np.poly1d(z3_1)
    z4_1 = np.polyfit(epsilons4, data4_1, 5)
    p4_1 = np.poly1d(z4_1)
    z5_1 = np.polyfit(epsilons5, data5_1, 5)
    p5_1 = np.poly1d(z5_1)
    z6_1 = np.polyfit(epsilons6, data6_1, 5)
    p6_1 = np.poly1d(z6_1)
    z7_1 = np.polyfit(epsilons7, data7_1, 5)
    p7_1 = np.poly1d(z7_1)
    z8_1 = np.polyfit(epsilons8, data8_1, 5)
    p8_1 = np.poly1d(z8_1)
    z9_1 = np.polyfit(epsilons9, data9_1, 5)
    p9_1 = np.poly1d(z9_1)
    z10_1 = np.polyfit(epsilons10, data10_1, 5)
    p10_1 = np.poly1d(z10_1)

    z1_2 = np.polyfit(epsilons1, data1_2, 5)
    p1_2 = np.poly1d(z1_2)
    z2_2 = np.polyfit(epsilons2, data2_2, 5)
    p2_2 = np.poly1d(z2_2)
    z3_2 = np.polyfit(epsilons3, data3_2, 5)
    p3_2 = np.poly1d(z3_2)
    z4_2 = np.polyfit(epsilons4, data4_2, 5)
    p4_2 = np.poly1d(z4_2)
    z5_2 = np.polyfit(epsilons5, data5_2, 5)
    p5_2 = np.poly1d(z5_2)
    z6_2 = np.polyfit(epsilons6, data6_2, 5)
    p6_2 = np.poly1d(z6_2)
    z7_2 = np.polyfit(epsilons7, data7_2, 5)
    p7_2 = np.poly1d(z7_2)
    z8_2 = np.polyfit(epsilons8, data8_2, 5)
    p8_2 = np.poly1d(z8_2)
    z9_2 = np.polyfit(epsilons9, data9_2, 5)
    p9_2 = np.poly1d(z9_2)
    z10_2 = np.polyfit(epsilons10, data10_2, 5)
    p10_2 = np.poly1d(z10_2)

    z1_3 = np.polyfit(epsilons1, data1_3, 5)
    p1_3 = np.poly1d(z1_3)
    z2_3 = np.polyfit(epsilons2, data2_3, 5)
    p2_3 = np.poly1d(z2_3)
    z3_3 = np.polyfit(epsilons3, data3_3, 5)
    p3_3 = np.poly1d(z3_3)
    z4_3 = np.polyfit(epsilons4, data4_3, 5)
    p4_3 = np.poly1d(z4_3)
    z5_3 = np.polyfit(epsilons5, data5_3, 5)
    p5_3 = np.poly1d(z5_3)
    z6_3 = np.polyfit(epsilons6, data6_3, 5)
    p6_3 = np.poly1d(z6_3)
    z7_3 = np.polyfit(epsilons7, data7_3, 5)
    p7_3 = np.poly1d(z7_3)
    z8_3 = np.polyfit(epsilons8, data8_3, 5)
    p8_3 = np.poly1d(z8_3)
    z9_3 = np.polyfit(epsilons9, data9_3, 5)
    p9_3 = np.poly1d(z9_3)
    z10_3 = np.polyfit(epsilons10, data10_3, 5)
    p10_3 = np.poly1d(z10_3)

    z1_4 = np.polyfit(epsilons1, data1_4, 5)
    p1_4 = np.poly1d(z1_4)
    z2_4 = np.polyfit(epsilons2, data2_4, 5)
    p2_4 = np.poly1d(z2_4)
    z3_4 = np.polyfit(epsilons3, data3_4, 5)
    p3_4 = np.poly1d(z3_4)
    z4_4 = np.polyfit(epsilons4, data4_4, 5)
    p4_4 = np.poly1d(z4_4)
    z5_4 = np.polyfit(epsilons5, data5_4, 5)
    p5_4 = np.poly1d(z5_4)
    z6_4 = np.polyfit(epsilons6, data6_4, 5)
    p6_4 = np.poly1d(z6_4)
    z7_4 = np.polyfit(epsilons7, data7_4, 5)
    p7_4 = np.poly1d(z7_4)
    z8_4 = np.polyfit(epsilons8, data8_4, 5)
    p8_4 = np.poly1d(z8_4)
    z9_4 = np.polyfit(epsilons9, data9_4, 5)
    p9_4 = np.poly1d(z9_4)
    z10_4 = np.polyfit(epsilons10, data10_4, 5)
    p10_4 = np.poly1d(z10_4)

    ax1.plot(xp1, p1_1(xp1), '-', color='lime', label=r'($\alpha$=1)')
    ax2.plot(xp1, p1_2(xp1), '-', color='lime', label=r'($\alpha$=1)')
    ax3.plot(xp1, p1_3(xp1), '-', color='lime', label=r'($\alpha$=1)')
    ax4.plot(xp1, p1_4(xp1), '-', color='lime', label=r'($\alpha$=1)')
    # plt.plot(epsilons1, data1, color='darkgray')
    # plt.plot(xp1, p1(xp1), '-', color='lime', label=r'($\alpha$=same)')
    # plt.plot(epsilons2, data2, color='darkgray')
    ax1.plot(xp2, p2_1(xp2), '-', color='steelblue',
             label=r'($\alpha$=$\frac{' + str(proportions[1][0]) + '}{' + str(proportions[1][1]) + '}$)')
    ax2.plot(xp2, p2_2(xp2), '-', color='steelblue',
             label=r'($\alpha$=$\frac{' + str(proportions[1][0]) + '}{' + str(proportions[1][1]) + '}$)')
    ax3.plot(xp2, p2_3(xp2), '-', color='steelblue',
             label=r'($\alpha$=$\frac{' + str(proportions[1][0]) + '}{' + str(proportions[1][1]) + '}$)')
    ax4.plot(xp2, p2_4(xp2), '-', color='steelblue',
             label=r'($\alpha$=$\frac{' + str(proportions[1][0]) + '}{' + str(proportions[1][1]) + '}$)')
    # plt.plot(epsilons3, data3, color='darkgray')
    ax1.plot(xp3, p3_1(xp3), '-', color='darkviolet',
             label=r'($\alpha$=$\frac{' + str(proportions[2][0]) + '}{' + str(proportions[2][1]) + '}$)')
    ax2.plot(xp3, p3_2(xp3), '-', color='darkviolet',
             label=r'($\alpha$=$\frac{' + str(proportions[2][0]) + '}{' + str(proportions[2][1]) + '}$)')
    ax3.plot(xp3, p3_3(xp3), '-', color='darkviolet',
             label=r'($\alpha$=$\frac{' + str(proportions[2][0]) + '}{' + str(proportions[2][1]) + '}$)')
    ax4.plot(xp3, p3_4(xp3), '-', color='darkviolet',
             label=r'($\alpha$=$\frac{' + str(proportions[2][0]) + '}{' + str(proportions[2][1]) + '}$)')
    # plt.plot(epsilons4, data4, color='darkgray')

    '''
     ax1.plot(xp4, p4_1(xp4), '-', color='cornflowerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[3][0]) + '}{' + str(proportions[3][1]) + '}$)')
    ax2.plot(xp4, p4_2(xp4), '-', color='cornflowerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[3][0]) + '}{' + str(proportions[3][1]) + '}$)')
    ax3.plot(xp4, p4_3(xp4), '-', color='cornflowerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[3][0]) + '}{' + str(proportions[3][1]) + '}$)')
    ax4.plot(xp4, p4_4(xp4), '-', color='cornflowerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[3][0]) + '}{' + str(proportions[3][1]) + '}$)')
    '''

    # plt.plot(epsilons5, data5, color='darkgray')
    ax1.plot(xp5, p5_1(xp5), '-', color='blue',
             label=r'($\alpha$=$\frac{' + str(proportions[4][0]) + '}{' + str(proportions[4][1]) + '}$)')
    ax2.plot(xp5, p5_2(xp5), '-', color='blue',
             label=r'($\alpha$=$\frac{' + str(proportions[4][0]) + '}{' + str(proportions[4][1]) + '}$)')
    ax3.plot(xp5, p5_3(xp5), '-', color='blue',
             label=r'($\alpha$=$\frac{' + str(proportions[4][0]) + '}{' + str(proportions[4][1]) + '}$)')
    ax4.plot(xp5, p5_4(xp5), '-', color='blue',
             label=r'($\alpha$=$\frac{' + str(proportions[4][0]) + '}{' + str(proportions[4][1]) + '}$)')
    # plt.plot(epsilons6, data6, color='darkgray')
    ax1.plot(xp6, p6_1(xp6), '-', color='dodgerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[5][0]) + '}{' + str(proportions[5][1]) + '}$)')
    ax2.plot(xp6, p6_2(xp6), '-', color='dodgerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[5][0]) + '}{' + str(proportions[5][1]) + '}$)')
    ax3.plot(xp6, p6_3(xp6), '-', color='dodgerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[5][0]) + '}{' + str(proportions[5][1]) + '}$)')
    ax4.plot(xp6, p6_4(xp6), '-', color='dodgerblue',
             label=r'($\alpha$=$\frac{' + str(proportions[5][0]) + '}{' + str(proportions[5][1]) + '}$)')
    # plt.plot(epsilons7, data7, color='darkgray')
    ax1.plot(xp7, p7_1(xp7), '-', color='teal',
             label=r'($\alpha$=$\frac{' + str(proportions[6][0]) + '}{' + str(proportions[6][1]) + '}$)')
    ax2.plot(xp7, p7_2(xp7), '-', color='teal',
             label=r'($\alpha$=$\frac{' + str(proportions[6][0]) + '}{' + str(proportions[6][1]) + '}$)')
    ax3.plot(xp7, p7_3(xp7), '-', color='teal',
             label=r'($\alpha$=$\frac{' + str(proportions[6][0]) + '}{' + str(proportions[6][1]) + '}$)')
    ax4.plot(xp7, p7_4(xp7), '-', color='teal',
             label=r'($\alpha$=$\frac{' + str(proportions[6][0]) + '}{' + str(proportions[6][1]) + '}$)')
    # plt.plot(epsilons8, data8, color='darkgray')
    ax1.plot(xp8, p8_1(xp8), '-', color='royalblue',
             label=r'($\alpha$=$\frac{' + str(proportions[7][0]) + '}{' + str(proportions[7][1]) + '}$)')
    ax2.plot(xp8, p8_2(xp8), '-', color='royalblue',
             label=r'($\alpha$=$\frac{' + str(proportions[7][0]) + '}{' + str(proportions[7][1]) + '}$)')
    ax3.plot(xp8, p8_3(xp8), '-', color='royalblue',
             label=r'($\alpha$=$\frac{' + str(proportions[7][0]) + '}{' + str(proportions[7][1]) + '}$)')
    ax4.plot(xp8, p8_4(xp8), '-', color='royalblue',
             label=r'($\alpha$=$\frac{' + str(proportions[7][0]) + '}{' + str(proportions[7][1]) + '}$)')
    # plt.plot(epsilons9, data9, color='darkgray')
    ax1.plot(xp9, p9_1(xp9), '-', color='deepskyblue',
             label=r'($\alpha$=$\frac{' + str(proportions[8][0]) + '}{' + str(proportions[8][1]) + '}$)')
    ax2.plot(xp9, p9_2(xp9), '-', color='deepskyblue',
             label=r'($\alpha$=$\frac{' + str(proportions[8][0]) + '}{' + str(proportions[8][1]) + '}$)')
    ax3.plot(xp9, p9_3(xp9), '-', color='deepskyblue',
             label=r'($\alpha$=$\frac{' + str(proportions[8][0]) + '}{' + str(proportions[8][1]) + '}$)')
    ax4.plot(xp9, p9_4(xp9), '-', color='deepskyblue',
             label=r'($\alpha$=$\frac{' + str(proportions[8][0]) + '}{' + str(proportions[8][1]) + '}$)')

    ax1.plot(xp10, p10_1(xp10), '-', color='darkred',
             label='Debiased model')
    ax2.plot(xp10, p10_2(xp10), '-', color='darkred',
             label='Debiased model')
    ax3.plot(xp10, p10_3(xp10), '-', color='darkred',
             label='Debiased model')
    ax4.plot(xp10, p10_4(xp10), '-', color='darkred',
             label='Debiased model')

    # plt.subplots_adjust(hspace=0.5, wspace=0.4)
    ax1.legend(prop={'size': 6})
    ax2.legend(prop={'size': 6})
    ax3.legend(prop={'size': 6})
    ax4.legend(prop={'size': 6})
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    #plt.show()
    plt.savefig(str(names[name_index]) + '_multiple_selection_plus_debiased_3.pdf', dpi=300)
    plt.show()
    plt.clf()
    print('hi')


