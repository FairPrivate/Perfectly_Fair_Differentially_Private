import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

epsilons1 = [0.01*x for x in range(2000)]
epsilons1[0] = 0.0001

epsilons2 = [0.01*x for x in range(2000)]
epsilons2[0] = 0.0001

epsilons3 = [0.01*x for x in range(2000)]
epsilons3[0] = 0.0001

epsilons4 = [0.01*x for x in range(2000)]
epsilons4[0] = 0.0001

epsilons5 = [0.0102*x for x in range(2000)]
epsilons5[0] = 0.0001

epsilons6 = [0.0105*x for x in range(2000)]
epsilons6[0] = 0.0001

epsilons7 = [0.011*x for x in range(2000)]
epsilons7[0] = 0.0001

epsilons8 = [0.015*x for x in range(2000)]
epsilons8[0] = 0.0001

epsilons9 = [0.02*x for x in range(2000)]
epsilons9[0] = 0.0001

epsilons10 = [0.1*x for x in range(200)]
epsilons10[0] = 0.0001

names = ['wha_equal_opportunity', 'wha_accuracy_in_all_iterations']

wha_file_name_1_105 = ['wha_1_105_equal_opportunity', 'wha_1_105_accuracy_in_all_iterations']

wha_file_name_1_102 = ['wha_1_102_equal_opportunity', 'wha_1_102_accuracy_in_all_iterations']

wha_file_name_1_1 = ['wha_1_1_equal_opportunity', 'wha_1_1_accuracy_in_all_iterations']

wha_file_name_102_1 = ['wha_102_1_equal_opportunity', 'wha_102_1_accuracy_in_all_iterations']

wha_file_name_105_1 = ['wha_105_1_equal_opportunity', 'wha_105_1_accuracy_in_all_iterations']

wha_file_name_11_1 = ['wha_11_1_equal_opportunity', 'wha_11_1_accuracy_in_all_iterations']

wha_file_name_15_1 = ['wha_15_1_equal_opportunity', 'wha_15_1_accuracy_in_all_iterations']

wha_file_name_2_1 = ['wha_2_1_equal_opportunity', 'wha_2_1_accuracy_in_all_iterations']


same_noise_wha_file_name = ['wha_same_equal_opportunity', 'wha_same_accuracy_in_all_iterations']

adversarial = ['wha_equal_opportunity_debiased_multiple_selection_1',
               'wha_accuracy_in_all_iterations_debiased_multiple_selection_1']

expeiments = [same_noise_wha_file_name, wha_file_name_1_105, wha_file_name_1_102, wha_file_name_1_1, wha_file_name_102_1
              , wha_file_name_105_1, wha_file_name_11_1, wha_file_name_15_1, wha_file_name_2_1, adversarial]
proportions = [['same'], [1, 1.05], [1, 1.02], [1, 1], [1.02, 1], [1.05, 1], [1.1, 1], [1.5, 1], [2, 1]]
for name_index in range(len(wha_file_name_1_1)):
    data1 = pd.read_csv(str(expeiments[0][name_index])+'.csv')
    data2 = pd.read_csv(str(expeiments[1][name_index])+'.csv')
    data3 = pd.read_csv(str(expeiments[2][name_index])+'.csv')
    data4 = pd.read_csv(str(expeiments[3][name_index])+'.csv')
    data5 = pd.read_csv(str(expeiments[4][name_index])+'.csv')
    data6 = pd.read_csv(str(expeiments[5][name_index])+'.csv')
    data7 = pd.read_csv(str(expeiments[6][name_index])+'.csv')
    data8 = pd.read_csv(str(expeiments[7][name_index])+'.csv')
    data9 = pd.read_csv(str(expeiments[8][name_index])+'.csv')
    data10 = pd.read_csv(str(expeiments[9][name_index])+'.csv')

    data1 = data1.drop(columns='Unnamed: 0')
    data2 = data2.drop(columns='Unnamed: 0')
    data3 = data3.drop(columns='Unnamed: 0')
    data4 = data4.drop(columns='Unnamed: 0')
    data5 = data5.drop(columns='Unnamed: 0')
    data6 = data6.drop(columns='Unnamed: 0')
    data7 = data7.drop(columns='Unnamed: 0')
    data8 = data8.drop(columns='Unnamed: 0')
    data9 = data9.drop(columns='Unnamed: 0')
    data10 = data10.drop(columns='Unnamed: 0')
    datas = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]

    for data_index in range(len(datas)):
        data11 = np.array(datas[data_index])
        datas[data_index] = []
        for each in data11:
            datas[data_index].append(each[0])

    data1 = datas[0]
    data2 = datas[1]
    data3 = datas[2]
    data4 = datas[3]
    data5 = datas[4]
    data6 = datas[5]
    data7 = datas[6]
    data8 = datas[7]
    data9 = datas[8]
    data10 = datas[9]


    data10 = data10[:200]
    epsilons10 = epsilons10[:200]

    data8 = data8[:1335]
    epsilons8 = epsilons8[:1335]

    data9 = data9[:1001]
    epsilons9 = epsilons9[:1001]

    plt.rcParams["font.family"] = "Times New Roman"

    if name_index == 0:
        plt.title(r'Fairness $(\gamma)$ as a function of $\epsilon_{1}$', size=10)
        plt.ylabel('Fairness $(\gamma)$', size=10)
        plt.xlabel(r'Privacy loss $\epsilon_{1}$', size=10)
        plt.grid()

    if name_index == 1:
        #fig.suptitle(r'Proportion of True selection in 10,000 runs as a function of $\epsilon_{1}$', size=15)
        plt.title(r'Accuracy $(\Theta)$ as a function of $\epsilon_{1}$', size=10)
        plt.ylabel(r'Accuracy $(\Theta)$', size=10)
        plt.xlabel(r'Privacy loss $\epsilon_{1}$', size=10)
        plt.grid()


    xp1 = np.linspace(0, 20, 2000)
    xp2 = np.linspace(0, 20, 2000)
    xp3 = np.linspace(0, 20, 2000)
    xp4 = np.linspace(0, 20, 2000)
    xp5 = np.linspace(0, 20, 2000)
    xp6 = np.linspace(0, 20, 2000)
    xp7 = np.linspace(0, 20, 2000)
    xp8 = np.linspace(0, 20, 2000)
    xp9 = np.linspace(0, 20, 2000)
    xp10 = np.linspace(0, 20, 2000)

    z1 = np.polyfit(epsilons1, data1, 5)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(epsilons2, data2, 5)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(epsilons3, data3, 5)
    p3 = np.poly1d(z3)
    z4 = np.polyfit(epsilons4, data4, 5)
    p4 = np.poly1d(z4)
    z5 = np.polyfit(epsilons5, data5, 5)
    p5 = np.poly1d(z5)
    z6 = np.polyfit(epsilons6, data6, 5)
    p6 = np.poly1d(z6)
    z7 = np.polyfit(epsilons7, data7, 5)
    p7 = np.poly1d(z7)
    z8 = np.polyfit(epsilons8, data8, 5)
    p8 = np.poly1d(z8)
    z9 = np.polyfit(epsilons9, data9, 5)
    p9 = np.poly1d(z9)
    z10 = np.polyfit(epsilons10, data10, 5)
    p10 = np.poly1d(z10)

    #plt.plot(epsilons1, data1, color='darkgray')
    plt.plot(xp1, p1(xp1), '-', color='lime', label=r'($\alpha$=1)')
    #plt.plot(epsilons2, data2, color='darkgray')
    plt.plot(xp2, p2(xp2), '-', color='steelblue', label =r'($\alpha$=$\frac{'+str(proportions[1][0])+'}{'+str(proportions[1][1])+'}$)')
    #plt.plot(epsilons3, data3, color='darkgray')
    plt.plot(xp3, p3(xp3), '-', color='darkviolet', label =r'($\alpha$=$\frac{'+str(proportions[2][0])+'}{'+str(proportions[2][1])+'}$)')
    #plt.plot(epsilons4, data4, color='darkgray')
    #plt.plot(xp4, p4(xp4), '-', color='cornflowerblue', label= r'($\alpha$=$\frac{'+str(proportions[3][0])+'}{'+str(proportions[3][1])+'}$)')
    #plt.plot(epsilons5, data5, color='darkgray')
    plt.plot(xp5, p5(xp5), '-', color='blue', label= r'($\alpha$=$\frac{'+str(proportions[4][0])+'}{'+str(proportions[4][1])+'}$)')
    #plt.plot(epsilons6, data6, color='darkgray')
    plt.plot(xp6, p6(xp6), '-', color='dodgerblue', label = r'($\alpha$=$\frac{'+str(proportions[5][0])+'}{'+str(proportions[5][1])+'}$)')
    #plt.plot(epsilons7, data7, color='darkgray')
    plt.plot(xp7, p7(xp7), '-', color='teal', label = r'($\alpha$=$\frac{'+str(proportions[6][0])+'}{'+str(proportions[6][1])+'}$)')
    #plt.plot(epsilons8, data8, color='darkgray')
    plt.plot(xp8, p8(xp8), '-', color='royalblue', label = r'($\alpha$=$\frac{'+str(proportions[7][0])+'}{'+str(proportions[7][1])+'}$)')
    #plt.plot(epsilons9, data9, color='darkgray')
    plt.plot(xp9, p9(xp9), '-', color='deepskyblue', label = r'($\alpha$=$\frac{'+str(proportions[8][0])+'}{'+str(proportions[8][1])+'}$)')
    plt.plot(xp10, p10(xp10), '-', color='darkred', label = r'Debiased model')

    plt.legend()
    plt.savefig(str(names[name_index]) + '_all_in_one_1_plus_debiased_1.pdf', dpi=300)
    plt.show()
    plt.clf()
