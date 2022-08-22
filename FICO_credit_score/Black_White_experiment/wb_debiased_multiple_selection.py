import pandas as pd
import numpy as np
import random

from bisect import bisect_left
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets import StandardDataset
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

number_of_iterations = 5000
number_of_samples = 10
# WB
Y0b = 0.33655060999999405
Y0a = 0.2413262699999973
alpha_a = 0.8793309517363427

P00 = pd.read_csv('P00_a0.csv')
P01 = pd.read_csv('P01_a1.csv')
P11 = pd.read_csv('P11_b1.csv')
P10 = pd.read_csv('P10_b0.csv')

a = [P00, P01, P10, P11]
for each in a:
    each = each.drop('Unnamed: 0', axis=1, inplace=True)

P00 = P00.to_numpy()
P01 = P01.to_numpy()
P10 = P10.to_numpy()
P11 = P11.to_numpy()

CDF = pd.read_csv("CDF.csv")
rho = CDF['Score'] / 100
epsilons = [0.1 * x for x in range(250)]
#epsilon1 = [0.1 * x for x in range(250)]
#epsilon2 = [5 * x for x in range(200)]
#epsilons = epsilon1 #+ epsilon2[4:]
epsilons[0] = 0.0001
sensitivity = 1

db = pd.read_csv('wb_generated_samples_corrected.csv')
db.drop('Unnamed: 0', axis=1, inplace=True)
dataset = StandardDataset(db, label_name='label', favorable_classes=[1], protected_attribute_names=['protected_class'],
                          privileged_classes=[[0]])

attr = dataset.protected_attribute_names[0]
idx = dataset.protected_attribute_names.index(attr)
privileged_groups = [{attr: dataset.privileged_protected_attributes[idx][0]}]
unprivileged_groups = [{attr: dataset.unprivileged_protected_attributes[idx][0]}]
# dataset_minority_train, dataset_minority_test = dataset_minority.split([0.99], shuffle=False)
sess = tf.Session()
debiased_model_01 = AdversarialDebiasing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups,
                                         scope_name='debiased_classifier', debias=True, sess=sess, num_epochs=10,
                                         batch_size=20,
                                         equal_opportunity=True, adversary_loss_weight=8.0)

debiased_model_01.fit(dataset)
# 10, 30
# 10, 20 No
# 20, 1

a_times_iterations = []
b_times_iterations = []
a1_times_iterations = []
a0_times_iterations = []
b1_times_iterations = []
b0_times_iterations = []


True_a_debiased_iteration_1 = []
True_b_debiased_iteration_1 = []
False_a_debiased_iteration_1 = []
False_b_debiased_iteration_1 = []

True_a_debiased_iteration_2 = []
True_b_debiased_iteration_2 = []
False_a_debiased_iteration_2 = []
False_b_debiased_iteration_2 = []

True_a_debiased_iteration_3 = []
True_b_debiased_iteration_3 = []
False_a_debiased_iteration_3 = []
False_b_debiased_iteration_3 = []

True_a_debiased_iteration_4 = []
True_b_debiased_iteration_4 = []
False_a_debiased_iteration_4 = []
False_b_debiased_iteration_4 = []



True_a_original_iteration_1 = []
True_b_original_iteration_1 = []
False_a_original_iteration_1 = []
False_b_original_iteration_1 = []

True_a_original_iteration_2 = []
True_b_original_iteration_2 = []
False_a_original_iteration_2 = []
False_b_original_iteration_2 = []

True_a_original_iteration_3 = []
True_b_original_iteration_3 = []
False_a_original_iteration_3 = []
False_b_original_iteration_3 = []

True_a_original_iteration_4 = []
True_b_original_iteration_4 = []
False_a_original_iteration_4 = []
False_b_original_iteration_4 = []

True_a_iteration_original = []
True_b_iteration_original = []
False_a_iteration_original = []
False_b_iteration_original = []

True_a_iteration_debiased = []
True_b_iteration_debiased = []
False_a_iteration_debiased = []
False_b_iteration_debiased = []

without_noise_True_a_iteration_debiased = []
without_noise_True_b_iteration_debiased = []
without_noise_False_a_iteration_debiased = []
without_noise_False_b_iteration_debiased = []

without_noise_True_a_iteration_original = []
without_noise_True_b_iteration_original = []
without_noise_False_a_iteration_original = []
without_noise_False_b_iteration_original = []
a_t, a_f, b_t, b_f = [], [], [], []
similar_iteration = []
are_not_same = 0
has_b_iteration = 0
for ite in range(number_of_iterations):
    Q_a1 = []
    Q_b1 = []
    Q_a0 = []
    Q_b0 = []
    for i in range(number_of_samples):
        randa1 = random.random()
        posa1 = bisect_left(np.cumsum(P01), randa1)
        Q_a1.append(rho[posa1])  # Samples from Pr(R=r|A=0, Y=1)

        randb1 = random.random()
        posb1 = bisect_left(np.cumsum(P11), randb1)
        Q_b1.append(rho[posb1])  # Samples from Pr(R=r|A=1, Y=1)

        randa0 = random.random()
        posa0 = bisect_left(np.cumsum(P00), randa0)
        if posa0 == 198:
            posa0 = 197
        Q_a0.append(rho[posa0])  # Samples from Pr(R=r|A=0, Y=0)

        randb0 = random.random()
        posb0 = bisect_left(np.cumsum(P10), randb0)
        Q_b0.append(rho[posb0])  # Samples from Pr(R=r|A=1, Y=0)

    rand_numa = [random.random() for o in range(number_of_samples)]
    temp0a = []  # Index of A=0 and Y=1
    for j in range(number_of_samples):
        if rand_numa[j] < Y0a:
            temp0a.append(0)
        else:
            temp0a.append(1)
    Q_a_first = [x * y for (x, y) in zip(temp0a, Q_a1)]  # Pr(R=r|A=0, Y=1)
    temp0a_reversed = []  # Index of A=0 and Y=0
    for each in temp0a:
        if each == 0:
            temp0a_reversed.append(1)
        else:
            temp0a_reversed.append(0)
    Q_a_second = [x * y for (x, y) in zip(temp0a_reversed, Q_a0)]  # Pr(R=r|A=0, Y=0)
    Q_a = [x + y for (x, y) in zip(Q_a_first, Q_a_second)]

    rand_numb = [random.random() for o in range(number_of_samples)]
    temp0b = []  # Index of A=1 and Y=1
    for k in range(number_of_samples):
        if rand_numb[k] < Y0b:
            temp0b.append(0)
        else:
            temp0b.append(1)
    Q_b_first = [x * y for (x, y) in zip(temp0b, Q_b1)]  # Pr(R=r|A=1, Y=1)
    temp0b_reversed = []
    for each in temp0b:
        if each == 0:
            temp0b_reversed.append(1)
        else:
            temp0b_reversed.append(0)
    Q_b_second = [x * y for (x, y) in zip(temp0b_reversed, Q_b0)]  # Pr(R=r|A=1, Y=0)
    Q_b = [x + y for (x, y) in zip(Q_b_first, Q_b_second)]

    temp = []  # Index of A=0
    rand_num = [random.random() for l in range(number_of_samples)]
    for t in range(number_of_samples):
        if rand_num[t] < alpha_a:
            temp.append(1)
        else:
            temp.append(0)

    temp_reversed = []  # Index of A=1
    for each in temp:
        if each == 0:
            temp_reversed.append(1)
        else:
            temp_reversed.append(0)

    Qual_first = [x * y for (x, y) in zip(temp, Q_a)]
    Qual_second = [x * y for (x, y) in zip(temp_reversed, Q_b)]
    Qual = [x + y for (x, y) in zip(Qual_first, Qual_second)]

    temp0a_np = np.array(temp0a)
    temp0b_np = np.array(temp0b)
    temp_np = np.array(temp)

    all_from_a = np.where(temp_np == 1)[0]
    all_a_1s = np.where(temp0a_np == 1)[0]
    all_a_0s = np.where(temp0a_np == 0)[0]
    a_y1s = list(set(all_from_a) & set(all_a_1s))
    a_y0s = list(set(all_from_a) & set(all_a_0s))

    all_from_b = np.where(temp_np == 0)[0]
    all_b_1s = np.where(temp0b_np == 1)[0]
    all_b_0s = np.where(temp0b_np == 0)[0]
    b_y1s = list(set(all_from_b) & set(all_b_1s))
    b_y0s = list(set(all_from_b) & set(all_b_0s))

    labels = []
    for ii in range(number_of_samples):
        if ii in b_y1s:
            labels.append(1)
        elif ii in a_y1s:
            labels.append(1)
        else:
            labels.append(0)

    a_times_iterations.append((len(a_y1s) + len(a_y0s)))
    b_times_iterations.append(len(b_y1s) + len(b_y0s))
    a1_times_iterations.append(len(a_y1s))
    b1_times_iterations.append(len(b_y1s))
    a0_times_iterations.append(len(a_y0s))
    b0_times_iterations.append(len(b_y0s))

    if 1 in temp_reversed:
        has_b_iteration += 1
    samples_db = pd.DataFrame()
    Qual = np.asarray(Qual)
    labels = np.asarray(labels)
    temp_reversed = np.asarray(temp_reversed)
    samples_db['score'] = Qual
    samples_db['protected_class'] = temp_reversed
    samples_db['label'] = labels
    samples_dataset = StandardDataset(samples_db, label_name='label', favorable_classes=[1],
                                      protected_attribute_names=['protected_class'], privileged_classes=[[0]])

    samples_prediction = debiased_model_01.predict(samples_dataset)
    debiased_samples_score = samples_prediction.scores

    max_index_without_noise_debiased = np.argmax(debiased_samples_score)
    max_index_without_noise_original = np.argmax(Qual)

    if max_index_without_noise_debiased in a_y1s:
        without_noise_True_a_iteration_debiased.append(1)
        without_noise_True_b_iteration_debiased.append(0)
        without_noise_False_a_iteration_debiased.append(0)
        without_noise_False_b_iteration_debiased.append(0)
    elif max_index_without_noise_debiased in a_y0s:
        without_noise_True_a_iteration_debiased.append(0)
        without_noise_True_b_iteration_debiased.append(0)
        without_noise_False_a_iteration_debiased.append(1)
        without_noise_False_b_iteration_debiased.append(0)
    elif max_index_without_noise_debiased in b_y1s:
        without_noise_True_a_iteration_debiased.append(0)
        without_noise_True_b_iteration_debiased.append(1)
        without_noise_False_a_iteration_debiased.append(0)
        without_noise_False_b_iteration_debiased.append(0)
    else:
        without_noise_True_a_iteration_debiased.append(0)
        without_noise_True_b_iteration_debiased.append(0)
        without_noise_False_a_iteration_debiased.append(0)
        without_noise_False_b_iteration_debiased.append(1)

    if max_index_without_noise_original in a_y1s:
        without_noise_True_a_iteration_original.append(1)
        without_noise_True_b_iteration_original.append(0)
        without_noise_False_a_iteration_original.append(0)
        without_noise_False_b_iteration_original.append(0)
    elif max_index_without_noise_original in a_y0s:
        without_noise_True_a_iteration_original.append(0)
        without_noise_True_b_iteration_original.append(0)
        without_noise_False_a_iteration_original.append(1)
        without_noise_False_b_iteration_original.append(0)
    elif max_index_without_noise_original in b_y1s:
        without_noise_True_a_iteration_original.append(0)
        without_noise_True_b_iteration_original.append(1)
        without_noise_False_a_iteration_original.append(0)
        without_noise_False_b_iteration_original.append(0)
    else:
        without_noise_True_a_iteration_original.append(0)
        without_noise_True_b_iteration_original.append(0)
        without_noise_False_a_iteration_original.append(0)
        without_noise_False_b_iteration_original.append(1)
    if max_index_without_noise_original != max_index_without_noise_debiased:
        are_not_same += 1

    True_a_original_epsilon = []
    True_b_original_epsilon = []
    False_a_original_epsilon = []
    False_b_original_epsilon = []

    True_a_debiased_epsilon = []
    True_b_debiased_epsilon = []
    False_a_debiased_epsilon = []
    False_b_debiased_epsilon = []

    a_t_eps, a_f_eps, b_t_eps, b_f_eps = 0, 0, 0, 0
    similar_epsilon = 0
    not_similar_epsilon = 0

    for ep in range(len(epsilons)):
        noise = [np.random.laplace(loc=0, scale=sensitivity / epsilons[ep]) for i in range(number_of_samples)]
        noisy_debiased_samples_score = [x + y for (x, y) in zip(debiased_samples_score, noise)]
        noisy_original_samples_score = [x + y for (x, y) in zip(Qual, noise)]
        res_orig = sorted(range(len(noisy_original_samples_score)), key=lambda sub: noisy_original_samples_score[sub])[-4:]
        res_debiased = sorted(range(len(noisy_debiased_samples_score)), key=lambda sub: noisy_debiased_samples_score[sub])[-4:]
        ta_o, tb_o, fa_o, fb_o = 0, 0, 0, 0
        True_a_epsilon_o_m, True_b_epsilon_o_m, False_a_epsilon_o_m, False_b_epsilon_o_m = [], [], [], []
        for max_index_o in reversed(res_orig):
            if max_index_o in a_y1s:
                ta_o += 1
                True_a_epsilon_o_m.append(ta_o)
                True_b_epsilon_o_m.append(tb_o)
                False_a_epsilon_o_m.append(fa_o)
                False_b_epsilon_o_m.append(fb_o)
            elif max_index_o in b_y1s:
                tb_o += 1
                True_a_epsilon_o_m.append(ta_o)
                True_b_epsilon_o_m.append(tb_o)
                False_a_epsilon_o_m.append(fa_o)
                False_b_epsilon_o_m.append(fb_o)
            elif max_index_o in a_y0s:
                fa_o += 1
                True_a_epsilon_o_m.append(ta_o)
                True_b_epsilon_o_m.append(tb_o)
                False_a_epsilon_o_m.append(fa_o)
                False_b_epsilon_o_m.append(fb_o)
            elif max_index_o in b_y0s:
                fb_o += 1
                True_a_epsilon_o_m.append(ta_o)
                True_b_epsilon_o_m.append(tb_o)
                False_a_epsilon_o_m.append(fa_o)
                False_b_epsilon_o_m.append(fb_o)
        True_a_original_epsilon.append(True_a_epsilon_o_m)
        True_b_original_epsilon.append(True_b_epsilon_o_m)
        False_a_original_epsilon.append(False_a_epsilon_o_m)
        False_b_original_epsilon.append(False_b_epsilon_o_m)

        ta, tb, fa, fb = 0, 0, 0, 0
        True_a_epsilon_m, True_b_epsilon_m, False_a_epsilon_m, False_b_epsilon_m = [], [], [], []
        for max_index in reversed(res_debiased):
            if max_index in a_y1s:
                ta += 1
                True_a_epsilon_m.append(ta)
                True_b_epsilon_m.append(tb)
                False_a_epsilon_m.append(fa)
                False_b_epsilon_m.append(fb)
            elif max_index in b_y1s:
                tb += 1
                True_a_epsilon_m.append(ta)
                True_b_epsilon_m.append(tb)
                False_a_epsilon_m.append(fa)
                False_b_epsilon_m.append(fb)
            elif max_index in a_y0s:
                fa += 1
                True_a_epsilon_m.append(ta)
                True_b_epsilon_m.append(tb)
                False_a_epsilon_m.append(fa)
                False_b_epsilon_m.append(fb)
            elif max_index in b_y0s:
                fb += 1
                True_a_epsilon_m.append(ta)
                True_b_epsilon_m.append(tb)
                False_a_epsilon_m.append(fa)
                False_b_epsilon_m.append(fb)
        True_a_debiased_epsilon.append(True_a_epsilon_m)
        True_b_debiased_epsilon.append(True_b_epsilon_m)
        False_a_debiased_epsilon.append(False_a_epsilon_m)
        False_b_debiased_epsilon.append(False_b_epsilon_m)

    True_a_debiased_epsilon_df = pd.DataFrame(True_a_debiased_epsilon)
    True_b_debiased_epsilon_df = pd.DataFrame(True_b_debiased_epsilon)
    False_a_debiased_epsilon_df = pd.DataFrame(False_a_debiased_epsilon)
    False_b_debiased_epsilon_df = pd.DataFrame(False_b_debiased_epsilon)

    True_a_debiased_iteration_1.append(True_a_debiased_epsilon_df[0])
    True_b_debiased_iteration_1.append(True_b_debiased_epsilon_df[0])
    False_a_debiased_iteration_1.append(False_a_debiased_epsilon_df[0])
    False_b_debiased_iteration_1.append(False_b_debiased_epsilon_df[0])

    True_a_debiased_iteration_2.append(True_a_debiased_epsilon_df[1])
    True_b_debiased_iteration_2.append(True_b_debiased_epsilon_df[1])
    False_a_debiased_iteration_2.append(False_a_debiased_epsilon_df[1])
    False_b_debiased_iteration_2.append(False_b_debiased_epsilon_df[1])

    True_a_debiased_iteration_3.append(True_a_debiased_epsilon_df[2])
    True_b_debiased_iteration_3.append(True_b_debiased_epsilon_df[2])
    False_a_debiased_iteration_3.append(False_a_debiased_epsilon_df[2])
    False_b_debiased_iteration_3.append(False_b_debiased_epsilon_df[2])

    True_a_debiased_iteration_4.append(True_a_debiased_epsilon_df[3])
    True_b_debiased_iteration_4.append(True_b_debiased_epsilon_df[3])
    False_a_debiased_iteration_4.append(False_a_debiased_epsilon_df[3])
    False_b_debiased_iteration_4.append(False_b_debiased_epsilon_df[3])



    True_a_original_epsilon_df = pd.DataFrame(True_a_original_epsilon)
    True_b_original_epsilon_df = pd.DataFrame(True_b_original_epsilon)
    False_a_original_epsilon_df = pd.DataFrame(False_a_original_epsilon)
    False_b_original_epsilon_df = pd.DataFrame(False_b_original_epsilon)

    True_a_original_iteration_1.append(True_a_original_epsilon_df[0])
    True_b_original_iteration_1.append(True_b_original_epsilon_df[0])
    False_a_original_iteration_1.append(False_a_original_epsilon_df[0])
    False_b_original_iteration_1.append(False_b_original_epsilon_df[0])

    True_a_original_iteration_2.append(True_a_original_epsilon_df[1])
    True_b_original_iteration_2.append(True_b_original_epsilon_df[1])
    False_a_original_iteration_2.append(False_a_original_epsilon_df[1])
    False_b_original_iteration_2.append(False_b_original_epsilon_df[1])

    True_a_original_iteration_3.append(True_a_original_epsilon_df[2])
    True_b_original_iteration_3.append(True_b_original_epsilon_df[2])
    False_a_original_iteration_3.append(False_a_original_epsilon_df[2])
    False_b_original_iteration_3.append(False_b_original_epsilon_df[2])

    True_a_original_iteration_4.append(True_a_original_epsilon_df[3])
    True_b_original_iteration_4.append(True_b_original_epsilon_df[3])
    False_a_original_iteration_4.append(False_a_original_epsilon_df[3])
    False_b_original_iteration_4.append(False_b_original_epsilon_df[3])

    a_t.append(a_t_eps)
    b_t.append(b_t_eps)
    a_f.append(a_f_eps)
    b_f.append(b_f_eps)
    similar_iteration.append(similar_epsilon)
    True_a_iteration_original.append(True_a_original_epsilon)
    True_b_iteration_original.append(True_b_original_epsilon)
    False_a_iteration_original.append(False_a_original_epsilon)
    False_b_iteration_original.append(False_b_original_epsilon)

    True_a_iteration_debiased.append(True_a_debiased_epsilon)
    True_b_iteration_debiased.append(True_b_debiased_epsilon)
    False_a_iteration_debiased.append(False_a_debiased_epsilon)
    False_b_iteration_debiased.append(False_b_debiased_epsilon)

qualified_a_population = sum(a1_times_iterations)
qualified_b_population = sum(b1_times_iterations)

y1 = sum(without_noise_True_a_iteration_original)
x1 = sum(without_noise_True_b_iteration_original)
without_noise_prob_a_original = y1 / float(qualified_a_population)
without_noise_prob_b_original = x1 / float(qualified_b_population)
without_noise_eqo_original = without_noise_prob_a_original - without_noise_prob_b_original
accuracy_original = (sum(without_noise_True_a_iteration_original) + sum(
    without_noise_True_b_iteration_original)) / float(qualified_a_population + qualified_b_population)
accuracy_all_in_ite_original = (sum(without_noise_True_a_iteration_original) + sum(
    without_noise_True_b_iteration_original)) / float(number_of_iterations)
accuracy_a_in_ite_original = (sum(without_noise_True_a_iteration_original)) / float(number_of_iterations)
accuracy_b_in_ite_original = (sum(without_noise_True_b_iteration_original)) / float(number_of_iterations)

y2 = sum(without_noise_True_a_iteration_debiased)
x2 = sum(without_noise_True_b_iteration_debiased)
without_noise_prob_a_debiased = y2 / float(qualified_a_population)
without_noise_prob_b_debiased = x2 / float(qualified_b_population)
without_noise_eqo_debiased = without_noise_prob_a_debiased - without_noise_prob_b_debiased
accuracy_debiased = (sum(without_noise_True_a_iteration_debiased) + sum(
    without_noise_True_b_iteration_debiased)) / float(qualified_a_population + qualified_b_population)
accuracy_all_in_ite_debiased = (sum(without_noise_True_a_iteration_debiased) + sum(
    without_noise_True_b_iteration_debiased)) / float(number_of_iterations)
accuracy_a_in_ite_debiased = (sum(without_noise_True_a_iteration_debiased)) / float(number_of_iterations)
accuracy_b_in_ite_debiased = (sum(without_noise_True_b_iteration_debiased)) / float(number_of_iterations)

print('hi')
data1 = {'Name': ['without noise original'],
         'prob_a': [without_noise_prob_a_original],
         'prob_b': [without_noise_prob_b_original],
         'equal_opportunity': [without_noise_eqo_original],
         'qualified_selected': [accuracy_original],
         'True_selection': [accuracy_all_in_ite_original],
         'True_a_selected_iterations': [accuracy_a_in_ite_original],
         'True_b_selected_iterations': [accuracy_b_in_ite_original]
         }

data2 = {'Name': ['without noise debiased'],
         'prob_a': [without_noise_prob_a_debiased],
         'prob_b': [without_noise_prob_b_debiased],
         'equal_opportunity': [without_noise_eqo_debiased],
         'qualified_selected': [accuracy_debiased],
         'True_selection': [accuracy_all_in_ite_debiased],
         'True_a_selected_iterations': [accuracy_a_in_ite_debiased],
         'True_b_selected_iterations': [accuracy_b_in_ite_debiased]
         }

df1 = pd.DataFrame.from_dict(data1)
df2 = pd.DataFrame.from_dict(data2)
df = pd.concat([df1, df2], ignore_index=True, axis=0)
df.to_csv('wb_without_noise_debiased_vs_original_multiple_selection.csv')
print('hi')

True_a_iteration_1_original_df = pd.DataFrame(True_a_original_iteration_1)
True_b_iteration_1_original_df = pd.DataFrame(True_b_original_iteration_1)
False_a_iteration_1_original_df = pd.DataFrame(False_a_original_iteration_1)
False_b_iteration_1_original_df = pd.DataFrame(False_b_original_iteration_1)

agg_True_a_iteration_1_original_df = True_a_iteration_1_original_df.sum(axis=0)
agg_True_b_iteration_1_original_df = True_b_iteration_1_original_df.sum(axis=0)
agg_False_a_iteration_1_original_df = False_a_iteration_1_original_df.sum(axis=0)
agg_False_b_iteration_1_original_df = False_b_iteration_1_original_df.sum(axis=0)

prob_a_original_1 = agg_True_a_iteration_1_original_df.div(qualified_a_population)
prob_b_original_1 = agg_True_b_iteration_1_original_df.div(qualified_b_population)
equal_opportunity_original_1 = prob_a_original_1 - prob_b_original_1
accuracy_in_all_original_1 = (agg_True_a_iteration_1_original_df + agg_True_b_iteration_1_original_df).div(
    qualified_a_population + qualified_b_population)
accuracy_in_all_iterations_original_1 = (agg_True_a_iteration_1_original_df + agg_True_b_iteration_1_original_df).div(
    number_of_iterations)
accuracy_in_a_iterations_original_1 = (agg_True_a_iteration_1_original_df).div(number_of_iterations)
accuracy_in_b_iterations_original_1 = (agg_True_b_iteration_1_original_df).div(number_of_iterations)

prob_a_original_1.to_csv('wb_prob_a_original_multiple_selection_1.csv')
prob_b_original_1.to_csv('wb_prob_b_original_multiple_selection_1.csv')
equal_opportunity_original_1.to_csv('wb_equal_opportunity_original_multiple_selection_1.csv')
accuracy_in_all_original_1.to_csv('wb_accuracy_in_all_original_multiple_selection_1.csv')
accuracy_in_all_iterations_original_1.to_csv(
    'wb_accuracy_in_all_iterations_original_multiple_selection_1.csv')
accuracy_in_a_iterations_original_1.to_csv(
    'wb_accuracy_in_a_iterations_original_multiple_selection_1.csv')
accuracy_in_b_iterations_original_1.to_csv(
    'wb_accuracy_in_b_iterations_original_multiple_selection_1.csv')

True_a_iteration_2_original_df = pd.DataFrame(True_a_original_iteration_2)
True_b_iteration_2_original_df = pd.DataFrame(True_b_original_iteration_2)
False_a_iteration_2_original_df = pd.DataFrame(False_a_original_iteration_2)
False_b_iteration_2_original_df = pd.DataFrame(False_b_original_iteration_2)

agg_True_a_iteration_2_original_df = True_a_iteration_2_original_df.sum(axis=0)
agg_True_b_iteration_2_original_df = True_b_iteration_2_original_df.sum(axis=0)
agg_False_a_iteration_2_original_df = False_a_iteration_2_original_df.sum(axis=0)
agg_False_b_iteration_2_original_df = False_b_iteration_2_original_df.sum(axis=0)

prob_a_original_2 = agg_True_a_iteration_2_original_df.div(qualified_a_population)
prob_b_original_2 = agg_True_b_iteration_2_original_df.div(qualified_b_population)
equal_opportunity_original_2 = prob_a_original_2 - prob_b_original_2
accuracy_in_all_original_2 = (agg_True_a_iteration_2_original_df + agg_True_b_iteration_2_original_df).div(
    qualified_a_population + qualified_b_population)
accuracy_in_all_iterations_original_2 = (agg_True_a_iteration_2_original_df + agg_True_b_iteration_2_original_df).div(
    number_of_iterations)
accuracy_in_a_iterations_original_2 = (agg_True_a_iteration_2_original_df).div(number_of_iterations)
accuracy_in_b_iterations_original_2 = (agg_True_b_iteration_2_original_df).div(number_of_iterations)

prob_a_original_2.to_csv('wb_prob_a_original_multiple_selection_2.csv')
prob_b_original_2.to_csv('wb_prob_b_original_multiple_selection_2.csv')
equal_opportunity_original_2.to_csv('wb_equal_opportunity_original_multiple_selection_2.csv')
accuracy_in_all_original_2.to_csv('wb_accuracy_in_all_original_multiple_selection_2.csv')
accuracy_in_all_iterations_original_2.to_csv(
    'wb_accuracy_in_all_iterations_original_multiple_selection_2.csv')
accuracy_in_a_iterations_original_2.to_csv(
    'wb_accuracy_in_a_iterations_original_multiple_selection_2.csv')
accuracy_in_b_iterations_original_2.to_csv(
    'wb_accuracy_in_b_iterations_original_multiple_selection_2.csv')


True_a_iteration_3_original_df = pd.DataFrame(True_a_original_iteration_3)
True_b_iteration_3_original_df = pd.DataFrame(True_b_original_iteration_3)
False_a_iteration_3_original_df = pd.DataFrame(False_a_original_iteration_3)
False_b_iteration_3_original_df = pd.DataFrame(False_b_original_iteration_3)

agg_True_a_iteration_3_original_df = True_a_iteration_3_original_df.sum(axis=0)
agg_True_b_iteration_3_original_df = True_b_iteration_3_original_df.sum(axis=0)
agg_False_a_iteration_3_original_df = False_a_iteration_3_original_df.sum(axis=0)
agg_False_b_iteration_3_original_df = False_b_iteration_3_original_df.sum(axis=0)

prob_a_original_3 = agg_True_a_iteration_3_original_df.div(qualified_a_population)
prob_b_original_3 = agg_True_b_iteration_3_original_df.div(qualified_b_population)
equal_opportunity_original_3 = prob_a_original_3 - prob_b_original_3
accuracy_in_all_original_3 = (agg_True_a_iteration_3_original_df + agg_True_b_iteration_3_original_df).div(
    qualified_a_population + qualified_b_population)
accuracy_in_all_iterations_original_3 = (agg_True_a_iteration_3_original_df + agg_True_b_iteration_3_original_df).div(
    number_of_iterations)
accuracy_in_a_iterations_original_3 = (agg_True_a_iteration_3_original_df).div(number_of_iterations)
accuracy_in_b_iterations_original_3 = (agg_True_b_iteration_3_original_df).div(number_of_iterations)

prob_a_original_3.to_csv('wb_prob_a_original_multiple_selection_3.csv')
prob_b_original_3.to_csv('wb_prob_b_original_multiple_selection_3.csv')
equal_opportunity_original_3.to_csv('wb_equal_opportunity_original_multiple_selection_3.csv')
accuracy_in_all_original_3.to_csv('wb_accuracy_in_all_original_multiple_selection_3.csv')
accuracy_in_all_iterations_original_3.to_csv(
    'wb_accuracy_in_all_iterations_original_multiple_selection_3.csv')
accuracy_in_a_iterations_original_3.to_csv(
    'wb_accuracy_in_a_iterations_original_multiple_selection_3.csv')
accuracy_in_b_iterations_original_3.to_csv(
    'wb_accuracy_in_b_iterations_original_multiple_selection_3.csv')


True_a_iteration_4_original_df = pd.DataFrame(True_a_original_iteration_4)
True_b_iteration_4_original_df = pd.DataFrame(True_b_original_iteration_4)
False_a_iteration_4_original_df = pd.DataFrame(False_a_original_iteration_4)
False_b_iteration_4_original_df = pd.DataFrame(False_b_original_iteration_4)

agg_True_a_iteration_4_original_df = True_a_iteration_4_original_df.sum(axis=0)
agg_True_b_iteration_4_original_df = True_b_iteration_4_original_df.sum(axis=0)
agg_False_a_iteration_4_original_df = False_a_iteration_4_original_df.sum(axis=0)
agg_False_b_iteration_4_original_df = False_b_iteration_4_original_df.sum(axis=0)

prob_a_original_4 = agg_True_a_iteration_4_original_df.div(qualified_a_population)
prob_b_original_4 = agg_True_b_iteration_4_original_df.div(qualified_b_population)
equal_opportunity_original_4 = prob_a_original_4 - prob_b_original_4
accuracy_in_all_original_4 = (agg_True_a_iteration_4_original_df + agg_True_b_iteration_4_original_df).div(
    qualified_a_population + qualified_b_population)
accuracy_in_all_iterations_original_4 = (agg_True_a_iteration_4_original_df + agg_True_b_iteration_4_original_df).div(
    number_of_iterations)
accuracy_in_a_iterations_original_4 = (agg_True_a_iteration_4_original_df).div(number_of_iterations)
accuracy_in_b_iterations_original_4 = (agg_True_b_iteration_4_original_df).div(number_of_iterations)

prob_a_original_4.to_csv('wb_prob_a_original_multiple_selection_4.csv')
prob_b_original_4.to_csv('wb_prob_b_original_multiple_selection_4.csv')
equal_opportunity_original_4.to_csv('wb_equal_opportunity_original_multiple_selection_4.csv')
accuracy_in_all_original_4.to_csv('wb_accuracy_in_all_original_multiple_selection_4.csv')
accuracy_in_all_iterations_original_4.to_csv(
    'wb_accuracy_in_all_iterations_original_multiple_selection_4.csv')
accuracy_in_a_iterations_original_4.to_csv(
    'wb_accuracy_in_a_iterations_original_multiple_selection_4.csv')
accuracy_in_b_iterations_original_4.to_csv(
    'wb_accuracy_in_b_iterations_original_multiple_selection_4.csv')


True_a_iteration_1_df = pd.DataFrame(True_a_debiased_iteration_1)
True_b_iteration_1_df = pd.DataFrame(True_b_debiased_iteration_1)
agg_True_a_iteration_1_df = True_a_iteration_1_df.sum(axis=0)
agg_True_b_iteration_1_df = True_b_iteration_1_df.sum(axis=0)
prob_a_1 = agg_True_a_iteration_1_df.div(qualified_a_population)
prob_b_1 = agg_True_b_iteration_1_df.div(qualified_b_population)
equal_opportunity_1 = prob_a_1 - prob_b_1
accuracy_in_all_1 = (agg_True_a_iteration_1_df+agg_True_b_iteration_1_df).div(qualified_a_population+qualified_b_population)
accuracy_in_all_iterations_1 = (agg_True_a_iteration_1_df+agg_True_b_iteration_1_df).div(number_of_iterations)
accuracy_in_a_iterations_1 = (agg_True_a_iteration_1_df).div(number_of_iterations)
accuracy_in_b_iterations_1 = (agg_True_b_iteration_1_df).div(number_of_iterations)


prob_a_1.to_csv('wb_prob_a_debiased_multiple_selection_1.csv')
prob_b_1.to_csv('wb_prob_b_debiased__multiple_selection_1.csv')
equal_opportunity_1.to_csv('wb_equal_opportunity_debiased_multiple_selection_1.csv')
accuracy_in_all_1.to_csv('wb_accuracy_debiased_multiple_selection_1.csv')
accuracy_in_all_iterations_1.to_csv(
    'wb_accuracy_in_all_iterations_debiased_multiple_selection_1.csv')
accuracy_in_a_iterations_1.to_csv(
    'wb_accuracy_in_a_iterations_debiased_multiple_selection_1.csv')
accuracy_in_b_iterations_1.to_csv(
    'wb_accuracy_in_b_iterations_debiased_multiple_selection_1.csv')


True_a_iteration_2_df = pd.DataFrame(True_a_debiased_iteration_2)
True_b_iteration_2_df = pd.DataFrame(True_b_debiased_iteration_2)
agg_True_a_iteration_2_df = True_a_iteration_2_df.sum(axis=0)
agg_True_b_iteration_2_df = True_b_iteration_2_df.sum(axis=0)
prob_a_2 = agg_True_a_iteration_2_df.div(qualified_a_population)
prob_b_2 = agg_True_b_iteration_2_df.div(qualified_b_population)
equal_opportunity_2 = prob_a_2 - prob_b_2
accuracy_in_all_2 = (agg_True_a_iteration_2_df+agg_True_b_iteration_2_df).div(qualified_a_population+qualified_b_population)
accuracy_in_all_iterations_2 = (agg_True_a_iteration_2_df+agg_True_b_iteration_2_df).div(number_of_iterations)
accuracy_in_a_iterations_2 = (agg_True_a_iteration_2_df).div(number_of_iterations)
accuracy_in_b_iterations_2 = (agg_True_b_iteration_2_df).div(number_of_iterations)


prob_a_2.to_csv('wb_prob_a_debiased_multiple_selection_2.csv')
prob_b_2.to_csv('wb_prob_b_debiased__multiple_selection_2.csv')
equal_opportunity_2.to_csv('wb_equal_opportunity_debiased_multiple_selection_2.csv')
accuracy_in_all_2.to_csv('wb_accuracy_debiased_multiple_selection_2.csv')
accuracy_in_all_iterations_2.to_csv(
    'wb_accuracy_in_all_iterations_debiased_multiple_selection_2.csv')
accuracy_in_a_iterations_2.to_csv(
    'wb_accuracy_in_a_iterations_debiased_multiple_selection_2.csv')
accuracy_in_b_iterations_2.to_csv(
    'wb_accuracy_in_b_iterations_debiased_multiple_selection_2.csv')


True_a_iteration_3_df = pd.DataFrame(True_a_debiased_iteration_3)
True_b_iteration_3_df = pd.DataFrame(True_b_debiased_iteration_3)
agg_True_a_iteration_3_df = True_a_iteration_3_df.sum(axis=0)
agg_True_b_iteration_3_df = True_b_iteration_3_df.sum(axis=0)
prob_a_3 = agg_True_a_iteration_3_df.div(qualified_a_population)
prob_b_3 = agg_True_b_iteration_3_df.div(qualified_b_population)
equal_opportunity_3 = prob_a_3 - prob_b_3
accuracy_in_all_3 = (agg_True_a_iteration_3_df+agg_True_b_iteration_3_df).div(qualified_a_population+qualified_b_population)
accuracy_in_all_iterations_3 = (agg_True_a_iteration_3_df+agg_True_b_iteration_3_df).div(number_of_iterations)
accuracy_in_a_iterations_3 = (agg_True_a_iteration_3_df).div(number_of_iterations)
accuracy_in_b_iterations_3 = (agg_True_b_iteration_3_df).div(number_of_iterations)


prob_a_3.to_csv('wb_prob_a_debiased_multiple_selection_3.csv')
prob_b_3.to_csv('wb_prob_b_debiased__multiple_selection_3.csv')
equal_opportunity_3.to_csv('wb_equal_opportunity_debiased_multiple_selection_3.csv')
accuracy_in_all_3.to_csv('wb_accuracy_debiased_multiple_selection_3.csv')
accuracy_in_all_iterations_3.to_csv(
    'wb_accuracy_in_all_iterations_debiased_multiple_selection_3.csv')
accuracy_in_a_iterations_3.to_csv(
    'wb_accuracy_in_a_iterations_debiased_multiple_selection_3.csv')
accuracy_in_b_iterations_3.to_csv(
    'wb_accuracy_in_b_iterations_debiased_multiple_selection_3.csv')

True_a_iteration_4_df = pd.DataFrame(True_a_debiased_iteration_4)
True_b_iteration_4_df = pd.DataFrame(True_b_debiased_iteration_4)
agg_True_a_iteration_4_df = True_a_iteration_4_df.sum(axis=0)
agg_True_b_iteration_4_df = True_b_iteration_4_df.sum(axis=0)
prob_a_4 = agg_True_a_iteration_4_df.div(qualified_a_population)
prob_b_4 = agg_True_b_iteration_4_df.div(qualified_b_population)
equal_opportunity_4 = prob_a_4 - prob_b_4
accuracy_in_all_4 = (agg_True_a_iteration_4_df+agg_True_b_iteration_4_df).div(qualified_a_population+qualified_b_population)
accuracy_in_all_iterations_4 = (agg_True_a_iteration_4_df+agg_True_b_iteration_4_df).div(number_of_iterations)
accuracy_in_a_iterations_4 = (agg_True_a_iteration_4_df).div(number_of_iterations)
accuracy_in_b_iterations_4 = (agg_True_b_iteration_4_df).div(number_of_iterations)

prob_a_4.to_csv('wb_prob_a_debiased_multiple_selection_4.csv')
prob_b_4.to_csv('wb_prob_b_debiased__multiple_selection_4.csv')
equal_opportunity_4.to_csv('wb_equal_opportunity_debiased_multiple_selection_4.csv')
accuracy_in_all_4.to_csv('wb_accuracy_debiased_multiple_selection_4.csv')
accuracy_in_all_iterations_4.to_csv(
    'wb_accuracy_in_all_iterations_debiased_multiple_selection_4.csv')
accuracy_in_a_iterations_4.to_csv(
    'wb_accuracy_in_a_iterations_debiased_multiple_selection_4.csv')
accuracy_in_b_iterations_4.to_csv(
    'wb_accuracy_in_b_iterations_debiased_multiple_selection_4.csv')
