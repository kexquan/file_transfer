import pdb

import numpy as np
import pandas as pd
import seaborn as sns

from dataLoader import load_ADReSS2020_train_data
from dataLoader import load_AD2021_train_data
from dataLoader import load_jccocc_moca_data
from dataLoader import load_feature_selection_repository
from dataLoader import load_sythetic_data
from dataLoader import load_realworld_dataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplotlib.pyplot as plt

from torch.multiprocessing import Pool
from itertools import repeat
from sklearn.utils import shuffle

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE

from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker

def classification(X_train, y_train, X_test, y_test):
    clf = classifier()
    clf.fit(X_train, y_train)
    pre_dict_label = clf.predict(X_test)
    predict_pro_trn = clf.predict_proba(X_train)
    predict_pro_tst = clf.predict_proba(X_test)
    # return accuracy_score(y_test, pre_dict_label)
    return pre_dict_label, predict_pro_trn, predict_pro_tst

def Standardize(X_train, X_test):
    std = StandardScaler().fit(X_train)
    X_train = std.transform(X_train)
    X_test = std.transform(X_test)
    return X_train, X_test

def obtain_index(random_state, subject_id):

    subject_unique_id = np.unique(subject_id)

    id_list = np.array_split(shuffle(subject_unique_id, random_state=random_state), 10)
    train_index_all = []
    test_index_all = []
    for ids in id_list:
        test_index = np.concatenate([np.where(id == subject_id)[0] for id in ids])
        trn_index = np.delete(range(len(subject_id)), test_index)
        test_index_all.append(test_index)
        train_index_all.append(trn_index)
        assert len(np.unique(subject_id[test_index])) + len(np.unique(subject_id[trn_index])) == len(subject_unique_id)

    return train_index_all, test_index_all


def run(trn_feats_all,
        trn_labels_all,
        random_state,
        subject_id,
        consider_speaker,
        average,
        datasetfold,
        data,
        method):

    acc_all_folds = []
    pre_all_folds = []
    rec_all_folds = []
    f1__all_folds = []

    # for random_state in random_states:

    save_base = 'results/'

    # file1 = datasetfold + '/' + data + '_' + method + '_randomstate_' + str(random_state) + ".npz"
    file1 = datasetfold + '/' + data + '_' + '_'.join(method) + '_randomstate_' + str(random_state) + ".npz"
    save_path1 = save_base + file1
    results = np.load(save_path1, allow_pickle=True)['arr_0'].item()

    if consider_speaker:
        train_index_all, test_index_all = obtain_index(random_state, subject_id)
    else:
        cv_outer = KFold(n_splits=len(results['index_selected']), shuffle=True, random_state=random_state)
        train_index_all = []
        test_index_all = []
        for train_index, test_index in cv_outer.split(trn_feats_all):
            train_index_all.append(train_index)
            test_index_all.append(test_index)

    acc_10_folds = []
    pre_10_folds = []
    rec_10_folds = []
    f1_10_folds = []

    for i in range(len(train_index_all)):

        acc_all_num_fs = []
        pre_all_num_fs = []
        rec_all_num_fs = []
        f1_all_num_fs = []

        for ii in range(len(results['index_selected'][0])):

            index_selected = results['index_selected'][i][ii]

            trn_index = train_index_all[i]
            tst_index = test_index_all[i]

            training_feats = trn_feats_all[trn_index, :][:, index_selected]
            testing_feats = trn_feats_all[tst_index, :][:, index_selected]

            training_feats, testing_feats = Standardize(training_feats, testing_feats)

            pre_labels, predict_pro_trn, predict_pro_tst = classification(training_feats, trn_labels_all[trn_index], testing_feats, trn_labels_all[tst_index])

            acc, pre, rec, f1_ = all_score(trn_labels_all[tst_index], pre_labels, average)

            # try:
            #     assert results['accuracy'][i][ii] == acc
            # except:
            #     print(method)
            #     print('Fold:', i)
            #     print('Number of selected features:', results['num_fs'][0][ii])
            #     print(results['accuracy'][i][ii])
            #     print(acc)

            acc_all_num_fs.append(acc)
            pre_all_num_fs.append(pre)
            rec_all_num_fs.append(rec)
            f1_all_num_fs.append(f1_)

        acc_10_folds.append(acc_all_num_fs)
        pre_10_folds.append(pre_all_num_fs)
        rec_10_folds.append(rec_all_num_fs)
        f1_10_folds.append(f1_all_num_fs)

    return np.array(acc_10_folds), np.array(pre_10_folds), np.array(rec_10_folds), np.array(f1_10_folds), results['num_fs'][0]

            # acc_all_folds.append(acc_all_num_fs)
            # pre_all_folds.append(pre_all_num_fs)
            # rec_all_folds.append(rec_all_num_fs)
            # f1__all_folds.append(f1_all_num_fs)

    # return np.array(acc_all_folds), np.array(pre_all_folds), np.array(rec_all_folds), np.array(f1__all_folds), results['num_fs'][0]

def syethetic_run(trn_feats_all,
        trn_labels_all,
        random_states,
        subject_id,
        consider_speaker,
        average,
        datasetfold,
        data,
        method):

    acc_all_folds = []
    pre_all_folds = []
    rec_all_folds = []
    f1__all_folds = []

    index_selected_all_folds = []
    importances_all_folds = []

    for random_state in random_states:

        save_base = 'results/'

        file1 = datasetfold + '/' + data + '_' + method + '_randomstate_' + str(random_state) + ".npz"
        save_path1 = save_base + file1
        results = np.load(save_path1, allow_pickle=True)['arr_0'].item()

        if consider_speaker:
            train_index_all, test_index_all = obtain_index(random_state, subject_id)
        else:
            cv_outer = KFold(n_splits=len(results['index_selected']), shuffle=True, random_state=random_state)
            train_index_all = []
            test_index_all = []
            for train_index, test_index in cv_outer.split(trn_feats_all):
                train_index_all.append(train_index)
                test_index_all.append(test_index)


        for i in range(len(train_index_all)):

            acc_all_num_fs = []
            pre_all_num_fs = []
            rec_all_num_fs = []
            f1_all_num_fs = []

            for ii in range(len(results['index_selected'][0])):

                index_selected = results['index_selected'][i][ii]

                index_selected_all_folds.append(index_selected)
                importances_all_folds.append(results['importances'][i][ii])

                trn_index = train_index_all[i]
                tst_index = test_index_all[i]

                training_feats = trn_feats_all[trn_index, :][:, index_selected]
                testing_feats = trn_feats_all[tst_index, :][:, index_selected]

                training_feats, testing_feats = Standardize(training_feats, testing_feats)

                pre_labels, predict_pro_trn, predict_pro_tst = classification(training_feats, trn_labels_all[trn_index], testing_feats, trn_labels_all[tst_index])

                acc, pre, rec, f1_ = all_score(trn_labels_all[tst_index], pre_labels, average)

                # try:
                #     assert results['accuracy'][i][ii] == acc
                # except:
                #     print(method)
                #     print('Fold:', i)
                #     print('Number of selected features:', results['num_fs'][0][ii])
                #     print(results['accuracy'][i][ii])
                #     print(acc)

                acc_all_num_fs.append(acc)
                pre_all_num_fs.append(pre)
                rec_all_num_fs.append(rec)
                f1_all_num_fs.append(f1_)

            acc_all_folds.append(acc_all_num_fs)
            pre_all_folds.append(pre_all_num_fs)
            rec_all_folds.append(rec_all_num_fs)
            f1__all_folds.append(f1_all_num_fs)

    return acc_all_folds, index_selected_all_folds, importances_all_folds, results['num_fs'][0]

def plot_with_variance(reward_mean, reward_var, color='yellow', savefig_dir=None):
    """plot_with_variance
        reward_mean: typr list, containing all the means of reward summmary scalars collected during training
        reward_var: type list, containing all variance
        savefig_dir: if not None, this must be a str representing the directory to save the figure
    """
    lower = [x - y for x, y in zip(reward_mean, reward_var)]
    upper = [x + y for x, y in zip(reward_mean, reward_var)]
    plt.figure()
    xaxis = list(range(1, len(lower) + 1))
    plt.plot(xaxis, reward_mean, '--o', markersize=2.5, color=color)
    plt.fill_between(xaxis, lower, upper, color=color, alpha=0.2)
    plt.grid()
    plt.xlabel('Number of features in feature subset', fontsize=15)
    plt.ylabel('Recognition accuracy', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if savefig_dir is not None and type(savefig_dir) is str:
        plt.savefig(savefig_dir, format='jpg')
    plt.show()


def obtain_performance_on_test_data(method, accuracy_trn_all):

    accuracy_trn_max = []
    precision_trn_max = []
    recall_trn_max = []
    f1_trn_max = []

    ## All folds
    index_selected_all = []
    predicted_acc_all = []
    predicted_pre_all = []
    predicted_rec_all = []
    predicted_f1_all = []

    # union
    accuracy_test_union = []
    precision_test_union = []
    recall_test_union = []
    f1_test_union = []

    ## vote
    accuracy_test_vote = []
    precision_test_vote = []
    recall_test_vote = []
    f1_test_vote = []

    ## Stack
    accuracy_test = []
    precision_test = []
    recall_test = []
    f1_test = []

    #################################################################
    for i in range(len(random_states)):

        random_state = random_states[i]

        file = 'results/' + datasetfold + '/' + data + '_' + '_'.join(method) + '_randomstate_' + str(random_state) + ".npz"
        results = np.load(file, allow_pickle=True)['arr_0'].item()

        predict_probablity_trn_all = []
        predict_probablity_tst_all = []

        labels_tst_10_folds = []

        index_selected_10_fols = []

        for j in range(10):

            if same_nfs:
                n_fs = np.concatenate(accuracy_trn_all).mean(axis=0).argmax()
            else:
                n_fs = np.array(accuracy_trn_all)[i].mean(axis=0).argmax()

            index_selected = results['index_selected'][j][n_fs]

            index_selected_10_fols.append(index_selected)

            index_selected_all.append(index_selected)

            trn_feats = trn_feats_all[:, index_selected]
            tst_feats = tst_feats_all[:, index_selected]
            trn_feats, tst_feats = Standardize(trn_feats, tst_feats)
            predict_label, pro_trn, pro_tst = classification(trn_feats, trn_labels_all, tst_feats, tst_labels_all)

            # append labels
            labels_tst_10_folds.append(predict_label)

            # append scores
            predict_probablity_trn_all.append(pro_trn)
            predict_probablity_tst_all.append(pro_tst)

            ## All folds accuracy
            acc__, pre__, rec__, f1__ = all_score(tst_labels_all, predict_label, average)
            predicted_acc_all.append(acc__)
            predicted_pre_all.append(pre__)
            predicted_rec_all.append(rec__)
            predicted_f1_all.append(f1__)

        ## Union
        index_selected_unique = np.unique(np.concatenate((index_selected_10_fols)))

        print('Number of selected features:', len(index_selected_unique))

        trn_feats_u = trn_feats_all[:, index_selected_unique]
        tst_feats_u = tst_feats_all[:, index_selected_unique]
        trn_feats_u, tst_feats_u = Standardize(trn_feats_u, tst_feats_u)
        predict_label_union, _, _ = classification(trn_feats_u, trn_labels_all, tst_feats_u, tst_labels_all)

        ## majority voting
        labels_tst_10_folds = np.array(labels_tst_10_folds)
        vote_labels = np.array([pd.value_counts(labels_tst_10_folds[:, i]).index[0] for i in range(labels_tst_10_folds.shape[1])])

        ## stacking
        trn_pro = np.concatenate((predict_probablity_trn_all), axis=1)
        tst_pro = np.concatenate((predict_probablity_tst_all), axis=1)
        clf = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000).fit(trn_pro, trn_labels_all)
        predicted_labels1 = clf.predict(tst_pro)

        if not tst_voting:
            # append stack acc
            acc, pre, rec, f1 = all_score(tst_labels_all, predicted_labels1, average)
            accuracy_test.append(acc)
            precision_test.append(pre)
            recall_test.append(rec)
            f1_test.append(f1)

            # append vote acc
            acc_vote, pre_vote, rec_vote, f1_vote = all_score(tst_labels_all, vote_labels, average)
            accuracy_test_vote.append(acc_vote)
            precision_test_vote.append(pre_vote)
            recall_test_vote.append(rec_vote)
            f1_test_vote.append(f1_vote)

            # append union acc
            acc_union, pre_union, rec_union, f1_union = all_score(tst_labels_all, predict_label_union, average)
            accuracy_test_union.append(acc_union)
            precision_test_union.append(pre_union)
            recall_test_union.append(rec_union)
            f1_test_union.append(f1_union)

        # else:
        #     tst_id_unique = np.unique(tst_subject_id)
        #     vote_labels = []
        #     true_labes = []
        #     for id in tst_id_unique:
        #         vote_label = pd.value_counts(predicted_labels1[np.where(tst_subject_id == id)[0]]).index[0]
        #         tru_label = tst_labels_all[np.where(tst_subject_id == id)[0]][0]
        #         vote_labels.append(vote_label)
        #         true_labes.append(tru_label)
        #     acc, pre, rec, f1 = all_score(true_labes, vote_labels, average)
        #     accuracy_test.append(acc)
        #     precision_test.append(pre)
        #     recall_test.append(rec)
        #     f1_test.append(f1)

    print('Before Stacking')
    print('test set: averaged accuracy: %.3f \pm %.3f' % (np.mean(predicted_acc_all), np.std(predicted_acc_all)))
    print('test set: averaged precision: %.3f \pm %.3f' % (np.mean(predicted_pre_all), np.std(predicted_pre_all)))
    print('test set: averaged recall: %.3f \pm %.3f' % (np.mean(predicted_rec_all), np.std(predicted_rec_all)))
    print('test set: averaged f1 score: %.3f \pm %.3f' % (np.mean(predicted_f1_all), np.std(predicted_f1_all)))

    print('Union')
    print('test set: averaged accuracy: %.3f \pm %.3f' % (np.mean(accuracy_test_union), np.std(accuracy_test_union)))
    print('test set: averaged precision: %.3f \pm %.3f' % (np.mean(precision_test_union), np.std(precision_test_union)))
    print('test set: averaged recall: %.3f \pm %.3f' % (np.mean(recall_test_union), np.std(recall_test_union)))
    print('test set: averaged f1 score: %.3f \pm %.3f' % (np.mean(f1_test_union), np.std(f1_test_union)))

    print('majority voting')
    print('test set: averaged accuracy: %.3f \pm %.3f' % (np.mean(accuracy_test_vote), np.std(accuracy_test_vote)))
    print('test set: averaged precision: %.3f \pm %.3f' % (np.mean(precision_test_vote), np.std(precision_test_vote)))
    print('test set: averaged recall: %.3f \pm %.3f' % (np.mean(recall_test_vote), np.std(recall_test_vote)))
    print('test set: averaged f1 score: %.3f \pm %.3f' % (np.mean(f1_test_vote), np.std(f1_test_vote)))

    print('Stacking')
    print('test set: averaged accuracy: %.3f \pm %.3f' % (np.mean(accuracy_test), np.std(accuracy_test)))
    print('test set: averaged precision: %.3f \pm %.3f' % (np.mean(precision_test), np.std(precision_test)))
    print('test set: averaged recall: %.3f \pm %.3f' % (np.mean(recall_test), np.std(recall_test)))
    print('test set: averaged f1 score: %.3f \pm %.3f' % (np.mean(f1_test), np.std(f1_test)))

    return predicted_acc_all, accuracy_test, index_selected_all

def pearson_correlation_test(trn_feats, trn_labels):
    name = ['feat_' + str(i) for i in range(trn_feats.shape[1])]
    name.append('target')
    name = np.array(name)
    data = np.concatenate((trn_feats, trn_labels.reshape(-1,1)), axis=1)
    df = pd.DataFrame(data, columns=name)
    pearson_corr_df = df.corr(method='pearson')
    pearson_corr_df = pearson_corr_df['target'].reset_index()
    pearson_corr_df.columns = ['STATISTIC', 'PEARSON_CORRELATION']
    pearson_corr_df['PEARSON_CORRELATION_ABS'] = abs(pearson_corr_df['PEARSON_CORRELATION'])
    importances = np.array(pearson_corr_df['PEARSON_CORRELATION'][:-1])
    return importances

######################################
def all_score(true_label, predict_label, average):

    acc = accuracy_score(true_label, predict_label)
    pre = precision_score(true_label, predict_label, average=average)
    rec = recall_score(true_label, predict_label, average=average)
    f1 = f1_score(true_label, predict_label, average=average)

    return acc, pre, rec, f1

def classifier():
    return SVC(kernel='rbf', C=1, probability=True, gamma='auto', random_state=0)

if __name__ == '__main__':

    datasetfold = "synthetic"

    datas = ['binary_classification']

    methods = ['NetAct_FIR', 'DFS', 'dropoutFR', 'DeepFIR', 'DDR', 'lasso_path', 'RF', 'FS', 'BS', 'RFE', 'fdr', 'ANOVA', 'PeaCorr', 'MutInfo', 'mRMR', 'CCM']
    # methods = ['NetAct_FIR']

    random_states = list(range(1))

    same_nfs = False

    consider_speaker = False

    trn_voting = False
    tst_voting = False
    average = 'macro'
    index_step1 = 49
    identifier = None

########################### syethetic
    for data in datas:

        if datasetfold == "AD2021_journal2" or datasetfold == "AD2021_journal2_0":
            trn_feats_all, trn_labels_all, tst_feats_all, tst_labels_all, trn_subject_id, tst_subject_id = load_AD2021_train_data()
        elif datasetfold == "ADReSS_journal2" or datasetfold == "ADReSS_journal2_0":
            trn_feats_all, trn_labels_all, tst_feats_all, tst_labels_all, trn_subject_id, tst_subject_id = load_ADReSS2020_train_data()
        elif datasetfold == 'jccocc_moca':
            trn_feats_all, trn_labels_all, tst_feats_all, tst_labels_all, trn_subject_id, tst_subject_id = load_jccocc_moca_data()
        elif datasetfold == 'feature_selection_repository':
            trn_feats_all, trn_labels_all, tst_feats_all, tst_labels_all, trn_subject_id, tst_subject_id = load_feature_selection_repository(data)
        elif datasetfold == 'synthetic':
            trn_feats_all, trn_labels_all, tst_feats_all, tst_labels_all, trn_subject_id, tst_subject_id = load_sythetic_data(1024, 1024, data)
        elif datasetfold == 'realworld':
            trn_feats_all, trn_labels_all, tst_feats_all, tst_labels_all, trn_subject_id, tst_subject_id = load_realworld_dataset(data)

        print('\n')
        print('%s: (%d, %d)' % (data, trn_feats_all.shape[0], trn_feats_all.shape[1]))

        for method in methods:
            print('method:', method)
            acc_all_folds, index_selected_all_folds, importances_all_folds, num_fs = syethetic_run(trn_feats_all, trn_labels_all, random_states, trn_subject_id, consider_speaker, average, datasetfold, data, method)

            count = pd.value_counts(np.concatenate((index_selected_all_folds)))
            count_index = np.array(count.index)
            count_values = np.array(count.values)
            count_values_sorted = count_values[np.argsort(count_index)]
            count_index_sorted = count_index[np.argsort(count_index)]
            complete_index = []
            complete_values = []
            j=0
            for i in range(10):
                complete_index.append(i)
                if i in count_index_sorted:
                    complete_values.append(count_values_sorted[j])
                    j = j + 1
                else:
                    complete_values.append(0)
            colours= []
            for i in range(10):
                if complete_values[i] == 5:
                    colours.append('r')
                else:
                    colours.append('g')

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            if importances_all_folds[0] == []:
                plt.bar(x=range(10), height=complete_values, align="center", color=colours)
                plt.title(method + ' (' + str(round(np.mean(acc_all_folds),3)) + ' ' + str(round(np.std(acc_all_folds),3)) +')')
                plt.xlabel('Feature Index')
                plt.savefig(data + '_' + method + '.jpg')
                plt.show()
            else:
                importance_mean = np.array(importances_all_folds).mean(axis=0).flatten()

                if method == 'DDR':
                    importance_mean = sigmoid(importance_mean)

                plt.bar(x=range(10), height=importance_mean, align="center", color=colours)
                plt.title(method + ' (' + str(round(np.mean(acc_all_folds),3)) + ' ' + str(round(np.std(acc_all_folds),3)) +')')
                plt.xlabel('Feature Index')
                # plt.savefig(data + '_' + method + '.jpg')
                plt.show()