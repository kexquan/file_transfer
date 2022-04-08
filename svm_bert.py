import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from dataLoader import load_AD2021_train_data
from dataLoader import load_ADReSS2020_train_data
from dataLoader import load_jccocc_moca_data
from dataLoader import load_jccocc_moca_data_2022_03_11
from dataLoader import load_jccocc_moca_data_2022_03_19

from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')

class Classifier():
    def __init__(
        self,
        mode='svm',
        pre='pca',
        C=1,
        kernel='rbf',
        n_components=None,
    ):
        self.mode = mode
        self.pre = pre
        self.C = C
        self.kernel = kernel
        self.n_components = n_components

    def fit(self, X, y, shuffle=True):
        if shuffle:
            pid = np.random.permutation(len(y))
            X, y = X[pid], y[pid]
        pipe = []
        if self.pre == 'sc':
            pipe.append(('scaler', StandardScaler()))
            # pipe.append(('scaler', RobustScaler()))
        elif self.pre == 'pca':
            pipe.append(('scaler', StandardScaler()))
            # pipe.append(('scaler', RobustScaler()))
            # pipe.append(('pca', PCA(n_components=self.n_components)))
        elif self.pre == 'robust':
            pipe.append(('scaler', RobustScaler()))

        if self.mode == 'svm':
            pipe.append(('cls', SVC(C=self.C, kernel=self.kernel, probability=True)))
            # pipe.append(('cls', RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced")))
        elif self.mode == 'lda':
            pipe.append(('cls', LDA()))
        self.pipe = Pipeline(pipe)
        self.pipe.fit(X, y)

    def test(self, X, y):
        scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        preds = self.pipe.predict(X)
        # logits = self.pipe.predict_proba(X)

        # if len(np.unique(y)) >2:
        #     scores['acc'] = accuracy_score(y, preds)
        #     scores['prec'] = precision_score(y, preds, average='macro')
        #     scores['rec'] = recall_score(y, preds, average='macro')
        #     scores['f1'] = f1_score(y, preds, average='macro')
        # else:
        #     scores['acc'] = accuracy_score(y, preds)
        #     scores['prec'] = precision_score(y, preds)
        #     scores['rec'] = recall_score(y, preds)
        #     scores['f1'] = f1_score(y, preds)

        scores['acc'] = accuracy_score(y, preds)
        # scores['acc'] = balanced_accuracy_score(y, preds)
        scores['prec'] = precision_score(y, preds, average='macro')
        scores['rec'] = recall_score(y, preds, average='macro')
        scores['f1'] = f1_score(y, preds, average='macro')

        return scores

    def cross_validate(self, X, y, cv=5, times=None):
        scores = {}
        times = 1 if times is None else times
        for _ in range(times):
            pid = np.random.permutation(len(y))
            X, y = X[pid], y[pid]
            cv = len(y) if cv == -1 else cv
            kf = KFold(n_splits=cv, shuffle=True)
            for trn_idx, tst_idx in kf.split(X):
                self.fit(X[trn_idx], y[trn_idx])
                tst_scores = self.test(X[tst_idx], y[tst_idx])
                for key, val in tst_scores.items():
                    if key not in scores.keys():
                        scores[key] = []
                    scores[key].append(val)
        for key, val in scores.items():
            scores[key] = np.asarray(val)
        return scores

    def fit_test(self, X_trn, y_trn, X_tst, y_tst, times=25):
        scores = {}
        for _ in range(times):
            self.fit(X_trn, y_trn)
            tst_scores = self.test(X_tst, y_tst)
            for key, val in tst_scores.items():
                if key not in scores.keys():
                    scores[key] = []
                scores[key].append(val)
        for key, val in scores.items():
            scores[key] = np.asarray(val)
        return scores

    def cv_test(self, features, labels, tst_features, test_ad, cv=5, times=5, testtimes=1):
        scores = self.cross_validate(features, labels, cv=cv, times=times)
        print(". \tacc\t      prec\t    rec\t\t  f1")
        print("cv{}:".format(cv), self.scores_str(scores))
        # print("All accs:"+(("\n"+" {:.2f}"*cv)*times).format(*scores['acc']))
        tst_scores = self.fit_test(features, labels, tst_features, test_ad, times=testtimes)
        print("tst:", self.scores_str(tst_scores))
        return scores, tst_scores

    def scores_str(self, scores):
        return "{:.3f} ({:.3f}) & {:.3f} ({:.3f}) & {:.3f} ({:.3f}) & {:.3f} ({:.3f})".format(
            scores['acc'].mean(), scores['acc'].std(), scores['prec'].mean(), scores['prec'].std(),
            scores['rec'].mean(), scores['rec'].std(), scores['f1'].mean(), scores['f1'].std())

    def print_scores(self, scores, mode='test'):
        if mode == 'test':
            print(("{:<6}"*4).format('acc', 'prec', 'rec', 'f1'))
            table = []
            for _, val in scores.items():
                table.append("{:<6}".format("{:.2f}".format(val)))
            print(''.join(table))
        else:
            print(("{:<12}"*4).format('acc', 'prec', 'rec', 'f1'))
            table = []
            for _, val in scores.items():
                table.append("{:<12}".format("{:.2f}({:.2f})".format(val.mean(), val.std())))
            print(''.join(table))

import os
def find_files(path, ext="", prefix=""):
    return [os.path.join(path, x) for x in os.listdir(path) if x.endswith(ext) and x.startswith(prefix)]
import pandas as pd

def load_test_labels(file):
    test_df_true = pd.read_csv(file)  # Read testing set meta data
    test_subject_id2 = test_df_true[test_df_true.columns.values[0]].apply(lambda x: x.split(';')[0]).values
    test_subject_labels = test_df_true[test_df_true.columns.values[0]].apply(
        lambda x: x.split(';')[3]).values.astype(np.int32)
    # make sure test_subject_id2 is in ascending order
    id_index2 = np.argsort(test_subject_id2)
    test_subject_id2 = test_subject_id2[id_index2]
    test_subject_labels = test_subject_labels[id_index2]
    return test_subject_labels


def load_COVFEFE_feats(file_path):
    def find_files(path, ext="", prefix=""):
        return [os.path.join(path, x) for x in os.listdir(path) if x.endswith(ext) and x.startswith(prefix)]

    files = find_files(file_path, ext='.csv')
    files.sort()
    features_all = []

    if np.array(pd.read_csv(files[0], ';')).shape[1] > np.array(pd.read_csv(files[0], ',')).shape[1]:
        sep = ';'
    else:
        sep = ','
    for file in files:
        temp = np.array(pd.read_csv(file, sep=sep))
        features_all.append(temp.flatten())
    try:
        features_all = np.array(features_all).astype(np.float)
    except:
        features_all = np.array(features_all)[:, 1:].astype(np.float)

    return features_all

if __name__ == "__main__":

    # bertTrnDF = pd.read_csv('../ADReSS_features/bert/trn_bert_features.csv')
    # bertTstDF = pd.read_csv('../ADReSS_features/bert/tst_bert_features.csv')
    # bert_features, tst_bert_features = [], []
    # for i in range(bertTrnDF.shape[0]):
    #     bert_features.append(bertTrnDF.iloc[i][-768:])
    # bert_features = np.asarray(bert_features, dtype='float')
    # for i in range(bertTstDF.shape[0]):
    #     tst_bert_features.append(bertTstDF.iloc[i][-768:])
    # tst_bert_features = np.asarray(tst_bert_features, dtype='float')
    # print('Shape: ', bert_features.shape, tst_bert_features.shape)
    # print('BERT')
    # trn_feats = bert_features
    # trn_labels = bertTrnDF.ad.values
    # tst_feats = tst_bert_features
    # tst_labels = bertTstDF.ad.values

    # #################################################################
    # random_states = range(10)
    # bert_scores_all = []
    # bert_tst_scores_all = []
    #
    # for random_state in random_states:
    #
    #     dataset = 'jccocc_moca'
    #     if dataset == 'ADReSS2020':
    #         trn_feats, trn_labels, tst_feats, tst_labels, subject_id, tst_subject_id = load_ADReSS2020_train_data()
    #     elif dataset == 'AD2021':
    #         trn_feats, trn_labels, tst_feats, tst_labels, subject_id, tst_subject_id = load_AD2021_train_data()
    #         files = find_files('/home11a/xiaoquan/learning/features/AD2021/train/lex_chinese/', ext='.csv')
    #     elif dataset == 'jccocc_moca':
    #         trn_feats, trn_labels, tst_feats, tst_labels, subject_id, tst_subject_id = load_jccocc_moca_data(random_state)
    #
    #     svm = Classifier(mode='svm', pre='pca', C=1, kernel='rbf')
    #     cv, times = 10, 10
    #     bert_scores, bert_tst_scores = svm.cv_test(trn_feats, trn_labels, tst_feats, tst_labels,
    #                                                cv=cv, times=times, testtimes=times)
    #
    #     bert_scores_all.append(bert_scores)
    #     bert_tst_scores_all.append(bert_tst_scores)
    #
    # # print(bert_scores_all)
    # # print(bert_tst_scores_all)
    #
    # print('Training set:')
    # acc = np.concatenate(([item['acc'] for item in bert_scores_all])).mean()
    # prec = np.concatenate(([item['prec'] for item in bert_scores_all])).mean()
    # rec = np.concatenate(([item['rec'] for item in bert_scores_all])).mean()
    # f1 = np.concatenate(([item['f1'] for item in bert_scores_all])).mean()
    # print(acc, prec, rec, f1)
    #
    # print('Test set:')
    # acc_tst = np.concatenate(([item['acc'] for item in bert_tst_scores_all])).mean()
    # prec_tst = np.concatenate(([item['prec'] for item in bert_tst_scores_all])).mean()
    # rec_tst = np.concatenate(([item['rec'] for item in bert_tst_scores_all])).mean()
    # f1_tst = np.concatenate(([item['f1'] for item in bert_tst_scores_all])).mean()
    # print(acc_tst, prec_tst, rec_tst, f1_tst)

    #####################################################################
    def Standardize(X_train, X_test):
        std = StandardScaler().fit(X_train)
        X_train = std.transform(X_train)
        X_test = std.transform(X_test)
        return X_train, X_test

    def classification(X_train, y_train, X_test, y_test):
        clf = classifier()
        clf.fit(X_train, y_train)
        pre_dict_label = clf.predict(X_test)
        # return balanced_accuracy_score(y_test, pre_dict_label)
        # return accuracy_score(y_test, pre_dict_label)
        return pre_dict_label, [], []

    def classifier():
        # return SVC(kernel='linear', C=1, probability=True, random_state=0)
        return SVC(kernel='rbf', C=1, probability=True, gamma='auto', random_state=0)

    from feature_selection_1 import fdr_select_features
    from feature_selection_1 import ANOVA_select_features
    from feature_selection_1 import mutual_info
    from feature_selection_1 import pearson_correlation_test
    from feature_selection_1 import random_forests_select_features

    dataset = 'jccocc_2022_03_19'
    random_states = list(range(5))
    n_folds = 5
    consider_speaker = True

    trn_voting = False
    test_voting = False
    average = 'macro'

    if dataset == 'ADReSS2020':
        trn_feats, trn_labels, tst_feats, tst_labels, subject_id, tst_subject_id = load_ADReSS2020_train_data()
    elif dataset == 'AD2021':
        trn_feats, trn_labels, tst_feats, tst_labels, subject_id, tst_subject_id = load_AD2021_train_data()
        files = find_files('/home11a/xiaoquan/learning/features/AD2021/train/lex_chinese/', ext='.csv')
    elif dataset == 'jccocc_moca':
        trn_feats, trn_labels, tst_feats, tst_labels, subject_id, tst_subject_id = load_jccocc_moca_data()
    elif dataset == 'jccocc_2022_03_11':
        trn_feats, trn_labels, tst_feats, tst_labels, subject_id, tst_subject_id = load_jccocc_moca_data_2022_03_11()
    elif dataset == 'jccocc_2022_03_19':
        trn_feats, trn_labels, tst_feats, tst_labels, subject_id, tst_subject_id = load_jccocc_moca_data_2022_03_19()

    acc_all_folds = []
    pre_all_folds = []
    rec_all_folds = []
    f1_all_folds = []
    for random_state in random_states:

        if consider_speaker:
            subject_unique_id = np.unique(subject_id)
            id_list = np.array_split(shuffle(subject_unique_id, random_state=random_state), n_folds)
            train_index_all = []
            test_index_all = []
            for ids in id_list:
                test_index = np.concatenate([np.where(id == subject_id)[0] for id in ids])
                trn_index = np.delete(range(len(subject_id)), test_index)
                test_index_all.append(test_index)
                train_index_all.append(trn_index)
                assert len(np.unique(subject_id[test_index])) + len(np.unique(subject_id[trn_index])) == len(subject_unique_id)
        else:
            cv_outer = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            train_index_all = []
            test_index_all = []
            for train_index, test_index in cv_outer.split(trn_feats):
                train_index_all.append(train_index)
                test_index_all.append(test_index)

        for i in range(len(train_index_all)):
            trn_index = train_index_all[i]
            tst_index = test_index_all[i]

            training_feats, testing_feats = Standardize(trn_feats[trn_index, :], trn_feats[tst_index, :])

            # _, selected_index, _ = fdr_select_features(training_feats, trn_labels[trn_index], n_fs)
            # _, selected_index, _ = ANOVA_select_features(training_feats, trn_labels[trn_index], n_fs)
            # _, selected_index, _ = pearson_correlation_test(training_feats, trn_labels[trn_index], n_fs)
            # indices, importances = random_forests_select_features(training_feats, trn_labels[trn_index], None)
            # selected_index = indices[:1000]

            # training_feats = training_feats[:, selected_index]
            # testing_feats = testing_feats[:, selected_index]

            pre_labels, _, _ = classification(training_feats, trn_labels[trn_index], testing_feats, trn_labels[tst_index])

            if not trn_voting:
                acc = accuracy_score(trn_labels[tst_index], pre_labels)
                pre = precision_score(trn_labels[tst_index], pre_labels, average=average)
                rec = recall_score(trn_labels[tst_index], pre_labels, average=average)
                f1 = f1_score(trn_labels[tst_index], pre_labels, average=average)
            else:
                # vote speaker label
                vote_labels = []
                true_labes = []
                tst_id = subject_id[tst_index]
                tst_true_labels = trn_labels[tst_index]
                tst_id_unique = np.unique(tst_id)
                for id in tst_id_unique:
                    vote_label = pd.value_counts(pre_labels[np.where(tst_id==id)[0]]).index[0]
                    tru_label = tst_true_labels[np.where(tst_id==id)[0]][0]
                    vote_labels.append(vote_label)
                    true_labes.append(tru_label)
                acc = accuracy_score(true_labes, vote_labels)
                pre = precision_score(true_labes, vote_labels, average=average)
                rec = recall_score(true_labes, vote_labels, average=average)
                f1 = f1_score(true_labes, vote_labels, average=average)
            acc_all_folds.append(acc)
            pre_all_folds.append(pre)
            rec_all_folds.append(rec)
            f1_all_folds.append(f1)

    print('cross validation:')
    print('acc mean: %.3f' % np.mean(acc_all_folds))
    # print('acc std: %.3f' % np.std(acc_all_folds))

    print('pre mean: %.3f' % np.mean(pre_all_folds))
    # print('pre std: %.3f' % np.std(pre_all_folds))

    print('rec mean: %.3f' % np.mean(rec_all_folds))
    # print('rec std: %.3f' % np.std(rec_all_folds))

    print('f1 mean: %.3f' % np.mean(f1_all_folds))
    # print('f1 std: %.3f' % np.std(f1_all_folds))

    print('test set:')
    training_feats, testing_feats = Standardize(trn_feats, tst_feats)
    pre_labels, _, _ = classification(training_feats, trn_labels, testing_feats, tst_labels)

    if not test_voting:
        print('acc: %.3f'% accuracy_score(tst_labels, pre_labels))
        print('pre: %.3f'% precision_score(tst_labels, pre_labels, average=average))
        print('rec: %.3f'% recall_score(tst_labels, pre_labels, average=average))
        print('f1: %.3f'% f1_score(tst_labels, pre_labels, average=average))
    else:
        tst_id = tst_subject_id
        tst_true_labels = tst_labels
        tst_id_unique = np.unique(tst_id)
        vote_labels = []
        true_labes = []
        for id in tst_id_unique:
            vote_label = pd.value_counts(pre_labels[np.where(tst_id == id)[0]]).index[0]
            tru_label = tst_true_labels[np.where(tst_id == id)[0]][0]
            vote_labels.append(vote_label)
            true_labes.append(tru_label)
        print(accuracy_score(true_labes, vote_labels))


