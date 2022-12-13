import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from prepare_data import generate_features

def performance(pred, target):
    tn, fp, fn, tp = confusion_matrix(pred,target).ravel()
    return accuracy_score(pred,target), tp/(tp+fn), tn/(tn+fp), matthews_corrcoef(pred,target)  #acc, sen, spe, mcc

def x_concate(feature1, feature2):
    feature = []
    for i in range(feature1.shape[0]):
        feature.append(np.concatenate((feature1[i], feature2[i])))
    return np.array(feature)
def ind_test(x_train, y_train, x_test, y_test):
    # Independent Test
    # print("Independent Test")
    clf = RandomForestClassifier(n_estimators=200, min_samples_split=6)
    clf.fit(x_train, y_train)
    importances = clf.feature_importances_
    pred = clf.predict(x_test)
    acc, sen, spe, mcc = performance(pred, y_test)
    # print(f'acc = {np.round((acc), 4)}, sen = {np.round(sen, 4)}, spe = {np.round(spe, 4)}, mcc = {np.round(mcc, 4)}')
    return np.round(acc, 4), np.round(sen, 4), np.round(spe, 4), np.round(mcc, 4), importances

def k_fold(x_train, y_train, x_test, y_test, fold):

    # 5-fold random forests
    # print(f'{fold}-fold')
    kf = KFold(n_splits=fold, shuffle=True, random_state=15)
    X_all, Y_all = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
    acc_array, sen_array, spe_array, mcc_array = [], [], [], []
    importances = [0] * X_all.shape[1]
    for k, (train_index, test_index) in enumerate(kf.split(X_all)):
        # training
        # print(f'fold:{k+1}:')
        clf = RandomForestClassifier(n_estimators=200, min_samples_split=6)
        X_train, Y_train = X_all[train_index], Y_all[train_index]
        X_test, Y_test = X_all[test_index], Y_all[test_index]
        clf.fit(X_train, Y_train)
        pred = clf.predict(X_test)
        importances += clf.feature_importances_
        acc, sen, spe, mcc = performance(pred, Y_test)
        # print(f'acc = {acc}, sen = {sen}, spe = {spe}, mcc = {mcc}')
        acc_array.append(acc)
        sen_array.append(sen)
        spe_array.append(spe)
        mcc_array.append(mcc)

    # print(f'Average:\nacc = {np.round(np.mean(acc_array), 4)}, sen = {np.round(np.mean(sen_array), 4)}, spe = {np.round(np.mean(spe_array), 4)}, mcc = {np.round(np.mean(mcc), 4)}')
    return np.round(np.mean(acc_array), 4), np.round(np.mean(sen_array), 4), np.round(np.mean(spe_array), 4), np.round(np.mean(mcc), 4), importances/fold

def run(x_train, y_train, x_test, y_test, fold, times):
    acc_ind, sen_ind, spe_ind, mcc_ind = [], [], [], []
    acc_fold, sen_fold, spe_fold, mcc_fold = [], [], [], []
    for i in range(times):
        acc, sen, spe, mcc, importances = ind_test(x_train, y_train, x_test, y_test)
        acc_ind.append(acc)
        sen_ind.append(sen)
        spe_ind.append(spe)
        mcc_ind.append(mcc)
        acc, sen, spe, mcc, importances = k_fold(x_train, y_train, x_test, y_test, fold)
        acc_fold.append(acc)
        sen_fold.append(sen)
        spe_fold.append(spe)
        mcc_fold.append(mcc)
    if times == 1:
        print(f'Independent_test\nacc = {np.round((acc_ind[0]), 4)}, sen = {np.round(sen_ind[0], 4)}, spe = {np.round(spe_ind[0], 4)}, mcc = {np.round(mcc_ind[0], 4)}')
        print(f'{fold}-fold\nacc = {np.round((acc_fold[0]), 4)}, sen = {np.round(sen_fold[0], 4)}, spe = {np.round(spe_fold[0], 4)}, mcc = {np.round(mcc_fold[0], 4)}')
    else:
        print(f'Independent_test\nacc = {np.round(np.mean(acc_ind), 4)} +- {np.round(np.std(acc_ind), 4)}, sen = {np.round(np.mean(sen_ind), 4)} +- {np.round(np.std(sen_ind), 4)}, spe = {np.round(np.mean(spe_ind), 4)} +- {np.round(np.std(spe_ind), 4)}, mcc = {np.round(np.mean(mcc_ind), 4)} +- {np.round(np.std(mcc_ind), 4)}')
        print(f'{fold}-fold\nacc = {np.round(np.mean(acc_fold), 4)} +- {np.round(np.std(acc_fold), 4)}, sen = {np.round(np.mean(sen_fold), 4)} +- {np.round(np.std(sen_fold), 4)}, spe = {np.round(np.mean(spe_fold), 4)} +- {np.round(np.std(spe_fold), 4)}, mcc = {np.round(np.mean(mcc_fold), 4)} +- {np.round(np.std(mcc_fold), 4)}')

def find_importance(x_train, y_train, x_test, y_test, fold, times):
    acc_ind, sen_ind, spe_ind, mcc_ind = [], [], [], []
    acc_fold, sen_fold, spe_fold, mcc_fold = [], [], [], []
    importances_ind = [0] * x_train.shape[1]
    importances_fold = [0] * x_train.shape[1]
    for i in range(times):
        acc, sen, spe, mcc, importances = ind_test(x_train, y_train, x_test, y_test)
        acc_ind.append(acc)
        sen_ind.append(sen)
        spe_ind.append(spe)
        mcc_ind.append(mcc)
        importances_ind += importances 
        acc, sen, spe, mcc, importances = k_fold(x_train, y_train, x_test, y_test, fold)
        acc_fold.append(acc)
        sen_fold.append(sen)
        spe_fold.append(spe)
        mcc_fold.append(mcc)
        importances_fold += importances
    # print("oringin_result:")
    if times == 1:
        # print(f'Independent_test\nacc = {np.round((acc_ind[0]), 4)}, sen = {np.round(sen_ind[0], 4)}, spe = {np.round(spe_ind[0], 4)}, mcc = {np.round(mcc_ind[0], 4)}')
        # print(f'{fold}-fold\nacc = {np.round((acc_fold[0]), 4)}, sen = {np.round(sen_fold[0], 4)}, spe = {np.round(spe_fold[0], 4)}, mcc = {np.round(mcc_fold[0], 4)}')
        return importances_ind, importances_fold 
    else:
        # print(f'Independent_test\nacc = {np.round(np.mean(acc_ind), 4)} +- {np.round(np.std(acc_ind), 4)}, sen = {np.round(np.mean(sen_ind), 4)} +- {np.round(np.std(sen_ind), 4)}, spe = {np.round(np.mean(spe_ind), 4)} +- {np.round(np.std(spe_ind), 4)}, mcc = {np.round(np.mean(mcc_ind), 4)} +- {np.round(np.std(mcc_ind), 4)}')
        # print(f'{fold}-fold\nacc = {np.round(np.mean(acc_fold), 4)} +- {np.round(np.std(acc_fold), 4)}, sen = {np.round(np.mean(sen_fold), 4)} +- {np.round(np.std(sen_fold), 4)}, spe = {np.round(np.mean(spe_fold), 4)} +- {np.round(np.std(spe_fold), 4)}, mcc = {np.round(np.mean(mcc_fold), 4)} +- {np.round(np.std(mcc_fold), 4)}')
        return importances_ind/times, importances_fold/times

def delete_feature(feature, importances, del_num):
    sort_array = sorted(importances)
    threshold = sort_array[del_num]
    del_col = []
    for i in range(len(importances)):
        if importances[i] < threshold:
            del_col.append(i)
    new_feature = np.delete(feature, del_col, 1)
    return new_feature

if __name__ == "__main__":

    # setting dataset path
    BM_train_path = "./dataset/benchmarkdataset_train.fasta"
    BM_test_path = "./dataset/benchmarkdataset_test.fasta"
    NT15_train_path = "./dataset/NT15dataset_train.fasta"
    NT15_test_path = "./dataset/NT15dataset_test.fasta"

    # prepare training and testing data
    BM_train_y, BM_train_AAC, BM_train_PAAC, BM_train_APAAC = generate_features(BM_train_path, "BM")
    BM_test_y, BM_test_AAC, BM_test_PAAC, BM_test_APAAC = generate_features(BM_test_path, "BM")
    NT15_train_y, NT15_train_AAC, NT15_train_PAAC, NT15_train_APAAC = generate_features(NT15_train_path, "NT15")
    NT15_test_y, NT15_test_AAC, NT15_test_PAAC, NT15_test_APAAC = generate_features(NT15_test_path, "NT15")
    """
    BM_train_AAC_PAAC = x_concate(BM_train_AAC, BM_train_PAAC)
    BM_train_AAC_APAAC = x_concate(BM_train_AAC, BM_train_APAAC)
    BM_train_PAAC_APAAC = x_concate(BM_train_PAAC, BM_train_APAAC)
    BM_train_AAC_PAAC_APAAC = x_concate(BM_train_AAC, BM_train_PAAC_APAAC)
    BM_test_AAC_PAAC = x_concate(BM_test_AAC, BM_test_PAAC)
    BM_test_AAC_APAAC = x_concate(BM_test_AAC, BM_test_APAAC)
    BM_test_PAAC_APAAC = x_concate(BM_test_PAAC, BM_test_APAAC)
    BM_test_AAC_PAAC_APAAC = x_concate(BM_test_AAC, BM_test_PAAC_APAAC)
    NT15_train_AAC_PAAC = x_concate(NT15_train_AAC, NT15_train_PAAC)
    NT15_train_AAC_APAAC = x_concate(NT15_train_AAC, NT15_train_APAAC)
    NT15_train_PAAC_APAAC = x_concate(NT15_train_PAAC, NT15_train_APAAC)
    NT15_train_AAC_PAAC_APAAC = x_concate(NT15_train_AAC, NT15_train_PAAC_APAAC)
    NT15_test_AAC_PAAC = x_concate(NT15_test_AAC, NT15_test_PAAC)
    NT15_test_AAC_APAAC = x_concate(NT15_test_AAC, NT15_test_APAAC)
    NT15_test_PAAC_APAAC = x_concate(NT15_test_PAAC, NT15_test_APAAC)
    NT15_test_AAC_PAAC_APAAC = x_concate(NT15_test_AAC, NT15_test_PAAC_APAAC)
    """ 
    # shuffle
    BM_train_y, BM_train_AAC, BM_train_PAAC, BM_train_APAAC = shuffle(BM_train_y, BM_train_AAC, BM_train_PAAC, BM_train_APAAC, random_state=0) 
    BM_test_y, BM_test_AAC, BM_test_PAAC, BM_test_APAAC = shuffle(BM_test_y, BM_test_AAC, BM_test_PAAC, BM_test_APAAC, random_state=0)
    NT15_train_y, NT15_train_AAC, NT15_train_PAAC, NT15_train_APAAC = shuffle(NT15_train_y, NT15_train_AAC, NT15_train_PAAC, NT15_train_APAAC, random_state=0)
    NT15_test_y, NT15_test_AAC, NT15_test_PAAC, NT15_test_APAAC = shuffle(NT15_test_y, NT15_test_AAC, NT15_test_PAAC, NT15_test_APAAC, random_state=0)
    """ 
    #Train one time
    print("One time")
    print("BM_AAC")
    BM_AAC_imp_ind, BM_AAC_imp_fold = find_importance(BM_train_AAC, BM_train_y, BM_test_AAC, BM_test_y, 5, 1)
    print("NT15_AAC")
    NT15_AAC_imp_ind, NT15_AAC_imp_fold = find_importance(NT15_train_AAC, NT15_train_y, NT15_test_AAC, NT15_test_y, 5, 1)
    print("BM_PAAC")
    BM_PAAC_imp_ind, BM_PAAC_imp_fold = find_importance(BM_train_PAAC, BM_train_y, BM_test_PAAC, BM_test_y, 5, 1)
    print("NT15_PAAC")
    NT15_PAAC_imp_ind, NT15_PAAC_imp_fold = find_importance(NT15_train_PAAC, NT15_train_y, NT15_test_PAAC, NT15_test_y, 5, 1)
    print("BM_APAAC")
    BM_APAAC_imp_ind, BM_APAAC_imp_fold = find_importance(BM_train_APAAC, BM_train_y, BM_test_APAAC, BM_test_y, 5, 1)
    print("NT15_APAAC")
    NT15_APAAC_imp_ind, NT15_APAAC_imp_fold = find_importance(NT15_train_APAAC, NT15_train_y, NT15_test_APAAC, NT15_test_y, 5, 1)
    """
    #Train ten times
    print("ten time")
    print("BM_AAC")
    BM_AAC_imp_ind_10, BM_AAC_imp_fold_10 = find_importance(BM_train_AAC, BM_train_y, BM_test_AAC, BM_test_y, 5, 10)
    new_BM_train_AAC = delete_feature(BM_train_AAC, BM_AAC_imp_fold_10, 3)
    new_BM_test_AAC = delete_feature(BM_test_AAC, BM_AAC_imp_fold_10, 3)
    run(new_BM_train_AAC, BM_train_y, new_BM_test_AAC, BM_test_y, 5, 10)
    print("NT15_AAC")
    NT15_AAC_imp_ind_10, NT15_AAC_imp_fold_10 = find_importance(NT15_train_AAC, NT15_train_y, NT15_test_AAC, NT15_test_y, 5, 10)
    new_NT15_train_AAC = delete_feature(NT15_train_AAC, NT15_AAC_imp_fold_10, 3)
    new_NT15_test_AAC = delete_feature(NT15_test_AAC, NT15_AAC_imp_fold_10, 3)
    run(new_NT15_train_AAC, NT15_train_y, new_NT15_test_AAC, NT15_test_y, 5, 10)
    print("BM_PAAC")
    BM_PAAC_imp_ind_10, BM_PAAC_imp_fold_10 = find_importance(BM_train_PAAC, BM_train_y, BM_test_PAAC, BM_test_y, 5, 10)
    new_BM_train_PAAC = delete_feature(BM_train_PAAC, BM_PAAC_imp_fold_10, 3)
    new_BM_test_PAAC = delete_feature(BM_test_PAAC, BM_PAAC_imp_fold_10, 3)
    run(new_BM_train_PAAC, BM_train_y, new_BM_test_PAAC, BM_test_y, 5, 10)
    print("NT15_PAAC")
    NT15_PAAC_imp_ind_10, NT15_PAAC_imp_fold_10 = find_importance(NT15_train_PAAC, NT15_train_y, NT15_test_PAAC, NT15_test_y, 5, 10)
    new_NT15_train_PAAC = delete_feature(NT15_train_PAAC, NT15_PAAC_imp_fold_10, 3)
    new_NT15_test_PAAC = delete_feature(NT15_test_PAAC, NT15_PAAC_imp_fold_10, 3)
    run(new_NT15_train_PAAC, NT15_train_y, new_NT15_test_PAAC, NT15_test_y, 5, 10)
    print("BM_APAAC")
    BM_APAAC_imp_ind_10, BM_APAAC_imp_fold_10 = find_importance(BM_train_APAAC, BM_train_y, BM_test_APAAC, BM_test_y, 5, 10)
    new_BM_train_APAAC = delete_feature(BM_train_APAAC, BM_APAAC_imp_fold_10, 3)
    new_BM_test_APAAC = delete_feature(BM_test_APAAC, BM_APAAC_imp_fold_10, 3)
    run(new_BM_train_APAAC, BM_train_y, new_BM_test_APAAC, BM_test_y, 5, 10)
    print("NT15_APAAC")
    NT15_APAAC_imp_ind_10, NT15_APAAC_imp_fold_10 = find_importance(NT15_train_APAAC, NT15_train_y, NT15_test_APAAC, NT15_test_y, 5, 10)
    new_NT15_train_APAAC = delete_feature(NT15_train_APAAC, NT15_APAAC_imp_fold_10, 3)
    new_NT15_test_APAAC = delete_feature(NT15_test_APAAC, NT15_APAAC_imp_fold_10, 3)
    run(new_NT15_train_APAAC, NT15_train_y, new_NT15_test_APAAC, NT15_test_y, 5, 10)


    """
    print("BM_PAAC_APAAC")
    ind_test(BM_train_PAAC_APAAC, BM_train_y, BM_test_PAAC_APAAC, BM_test_y)
    k_fold(BM_train_PAAC_APAAC, BM_train_y, BM_test_PAAC_APAAC, BM_test_y, 5)
    """
