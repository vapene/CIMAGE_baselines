import numpy as np
import functools

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

def label_classification(embeddings, y, train_mask, val_mask, test_mask):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    # Y = Y.reshape(-1, 1)
    # onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    # Y = onehot_encoder.transform(Y).toarray()
    X = normalize(X, norm='l2')

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    y_train = Y[train_mask]
    y_val = Y[val_mask]
    y_test = Y[test_mask]
    # X_train, X_test, y_train, y_test = train_test_split(X, Y,
    #                                                     test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    val_accuracy = accuracy_score(y_val, y_pred)

    y_pred = clf.predict_proba(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    test_accuracy = accuracy_score(y_test, y_pred)

    return val_accuracy, test_accuracy
