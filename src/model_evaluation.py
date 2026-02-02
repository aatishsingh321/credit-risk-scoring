from sklearn.metrics import roc_auc_score, roc_curve

def ks_statistic(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return max(tpr - fpr)

def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    ks = ks_statistic(y_test, probs)

    print("AUC Score:", round(auc, 4))
    print("KS Statistic:", round(ks, 4))
