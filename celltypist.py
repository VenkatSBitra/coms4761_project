# %%
import scanpy as sc
import pandas as pd
import numpy as np

# %%
male = male = sc.read_h5ad("cleaned_data/male_gonadal_raw.h5ad")
female = sc.read_h5ad("cleaned_data/female_gonadal_raw.h5ad")

# %%
def get_array(adata: sc.AnnData):
    sorted_genes = sorted(adata.var_names)
    df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=sorted_genes)
    df2 = adata.obs['cell_type']
    df.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    X = df.to_numpy()
    y = df2.to_numpy()
    return X, y, sorted_genes

# %%
male_X, male_y, sorted_genes = get_array(male)
female_X, female_y, _ = get_array(female)

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(male_y)

male_y = le.transform(male_y)
female_y = le.transform(female_y)

# %%
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
ohe.fit(male_y.reshape(-1, 1))

male_y_onehot = ohe.transform(male_y.reshape(-1, 1))
female_y_onehot = ohe.transform(female_y.reshape(-1, 1))

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(male_X, male_y, test_size=0.2, random_state=42, stratify=male_y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

# %%
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import scipy.special

from collections import Counter

# %%
sgd_clf = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)

# %%
y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred, target_names=[le.classes_[int(s.split('_')[1])] for s in ohe.get_feature_names_out()], zero_division=0))

# %%
y_other_pred = sgd_clf.predict(female_X)

accuracy_other = accuracy_score(female_y, y_other_pred)
f1_other = f1_score(female_y, y_other_pred, average='weighted')

print(f"Accuracy (Female): {accuracy_other:.4f}")
print(f"F1 Score (Female): {f1_other:.4f}")

print(classification_report(female_y, y_other_pred, target_names=[le.classes_[int(s.split('_')[1])] for s in ohe.get_feature_names_out()], zero_division=0))

# %%



