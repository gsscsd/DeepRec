import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
print(sys.path)
from deeprec.features.feature_columns import DenseFeature, SparseFeature

if __name__ == "__main__":
    data = pd.read_csv('criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    print(data.head())

    sparse_feats = [SparseFeature(name=i, dim=1, vocab_size=data[i].max()+1, emb_size=4, dtype=data[i].dtype) for i in sparse_features]
    for idx, feat in enumerate(sparse_feats):
        print(idx, sparse_feats[idx])
    dense_feats = [DenseFeature(name=i, dim=1, dtype=data[i].dtype) for i in dense_features]
    for idx, feat in enumerate(dense_feats):
        print(idx, dense_feats[idx])
