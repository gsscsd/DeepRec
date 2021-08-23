import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
print(sys.path)

from deeprec.features.feature_columns import DenseFeature, SparseFeature
from deeprec.models.base import BaseModel

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

    sparse_feats = [SparseFeature(name=i, dim=1, vocab_size=data[i].max() + 1, emb_size=4) for i in
                    sparse_features]
    for idx, feat in enumerate(sparse_feats):
        print(idx, sparse_feats[idx])
    dense_feats = [DenseFeature(name=i, dim=1) for i in dense_features]
    for idx, feat in enumerate(dense_feats):
        print(idx, dense_feats[idx])
    feature_names = dense_features + sparse_features
    print(feature_names)
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    model = BaseModel(dense_feature=dense_feats, sparse_feature=sparse_feats)
    model = model.get_model()
    print(model.summary())
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values, batch_size=64, epochs=30, verbose=2, validation_split=0.2, )
    # pred_ans = model.predict(test_model_input, batch_size=256)
    # print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

