"""
一次性脚本：基于Week7的训练集，建一个"历史相似案例"检索索引
跑一次即可，产出的index文件会被backend/agent.py加载使用
"""
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# 读取Week7产出的训练集（已经是标准化后的特征）
X_train = pd.read_csv("data/ml/X_train.csv", index_col=0, parse_dates=True)
y_train = pd.read_csv("data/ml/y_train.csv", index_col=0, parse_dates=True).iloc[:, 0]
train_meta = pd.read_csv("data/ml/train_meta.csv", index_col=0, parse_dates=True)  # 如果Week7没存这个文件，见下方说明

# 建 k-近邻索引（在标准化后的23维特征空间里找"距离最近"的历史样本）
nn_index = NearestNeighbors(n_neighbors=20, metric="euclidean")
nn_index.fit(X_train.values)

# 存下索引本身，以及对应的ticker/日期/涨跌标签（用于检索命中后还原成"哪只股票哪一天"）
joblib.dump(nn_index, "data/models/similar_case_index.pkl")

case_info = pd.DataFrame({
    "ticker": train_meta["ticker"].values,
    "date": X_train.index,
    "target": y_train.values
})
case_info.to_csv("data/models/similar_case_info.csv", index=False)

print(f"索引构建完成，覆盖 {len(X_train)} 个历史样本")
