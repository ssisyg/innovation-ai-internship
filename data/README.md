# data/ 目录说明

本目录下的数据文件未纳入版本控制（体积较大，且可由notebook脚本重新生成）：

- `raw/` — 原始OHLCV价格数据（由 01_data_collection.ipynb 生成）
- `processed/` — 归一化后的清洗数据（由 02_preprocessing.ipynb 生成）
- `features/` — 技术指标特征（由 03_feature_engineering.ipynb 生成）
- `sentiment/` — 新闻情感分析结果（由 05/06 notebook生成）
- `ml/` — 训练/测试集、scaler等（由 07_ml_preparation.ipynb 生成）
- `models/` — 训练好的模型、SHAP解释器（由 08_model_training.ipynb 生成）

按顺序运行 01→08 号 notebook 即可重新生成全部数据。
