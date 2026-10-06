# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import pandas as pd

LEN_TRAIN = 5000

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()[:LEN_TRAIN]
    y_train = train_df['target'].values[:LEN_TRAIN]
    X_validation = train_df['text'].astype(str).tolist()[LEN_TRAIN:]
    y_validation = train_df['target'].values[LEN_TRAIN:]
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_validation, y_validation, X_test_unlabeled

# 2. 调用函数来获取数据
#    这个函数会自动读取 .csv 文件并返回你需要的所有内容
X_train, y_train, X_validation, y_validation, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"训练集样本数量: {LEN_TRAIN}")
print(f"训练集标签数量: {LEN_TRAIN}")
print(f"验证集样本数量: {len(X_validation)}")
print(f"验证集标签数量: {len(y_validation)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# 清洗文本数据
import re

def clean(text):
    #去除头部和标点
    res = ' '.join(text.split('\n\n')[1:]).replace('\n', ' ')
    res = re.sub(r'[^A-Za-z0-9\s]', ' ', res)
    res = re.sub(r'\s+', ' ', res).strip().lower()
    return res

X_train_cleaned = [clean(text) for text in X_train]
X_validation_cleaned = [clean(text) for text in X_validation]
X_test_cleaned = [clean(text) for text in X_test_unlabeled]

# 打印一条清洗过的样本
print("第一个清洗过的样本内容:")
print(X_train_cleaned[0])
print("\n" + "="*50)

# --- 在这里开始你的实验！ ---
# 现在，你可以使用 X_train, y_train, 和 X_test_unlabeled 这三个变量
# 来进行后续的特征提取、模型训练和预测了。

# 举例：
# 1. 创建TF-IDF向量化器
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_cleaned)
X_validation_tfidf = vectorizer.transform(X_validation_cleaned)
X_test_tfidf = vectorizer.transform(X_test_cleaned)

# 2. 训练模型...
from sklearn.svm import SVC
from time import time

Cs = [0.2, 0.3, 0.5, 0.8]
svm_train = []
svm_validation = []
svm_time = []
best_model = None

for C in Cs:
    start = time()
    svm_model = SVC(kernel='linear', C=C)
    svm_model.fit(X_train_tfidf, y_train)
    svm_train.append(svm_model.score(X_train_tfidf, y_train))
    svm_validation.append(svm_model.score(X_validation_tfidf, y_validation))
    svm_time.append(time() - start)
    print(f'svm(C={C}): 训练完成')

from sklearn.linear_model import LogisticRegression
log_train = []
log_validation = []
log_time = []
for C in Cs:
    start = time()
    log_model = LogisticRegression(C=C)
    log_model.fit(X_train_tfidf, y_train)
    log_train.append(log_model.score(X_train_tfidf, y_train))
    log_validation.append(log_model.score(X_validation_tfidf, y_validation))
    log_time.append(time() - start)
    print(f'lr(C={C}): 训练完成')

    # 逻辑回归模型在正则化参数为0.5时表现最佳
    if C == 0.5:
        best_model = log_model

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

C = ['0.2', '0.3', '0.5', '0.8']
axes[0].bar(C, svm_train, label='train')
axes[0].bar(C, svm_validation, label='validation')
axes[0].set_title('svm训练和验证表现')
axes[0].set_xlabel('C')
axes[0].set_ylabel('score')
axes[0].legend()
axes[1].bar(C, log_train, label='train')
axes[1].bar(C, log_validation, label='validation')
axes[1].set_title('逻辑回归训练和验证表现')
axes[1].set_xlabel('C')
axes[1].set_ylabel('score')
axes[1].legend()
axes[2].bar(C, svm_time, label='svm')
axes[2].bar(C, log_time, label='逻辑回归')
axes[2].set_title('两种模型耗时对比')
axes[2].set_xlabel('C')
axes[2].set_ylabel('time')
axes[2].legend()

plt.show()

# 3. 对测试集进行预测...
predictions = best_model.predict(X_test_tfidf)

# 4. 保存你的预测结果...
pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)
