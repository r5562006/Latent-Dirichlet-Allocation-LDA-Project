潛在狄利克雷分配 (LDA)import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# 範例文本數據
documents = [
    "I love programming in Python",
    "Python is a great language for data science",
    "I use machine learning in my projects",
    "Data science and machine learning are closely related",
    "I enjoy learning new things about programming"
]

# 文本數據向量化
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# 訓練LDA模型
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# 顯示每個主題的關鍵詞
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

print_top_words(lda, vectorizer.get_feature_names_out(), 5)