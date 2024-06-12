import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelBinarizer

# Veri yükleme fonksiyonu
def load_dataset():
    try:
        data = pd.read_csv('/kaggle/input/phishingemails/Phishing_Email.csv', encoding='utf-8')
        data = data.dropna(subset=['Email Text'])  # NaN değerlerini kaldır
        text_data = data['Email Text'].values
        target_labels = data['Email Type'].values
        return text_data, target_labels
    except Exception as e:
        print("Veri kümesi yüklenirken hata oluştu:", e)
        return None, None

# Görünmez karakterleri kaldır
def remove_invisible_chars(text):
    try:
        return re.sub(r'[^\x00-\x7F]+', '', str(text))
    except Exception as e:
        print("Görünmez karakterler kaldırılırken hata oluştu:", e)
        return ""

# TF-IDF dönüşümü
def tfidf_transform(text_data):
    try:
        vectorizer = TfidfVectorizer(max_features=3000, analyzer='word', stop_words='english')
        X = vectorizer.fit_transform(text_data)
        return X, vectorizer
    except Exception as e:
        print("TF-IDF dönüşümünde hata oluştu:", e)
        return None, None

# Veri kümesini yükle ve ön işleme yap
text_data, target_labels = load_dataset()
if text_data is None or target_labels is None:
    exit()

# Görünmez karakterleri kaldır
text_data = [remove_invisible_chars(text) for text in text_data]

# TF-IDF dönüşümü
X, vectorizer = tfidf_transform(text_data)
if X is None:
    exit()

# Verileri eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, target_labels, test_size=0.2, random_state=42)

# Naif Bayes modeli
bayesian_model = MultinomialNB()
bayesian_model.fit(X_train, y_train)

# SVM Model (Support Vector Machine)
svm_model = SVC(kernel='linear', probability=True)  # Metin sınıflandırması için lineer çekirdek
svm_model.fit(X_train, y_train)

# Random Forest Modeli
rf_model = RandomForestClassifier(n_estimators=50)  # Ağaç sayısını azalt
rf_model.fit(X_train, y_train)

# Tahminleri değerlendirme
bayesian_pred = bayesian_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Model performansını değerlendir
accuracy_bayesian = accuracy_score(y_test, bayesian_pred)
precision_bayesian = precision_score(y_test, bayesian_pred, average='weighted', zero_division=1)
recall_bayesian = recall_score(y_test, bayesian_pred, average='weighted', zero_division=1)
f1_bayesian = f1_score(y_test, bayesian_pred, average='weighted')

accuracy_svm = accuracy_score(y_test, svm_pred)
precision_svm = precision_score(y_test, svm_pred, average='weighted', zero_division=1)
recall_svm = recall_score(y_test, svm_pred, average='weighted', zero_division=1)
f1_svm = f1_score(y_test, svm_pred, average='weighted')

accuracy_rf = accuracy_score(y_test, rf_pred)
precision_rf = precision_score(y_test, rf_pred, average='weighted', zero_division=1)
recall_rf = recall_score(y_test, rf_pred, average='weighted', zero_division=1)
f1_rf = f1_score(y_test, rf_pred, average='weighted')

print("Naif Bayes Model Doğruluğu:", accuracy_bayesian)
print("Naif Bayes Model Hassasiyeti:", precision_bayesian)
print("Naif Bayes Model Duyarlılığı:", recall_bayesian)
print("Naif Bayes Model F1-skoru:", f1_bayesian)

print("SVM Model Doğruluğu:", accuracy_svm)
print("SVM Model Hassasiyeti:", precision_svm)
print("SVM Model Duyarlılığı:", recall_svm)
print("SVM Model F1-skoru:", f1_svm)

print("Random Forest Model Doğruluğu:", accuracy_rf)
print("Random Forest Model Hassasiyeti:", precision_rf)
print("Random Forest Model Duyarlılığı:", recall_rf)
print("Random Forest Model F1-skoru:", f1_rf)
