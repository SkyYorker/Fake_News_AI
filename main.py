
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# Чтение данных
df=pd.read_csv('fake_news.csv')

# Разделение данных
X_train,X_test,y_train,y_test=train_test_split(df['text'], df.label, test_size=0.2, random_state=7)

# Векторизация текста с использованием списка стоп-слов
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Подоготовка данных
tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)

# Тренировка модели
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Предсказывание классов
y_pred=pac.predict(tfidf_test)

# Результат
score=accuracy_score(y_test,y_pred)
print(f'Точность: {round(score*100,2)}%')


# Графики

# Кривая ROC
RocCurveDisplay.from_estimator(pac, X=tfidf_test, y=y_test)
plt.title("Кривая ROC")
plt.show()

# Распределение предсказанных классов
sns.histplot(y_pred)
plt.title("Распределение предсказанных классов")
plt.xlabel("Классы")
plt.ylabel("Частота")
plt.show()


# Матрица ошибок
ConfusionMatrixDisplay.from_predictions(y_test,y_pred, cmap='viridis', colorbar=True)
plt.title("Матрица ошибок")
plt.show()

# Точность по классам
class_report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(class_report).transpose()
df_report['precision'].plot(kind='bar')
plt.title("Точность по классам")
plt.xlabel("Классы")
plt.ylabel("Точность")
plt.show()