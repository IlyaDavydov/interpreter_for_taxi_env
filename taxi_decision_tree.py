import pandas as pd
import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier

from features_extraction import extract_features

class TaxiDecisionTree:
    def __init__(self, max_depth=8, random_state=42):
        """
        Инициализация класса для обучения дерева решений для игры Taxi-v3.
        :param max_depth: Максимальная глубина дерева решений.
        :param random_state: Значение для генератора случайных чисел (для воспроизводимости).
        """
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def train(self, df):
        """
        Обучение дерева решений на данных DataFrame.
        :param df: pandas DataFrame с признаками и действиями.
        """
        # Признаки находятся во всех столбцах, кроме последнего (который содержит действия)
        X = df.iloc[:, :-1].values  # Все колонки, кроме последней
        y = df.iloc[:, -1].values   # Последняя колонка - это действия

        # Обучаем модель
        self.model.fit(X, y)

    def predict(self, state, env, prev_action):
        """
        Прогнозирование действия на основе текущего состояния.
        :param state: Текущее состояние игры в виде признаков (например, список признаков).
        :return: Предсказанное действие.
        """
        features_data = extract_features(state, env)
        features_data["previous_action"] = prev_action
        features = pd.DataFrame([features_data], index=[0])
        return self.model.predict(features)

    def save(self, filename):
        """
        Сохранение обученной модели в файл.
        :param filename: Имя файла для сохранения модели.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filename):
        """
        Загрузка обученной модели из файла.
        :param filename: Имя файла для загрузки модели.
        """
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
