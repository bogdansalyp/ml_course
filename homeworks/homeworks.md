# Домашние работы

## 01. Вступление

Придумайте вариант использования машинного обучения в жизни. Это может быть, например, проект, связанный с вашей рабочей областью: что-то, в чём вам всегда не хватало интеллектуальной автоматизации.

1. Выберите задачу.
2. Определите вид задачи (обучение с учителем, обучение без учителя, обучение с подкреплением).
3. Опишите задачу, в какой её части можно применить машинное обучение и какие выгоды от этого будут.

## 02. Инструменты

### Начальный уровень

1. Создать новый Jupyter Notebook.
2. Скачать Boston housing dataset по ссылке (https://github.com/mcsml/ml_basics/blob/master/train.csv) и разместить в той же директории, что и ваш Notebook. Описание данных можно найти здесь: https://www.kaggle.com/c/boston-housing.
3. Загрузить данные в DataFrame.
4. Построить график зависимости значений medv от значений dis.
5. Создать отдельный DataFrame, содержащий только данные для домов, чья цена выше средней цены всех домов в исходных данных.
6. Сохранить новый DataFrame в файл rich_boston.csv

### Продвинутые варианты

## 03. Регрессия

### Начальный уровень

Скачайте датасет https://github.com/mcsml/ml_basics/blob/master/hw2.csv и, используя scikit-learn, обучите модель.

### Продвинутые варианты

Варианты того, как вы можете усложнить себе задание, если хотите попрактиковаться:

1. Использовать не готовую модель scikit-learn, а реализацию на Python.
2. Оптимизировать функции, реализованные на Python.
3. Попробовать свои силы на датасете https://www.kaggle.com/c/boston-housing

## 04. Классификация

### Начальный уровень

### Продвинутые варианты

### Финальный проект - постепенно выполняйте его по ходу курса

Шаги выполнения:

1. Зарегистрироваться на сайте kaggle.com.
2. Скачать данные train.csv из раздела "Data" (https://www.kaggle.com/c/titanic/data).
3. Разбить данные на тренировочные и тестовые.
4. Создать и обучить модель на тренировочных данных.
5. Оценить модель на датасете test.

## 05. Предобработка данных

### Начальный уровень

Предобработайте данные из датасета Titanic - https://www.kaggle.com/c/3136/download/train.csv

1. Определите, какие из данных могут быть бесполезны и избавьтесь от них.
2. Примените one-hot encoding к нечисловым данным.
3. Стандартизируйте данные.

### Продвинутые варианты

## 06. Оценка и регуляризация

### Начальный уровень

Используйте датасет кредитных карт для создании простейшей модели классификации - https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls

1. Предобработайте данные и разделите их на тренировочный и тестовый датасеты.
2. Создайте простейший классификатор без регуляризации и обучите его на тренировочных данных.
3. Создайте простейший классификатор с регуляризацией и обучите его на тех же тренировочных данных.
4. Сравните точность двух классификаторов.
5. Оцените precision и recall для каждого из результатов.

### Продвинутые варианты

## 07. Дополнительные алгоритмы

### Начальный уровень

Используя данные из файла https://github.com/mcsml/ml_basics/blob/master/bioresponse.csv, обучить случайный лес (random forest).

Target-колонка называется Activity. D1 - D1770 - features.

### Продвинутые варианты

1. Обучить XGBClassifier на этих данных.
2. Поэкспериментировать с количеством моделей в композиции (n_estimators) и глубиной деревьев (max_depth).
3. Построить графики зависимости ошибки от этих гиперпараметров для случайного леса и XGBClassifier.

## 08. Нейронные сети

### Начальный уровень

### Продвинутые варианты

## 09. Обучение без учителя

### Начальный уровень

### Продвинутые варианты

## 10. Обучение с подкреплением

### Начальный уровень

### Продвинутые варианты

## 11. Свёрточные нейронные сети

### Начальный уровень

### Продвинутые варианты

## 12. Рекуррентные нейронные сети

### Начальный уровень

### Продвинутые варианты