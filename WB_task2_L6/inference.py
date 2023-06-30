import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import re
from pymorphy2 import MorphAnalyzer
import feature_engineering as fe


m = MorphAnalyzer()  # подгрузим лемматизатор из pymorphy
stop = stopwords.words('russian') + list(punctuation)  # а также стоп-слова и знаки-пунктуации


def lemmatize_word(token, pymorphy=m):
    """функция, выполняющая лематизацию слова"""
    return pymorphy.parse(token)[0].normal_form


def lemmatize_text(text):
    """функция, выполняющая лематизацию целого текста"""
    return [lemmatize_word(w) for w in text]


def preprocess_text(text, stop_w=stop, tokenizer=word_tokenize):
    """функция предобработки текста"""
    # удаляем все, кроме букв кириллицы
    text = re.sub('[^А-Яа-яёЁ\s]', ' ', text)
    # приводим к нижнему регистру и разбиваем на слова по символу пробела
    text = tokenizer(text.lower(), language='russian')
    text = [w for w in text if (w.replace('ё', 'е') not in stop_w) and (len(w) > 2)]
    text = lemmatize_text(text)  # лемматизируем
    return ' '.join(text)


def full_preprocess(df, column_transformer, vectorizer):
    df = fe.generate_features(df)  # сгенерируем признаки
    df['clear_text'] = df.text.apply(preprocess_text)  # выполняем подготовку текстов отзывов
    vectorized_text = vectorizer.transform(df.clear_text)  # векторизация текстов
    text_df = pd.DataFrame(vectorized_text.toarray(),
                           columns=vectorizer.get_feature_names_out(),
                           index=df.index)
    df = df.drop(columns=['id1', 'id2', 'id3', 'text', 'clear_text'])  # удаляем лишние колонки
    X = pd.concat([df, text_df], axis=1)
    return column_transformer.transform(X)  # нормируем признаки


if __name__ == '__main__':
    df = pd.read_csv('sample_data.csv')  # считаем файлик с сэмплом из исходных данных

    with open('column_transformer.pkl', 'rb') as f1:
        column_transformer = pickle.load(f1)
    with open('bow_vectorizer.pkl', 'rb') as f2:  # подгрузим готовый векторайзер текстов
        vectorizer = pickle.load(f2)
    with open('knn_bow_cv1.pkl', 'rb') as f3:  # подгрузим обученную модель
        model = pickle.load(f3)

    X = full_preprocess(df, column_transformer, vectorizer)  # выполним полную подготовку данных к инференсу
    preds = model.predict(X)  # получим предсказания модели
    print(preds)  # распечатаем предсказания модели
    np.save('predictions.npy', preds)  # сохраним предсказания модели в файлик
