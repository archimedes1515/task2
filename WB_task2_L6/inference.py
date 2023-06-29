import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import re
from pymorphy2 import MorphAnalyzer
import feature_engineering as fe


m = MorphAnalyzer()
stop = stopwords.words('russian') + list(punctuation)


def lemmatize_word(token, pymorphy=m):
    return pymorphy.parse(token)[0].normal_form


def lemmatize_text(text):
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
    df = fe.generate_features(df)
    df['clear_text'] = df.text.apply(preprocess_text)
    vectorized_text = vectorizer.transform(df.clear_text)
    text_df = pd.DataFrame(vectorized_text.toarray(),
                           columns=vectorizer.get_feature_names_out(),
                           index=df.index)
    df = df.drop(columns=['id1', 'id2', 'id3', 'text', 'clear_text'])
    X = pd.concat([df, text_df], axis=1)
    return column_transformer.transform(X)


if __name__ == '__main__':
    df = pd.read_csv('sample_data.csv')

    with open('column_transformer.pkl', 'rb') as f1:
        column_transformer = pickle.load(f1)
    with open('bow_vectorizer.pkl', 'rb') as f2:
        vectorizer = pickle.load(f2)
    with open('knn_bow_cv1.pkl', 'rb') as f3:
        model = pickle.load(f3)

    X = full_preprocess(df, column_transformer, vectorizer)
    preds = model.predict(X)
    print(preds)
    # np.save('predictions.npy', preds)
