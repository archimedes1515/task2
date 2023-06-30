import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import pandas as pd


def avg_word_len(s: str) -> float:
    """функция для расчета средней длины слова в отзыве"""
    return np.mean(list(map(len, word_tokenize(s, language='russian'))))


def avg_sent_len(s: str) -> float:
    """функция для расчета средней длины предложения в отзыве"""
    return np.mean(list(map(len, sent_tokenize(s, language='russian'))))


def caps_portion(s: str, text_len: int) -> float:
    """функция для расчета доли капса в отзыве"""
    l = re.findall('[А-ЯA-Z]+', s)
    return sum(map(len, l)) / text_len


def punct_portion(s: str, text_len: int) -> float:
    """функция для расчета доли пунктуации в отзыве"""
    l = re.findall(r'[!"#$%&\'()*+,-./:;<=>?@\\^_`{|}~]+', s)
    return sum(map(len, l)) / text_len


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """функция для генерации дополнительных фичей"""
    df['text_len'] = df.text.str.len()
    df['word_count'] = df.text.apply(lambda x: len(word_tokenize(x, language='russian')))
    df['sent_count'] = df.text.apply(lambda x: len(sent_tokenize(x, language='russian')))
    df['avg_word_len'] = df.text.apply(avg_word_len)
    df['avg_sent_len'] = df.text.apply(avg_sent_len)
    df['caps_portion'] = df.apply(lambda x: caps_portion(x.text, x.text_len), axis=1)
    df['punct_portion'] = df.apply(lambda x: punct_portion(x.text, x.text_len), axis=1)
    df['f9'] = df['f5'] / df['f4']
    df['f10'] = df['f2'] / df['f1']
    return df
