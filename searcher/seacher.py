from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
import string

from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()


def x_doesnt_content_number(x):
    for i in range(10):
        if f'{i}' in x:
            return False
    return True


def get_vectors():
    frm = pd.read_csv("lemmas/doc0_lemms_tdf_idf.txt", delimiter=" ", names=['word', 'idf', 'tf-idf', 'null'], )
    vects = np.zeros([100, 17124])
    lems = frm['word']
    idfs = frm['idf']
    vects[0] = np.array(frm['tf-idf'].array)
    for i in range(1, 100):
        frm = pd.read_csv(f"lemmas/doc{i}_lemms_tdf_idf.txt", delimiter=" ", names=['word', 'idf', 'tf-idf', 'null'], )
        vects[i] = np.array(frm['tf-idf'].array)
    return lems, idfs, vects


def count_tf(request, main_tokens, tf):
    simple_tokens = Counter()
    summ = 0
    text = request.lower()
    text = text.translate({ord(char): " " for char in string.punctuation})
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokenization = [word for word in tokens
                    if not word in stopwords.words('english')
                    and not word in stopwords.words('russian')
                    and len(word) > 2
                    and x_doesnt_content_number(word)]
    r = re.compile("[а-яА-Я]+")
    russian = [w for w in filter(r.match, tokenization)]
    for word in russian:
        lem = morph.normal_forms(word)[0]
        simple_tokens[lem] += 1
        summ += 1
    for key in simple_tokens.keys():
        if key in main_tokens.keys():
            tf[main_tokens[key]] = simple_tokens[key] / summ
    return tf


def count_vector(request, lems, idf):
    main_tokens = {lem: i for i, lem in enumerate(lems)}
    tf = np.zeros(len(lems))
    return idf * count_tf(request, main_tokens, tf)


def cos(a):
    def cos2(b):
        return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))

    return cos2


def search( req ):
    lems, idfs, vects = get_vectors()
    a = vects
    b = np.array(count_vector(req, lems, idfs))
    result = np.apply_along_axis(cos(b), -1, a)
    return result.argmax()