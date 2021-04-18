import numpy as np
import random as random
import matplotlib.pyplot as plt
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

"""with open('stop_words_english.txt', 'r') as f:
    stopwords = f.read().split('\n')"""

# Stem and remove stop words


def gather_data():
    path = '../data/'
    train_dir = path+'20news-bydate-train/'
    test_dir = path+'20news-bydate-test/'
    newsgroup_list = os.listdir(train_dir)
    # print(newsgroup_list)
    newsgroup_list.sort()
    stemmer = PorterStemmer()

    def collect_data_from(parent_dir):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir+newsgroup+'/'
            files = [(filename, dir_path+filename)
                     for filename in os.listdir(dir_path)
                     if os.path.isfile(dir_path+filename)]
            files.sort()
            u = len(files)
            it = 0
            for filename, filepath in files:
                with open(filepath, 'r') as f:
                    it += 1
                    if it % 10 == 0:
                        print(str(it)+'/'+str(u))
                    content = f.read().lower()
                    words = [stemmer.stem(word)
                             for word in re.split('\W+', content)
                             if word not in stopwords]
                    content = ' '.join(words)
                    data.append(str(label)+'<fff>'+filename+'<fff>'+content)
        return data

    train_data = collect_data_from(train_dir)
    with open('../data/20news-train-preprocessed.txt', 'w') as f:
        f.write('\n'.join(train_data))
    test_data = collect_data_from(test_dir)
    with open('../data/20news-test-preprocessed.txt', 'w') as f:
        f.write('\n'.join(test_data))
    full_data = train_data+test_data
    with open('../data/20news-full-preprocessed.txt', 'w') as f:
        f.write('\n'.join(full_data))
# gather_data()

# Get vocab and calculate idf


def generate_vocabulary(data_path):
    with open(data_path, 'r') as f:
        lines = f.read().split('\n')
    doc_count = {}
    corpus_size = len(lines)
    output = []
    for line in lines:
        features = line.split('<fff>')
        words = set(features[-1].split(' '))
        for word in words:
            if word in doc_count.keys():
                doc_count[word] += 1
            else:
                doc_count[word] = 1
    for word, df in doc_count.items():
        if df > 10 and re.match('[a-zA-Z]+', word):
            output.append(word+'<fff>'+str(np.log10(corpus_size/df)))
    output.sort()
    u = open('words_idf.txt', 'w')
    content = '\n'.join(output)
    u.write(content)

    # print('Success')
# generate_vocabulary('../data/20news_train_preprocessed.txt')

# Get tf-idf


def get_tf_idf(data_path):
    import numpy as np
    with open('words_idf.txt', 'r') as u:
        data = u.read().split('\n')
        word_idf = [(x.split('<fff>')[0], float(x.split('<fff>')[1]))
                    for x in data]
        word_ID = dict([(word, index)
                       for index, (word, idf) in enumerate(word_idf)])
        idf = dict(word_idf)
    with open(data_path, 'r') as f:
        lines = f.read().split('\n')
    content = []
    for line in lines:
        document = line.split('<fff>')
        words = [word for word in document[-1].split(' ') if word in idf]
        tf = dict()
        max_tf = 0
        for word in words:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
            max_tf = max(tf[word], max_tf)
        res = document[0]+'<fff>'+document[1]+'<fff>'
        sum_squares = 0
        word_tf_idf = []
        for word in tf.keys():
            tf_idf_value = tf[word]*1./max_tf*idf[word]
            sum_squares += tf_idf_value**2
            word_tf_idf.append((word_ID[word], tf_idf_value))
        word_tfidf_normalize = [str(index)+':'+str(tf_idf_value/np.sqrt(sum_squares))
                                for index, tf_idf_value in word_tf_idf]
        res = res+' '.join(word_tfidf_normalize)
        content.append(res)
    with open('tf_idf.txt', 'w') as r:
        r.write('\n'.join(content))


get_tf_idf('../data/20news_train_preprocessed.txt')
