import re
from gensim.summarization import summarize
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.sparse.linalg import svds
import networkx
import matplotlib.pyplot as plt
import time
import os


def summary(document):
    DOCUMENT = document
    DOCUMENT = re.sub(r'\n|\r', ' ', DOCUMENT)
    DOCUMENT = re.sub(r' +', ' ', DOCUMENT)
    DOCUMENT = DOCUMENT.strip()


    sentences = nltk.sent_tokenize(DOCUMENT)

    stop_words = nltk.corpus.stopwords.words('english')

    def normalize_document(doc):
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize document
        tokens = nltk.word_tokenize(doc)
        # filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create document from filtered tokens
        doc = ' '.join(filtered_tokens)
        return doc

    normalize_corpus = np.vectorize(normalize_document)

    norm_sentences = normalize_corpus(sentences)

    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(norm_sentences)
    dt_matrix = dt_matrix.toarray()

    vocab = tv.get_feature_names()
    td_matrix = dt_matrix.T


    def low_rank_svd(matrix, singular_count=2):
        u, s, vt = svds(matrix, k=singular_count)
        return u, s, vt

    num_sentences = int(len(sentences)*0.4)
    num_topics = 3

    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
    term_topic_mat, singular_values, topic_document_mat = u, s, vt

    # remove singular values below threshold                                         
    sv_threshold = 0.5
    min_sigma_value = max(singular_values) * sv_threshold
    singular_values[singular_values < min_sigma_value] = 0

    salience_scores = np.sqrt(np.dot(np.square(singular_values), 
                                     np.square(topic_document_mat)))

    top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
    top_sentence_indices.sort()

    lsa = '\n'.join(np.array(sentences)[top_sentence_indices])

    similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)

    similarity_graph = networkx.from_numpy_array(similarity_matrix)
    d = "static/graph"
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        if os.path.isfile(full_path):
            os.remove(full_path)


    plt.figure(figsize=(12, 6))
    networkx.draw_networkx(similarity_graph, node_color='lime')
    ts = time.time()
    imur='graph/'+str(ts)+'.png'
    plt.savefig('static/'+ imur)
    
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score 
                                            in scores.items()), 
                              reverse=True)

    top_sentence_indices = [ranked_sentences[index][1] 
                            for index in range(num_sentences)]
    top_sentence_indices.sort()

    tr = '\n'.join(np.array(sentences)[top_sentence_indices])

    return lsa , tr , imur
