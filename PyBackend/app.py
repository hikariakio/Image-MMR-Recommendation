from flask import Flask, request, jsonify,g
from flask_cors import CORS
import logging

import json,random
from json_getter import extract_names_from_json
from operator import itemgetter


import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import string
from sklearn.metrics.pairwise import cosine_similarity
from json_getter import extract_names_from_json
from gensim.models import KeyedVectors

import os
import gensim.downloader as api
from gensim import utils

stop_words = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)

sentences,sen_list,id_list,model_word2vec,model_glove,matrix_word2vec,matrix_glove = None,None,None,None,None,None,None

def SETUP():
    global sentences,sen_list,id_list,model_word2vec,model_glove,matrix_word2vec,matrix_glove
    sentences = extract_names_from_json("val2017/jsonAnnotation.json")
    sen_list = list(map(itemgetter(0), sentences))
    id_list = list(map(itemgetter(1), sentences))

    print("LOADING MODEL word2vec");
    model_word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True)

    print("LOADING MODEL glove");
    model_glove = api.load("glove-wiki-gigaword-300")

    print("SIM MATRIX word2vec");
    matrix_word2vec = np.loadtxt("Sim_matrix_word2vec.csv", delimiter=",")
    print("SIM MATRIX glove");
    matrix_glove = np.loadtxt("Sim_matrix_glove.csv", delimiter=",")

# def get_sentences():
#     if 'sentences' not in g:
#         g.sentences = extract_names_from_json("val2017/jsonAnnotation.json")
#     return g.sentences
#
# def get_model():
#     if 'model' not in g:
#         print("model start loading")
#         g.model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary = True)
#         print("model finish loading")
#     return g.model
#
# def get_simmatrix():
#     if 'matrix' not in g:
#         g.matrix = np.loadtxt("Sim_matrix_word2vec.csv", delimiter=",")
#     return g.matrix

def id_to_name(id):
    with open('id_name_dict.json', 'r') as f:
        id_name_dict = json.load(f)
    return (id_name_dict[str(id)])

def wordList_from_sentence(sentence):
    tokens = word_tokenize(sentence.lower())
    # Filter out puncts and common stopwords
    cleaned_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    return cleaned_tokens

def vector_from_wordlist(wordList,model):
    vectors = [model[word] for word in wordList if word in model]
    sentence_vector = np.mean(vectors, axis=0)
    return  sentence_vector

def vector_from_sentence(sentence,model):
    return vector_from_wordlist(wordList_from_sentence(sentence.lower()),model);


def similarity_score(vec1,vec2):
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity

def MMR_score_cache_max(target, sentences, vec_model, image_id, sim_matrix, lambda_value=0.5):
    scores = []
    if isinstance(target, str):
        target = wordList_from_sentence(target)

    target_vector = vector_from_wordlist(target, vec_model)

    for i, sentence in enumerate(sentences):
        sentence_vector = vector_from_wordlist(wordList_from_sentence(sentence), vec_model)

        rel_score = similarity_score(target_vector, sentence_vector)

        sentence_similarties = sim_matrix[i, :]
        sentence_similarties_without_i = np.delete(sentence_similarties, i)
        max_sim_score = np.max(sentence_similarties_without_i)
        mmr_score = (lambda_value * rel_score) - ((1 - lambda_value) * max_sim_score)

        scores.append((sentence, image_id[i], rel_score, max_sim_score, mmr_score))

    return scores

def MMR_score_cache_average (target, sentences, vec_model,image_id, sim_matrix, lambda_value = 0.5):
    scores= []
    if isinstance(target, str):
        target = wordList_from_sentence(target)

    target_vector = vector_from_wordlist(target, vec_model)

    for i,sentence in enumerate(sentences):
        sentence_vector = vector_from_wordlist(wordList_from_sentence(sentence), vec_model)

        rel_score = similarity_score(target_vector, sentence_vector)

        sentence_similarties = sim_matrix[i,:]
        sentence_similarties_without_i = np.delete(sentence_similarties,i)
        max_sim_score = np.mean(sentence_similarties_without_i)
        mmr_score = (lambda_value * rel_score)  - ((1-lambda_value) * max_sim_score)

        scores.append((sentence,image_id[i],rel_score,max_sim_score,mmr_score))

    return  scores

@app.route('/api/random-images', methods=['GET'])
def random_images():
    with open('id_name_dict.json', 'r') as f:
        id_name_dict = json.load(f)

    ids_list = list(id_name_dict.keys())
    ids_list = [int(id) for id in ids_list]
    random.shuffle(ids_list)
    top_50_ids = ids_list[:50]

    return jsonify(top_50_ids)  # Return the list as a JSON response

@app.route('/api/log-image-click', methods=['POST'])
def log_image_click():
    data = request.json
    image_id = data['id']
    lambda_value = data['lambda_value']
    model = data['model']
    mode = data['mode']

    clicked_image_name = (id_to_name(image_id))

    print(clicked_image_name, data)

    if(model == 'GoogleNews-word2vec' and mode == 'Maximum'):
        mmr_score_list = MMR_score_cache_max(target=clicked_image_name, sentences=sen_list, image_id=id_list, vec_model=model_word2vec, sim_matrix=matrix_word2vec, lambda_value=float(lambda_value))
    elif (model == 'GoogleNews-word2vec' and mode == 'Average'):
        mmr_score_list = MMR_score_cache_average(target=clicked_image_name, sentences=sen_list, image_id=id_list,
                                                 vec_model=model_word2vec, sim_matrix=matrix_word2vec,
                                                 lambda_value=float(lambda_value))
    elif (model == 'Wiki-gigaword-glove' and mode == 'Maximum'):
        mmr_score_list = MMR_score_cache_max(target=clicked_image_name, sentences=sen_list, image_id=id_list,
                                                     vec_model=model_glove, sim_matrix=matrix_glove,
                                                     lambda_value=float(lambda_value))
    elif (model == 'Wiki-gigaword-glove' and mode == 'Average'):
        mmr_score_list = MMR_score_cache_average(target=clicked_image_name, sentences=sen_list,
                                                         image_id=id_list, vec_model=model_glove,
                                                         sim_matrix=matrix_glove, lambda_value=float(lambda_value))
    sorted_mmr_score = sorted(mmr_score_list, key=lambda x: x[4], reverse=True)

    sorted_mmr_score = sorted_mmr_score[0:50]
    show_list = list(map(itemgetter(1), sorted_mmr_score))

    return jsonify(show_list)

if __name__ == '__main__':
    SETUP()
    app.run(debug=True,port=5000,use_reloader=False)



