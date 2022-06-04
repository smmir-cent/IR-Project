# -*- coding: utf-8 -*-
from copy import deepcopy
import json
import math
import time
import re
import pickle
from hazm import *
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

STOPWORDS = True
STEMMING = True

dict_size = 0
token_size = 0
normalizer =  Normalizer()
stemmer = Stemmer()
json_object = None
json_object_size = 12201
stopwords = []
all_terms = []
token_size_heaps_points = []
dict_size_heaps_points = []
pos_index = {}
tfidf_weights = {}
query_tfidf_weights = {}
result_weights = {}
documents_length = {}
champion_lists = {}

heaps_points = [500, 1000, 1500, 2000, json_object_size]

def loadIndex():
    global pos_index
    f = open("assets\index.dat", "rb")
    pos_index = pickle.load(f)
    f.close()
    # print(json.dumps(pos_index, sort_keys=False, indent=4))



def readData():
    global json_object
    with open("assets\IR_data_news_12k.json", "r") as read_file:
        json_object = json.load(read_file)


def preprocessing():
    global pos_index,json_object,stopwords,json_object_size,dict_size,token_size,token_size_heaps_points,dict_size_heaps_points,STOPWORDS,STEMMING
    start_time = time.time()
    for i in range(0,json_object_size):
        # print(i)
        content = json_object[str(i)]['content'].replace("\u200c", " ")
        norm = normalizer.normalize(content)
        tokens = word_tokenize(norm)
        position = 0
        for term in tokens:
            if (not STOPWORDS) or (term not in stopwords):
                if STEMMING:
                    term = stemmer.stem(term)
                token_size += 1
                if term in pos_index:
                    pos_index[term][0] = pos_index[term][0] + 1
                    if i in pos_index[term][1]:
                        pos_index[term][1][i].append(position)
                    else:
                        pos_index[term][1][i] = [position]                    
                    
                else:    
                   pos_index[term] = []
                   pos_index[term].append(1)
                   pos_index[term].append({})     
                   pos_index[term][1][i] = [position]
                   dict_size += 1
                position += 1
        if i+1 in heaps_points:
            dict_size_heaps_points.append(math.log10(dict_size))
            token_size_heaps_points.append(math.log10(token_size))


    end_time = time.time()
    print(f'index creation time: {end_time-start_time}')
    if STOPWORDS:
        f = open("assets\index.dat", "wb")
        pickle.dump(pos_index, f)
        f.close()



def ranking(all_terms,final_res):
    # print(final_res)
    display_pos = {}
    for docid in final_res:
        display_pos[docid] = []
    for term in all_terms:
        for docid in final_res:
            try:
                if docid in pos_index[term][1]:
                    display_pos[docid].append(int(sum(pos_index[term][1][docid])/len(pos_index[term][1][docid])))
            except:
                pass
    return sorted(final_res, key=final_res.get, reverse=True)[:5],display_pos



def queryProcessing(query):
    global all_terms,json_object_size,query_tfidf_weights,tfidf_weights,result_weights
    ######### preprocessing #########
    phrases = re.findall('"([^"]*)"', query)
    terms = re.sub('"([^"]*)"', '', query)
    norm = normalizer.normalize(terms)
    tokens = word_tokenize(norm)
    _not = False
    not_terms = []
    for item in deepcopy(tokens):
        # print(item)
        if item == '!':
            _not = True
        if item not in stopwords and item != '!':
            stemming = stemmer.stem(item)
            if _not:
                not_terms.append(stemming)
                tokens.remove(item)
                _not = False
            else:
                ind = tokens.index(item)
                tokens = tokens[:ind]+[stemming]+tokens[ind+1:]
        else:
            tokens.remove(item)

    print("and_terms")
    counters = dict(Counter(tokens))
    for item in tokens:
        print(item)
    # print(counters)

    ########### calculating tfidf_weights ###########
    docid_candidate = []
    for term in counters.keys():
        try:
            n_t = len(pos_index[term][1])
            query_tfidf_weights[term] = (1 + math.log10(counters[term])) * math.log10(json_object_size/n_t)
            docid_candidate = list(set(docid_candidate) | set(champion_lists[term]))
        except:
            n_t = 0
            query_tfidf_weights[term] = 0
    ########### /calculating tfidf_weights ###########

    for term in counters.keys():
        try:
            for docid in tfidf_weights[term]:
                if docid in docid_candidate:
                    if docid in result_weights:
                        result_weights[docid] += query_tfidf_weights[term] * tfidf_weights[term][docid]
                    else:
                        result_weights[docid] = query_tfidf_weights[term] * tfidf_weights[term][docid]
        except:
            pass
    for docid in result_weights.keys():
         result_weights[docid] /= math.sqrt(documents_length[docid])
    



    ######### /preprocessing #########

    ranked_doc,display_pos = ranking(all_terms=counters.keys(),final_res=result_weights)
    print(ranked_doc)
    f = open("result.txt", "w", encoding='utf-8')
    for i,doc in enumerate(ranked_doc):
        if len(display_pos[doc]) == 0:
            display_center = 0
        else:    
            display_center = int(sum(display_pos[doc])/len(display_pos[doc]))
        print(display_center)
        url = json_object[str(doc)]['url']
        title = json_object[str(doc)]['title']
        content = json_object[str(doc)]['content']
        norm = normalizer.normalize(content)
        tokens = word_tokenize(norm)
        counter = 0
        index = 0
        thr = 0
        while counter < len(tokens):
            index+=(len(tokens[counter]))
            if tokens[counter] not in stopwords:
                stemming = stemmer.stem(item)
                tokens[counter] = stemming
                thr += 1
            else:
                tokens[counter] = ''
            if thr > display_center:
                break
            counter += 1
        f.write(str(i+1)+") score "+str(result_weights[doc])+" \n\n")
        f.write("title:  "+title+"\n\n")
        f.write("url:  "+url+"\n\n")
        f.write(content[max(index-100,0):min(index+100,len(content))]+"\n\n")
        f.write("############################################\n\n")
    f.close()



def loadStopwords():
    global stopwords
    textfile = open("assets\hazm_stopwords.txt", "r",encoding="utf-8")
    for line in textfile:
        currentPlace = line[:-1]
        stopwords.append(currentPlace.replace("\u200c", " "))
    textfile.close()        



def tfidfCal():
    global pos_index,tfidf_weights,documents_length,champion_lists
    # print("before cal len(pos_index)",len(pos_index))
    for term in pos_index.keys():
        temp = pos_index[term]
        tfidf_weights[term] = {}
        n_t = len(temp[1])
        for keys_docs in temp[1].keys():
            doc_term = temp[1][keys_docs]
            f_td =  len(doc_term)
            tfidf_weights[term][keys_docs] = (1 + math.log10(f_td))*math.log10(json_object_size/n_t)
            if keys_docs in documents_length:
                documents_length[keys_docs] += tfidf_weights[term][keys_docs]**2
            else:
                documents_length[keys_docs] = tfidf_weights[term][keys_docs]**2
        ### create and sort tfidf champion_lists
        champion_lists[term] = sorted(tfidf_weights[term], key=tfidf_weights[term].get, reverse=True)[:75]
    # print("after cal len(pos_index)",len(pos_index))




if __name__ == "__main__":
    loadStopwords()
    readData()
    # preprocessing()
    loadIndex()
    tfidfCal()
    while True:
        query = input('your query: ')
        queryProcessing(query)
