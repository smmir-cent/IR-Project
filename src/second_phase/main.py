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
from scipy.optimize import curve_fit
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


def merging(pl1,pl2):
    counter_1, counter_2 = 0, 0
    size_1, size_2 = len(pl1), len(pl2)
    res = []
    while counter_1 < size_1 and counter_2 < size_2:
        if pl1[counter_1] == pl2[counter_2]:
            res.append(pl1[counter_1])
            counter_1 += 1
            counter_2 += 1
        elif pl1[counter_1] < pl2[counter_2]:
            counter_1 += 1
        else:
            counter_2 += 1
    return res


def softMerging(pl1,pl2):
    pl1.extend(pl2)
    return pl1

def checkSeq(l1,l2):
    counter_1, counter_2 = 0, 0
    size_1, size_2 = len(l1), len(l2)
    state = False
    while counter_1 < size_1 and counter_2 < size_2:
        if l1[counter_1] + 1 == l2[counter_2]:
            state = True
            counter_1 += 1
            counter_2 += 1
        elif l1[counter_1] < l2[counter_2]:
            counter_1 += 1
        else:
            counter_2 += 1
    return state

def ranking(all_terms,final_res):
    ranks = {}
    display_pos = {}
    for docid in final_res:
        ranks[docid] = 0
        display_pos[docid] = []
    for term in all_terms:
        for docid in final_res:
            if docid in pos_index[term][1]:
                ranks[docid] += len(pos_index[term][1][docid])
                display_pos[docid].append(int(sum(pos_index[term][1][docid])/len(pos_index[term][1][docid])))
    return sorted(ranks, key=ranks.get, reverse=True)[:5],display_pos


def execPhrases(phrases):
    global all_terms
    phrase_result = []
    first_phrase = True
    for phrase in phrases:
        norm = normalizer.normalize(phrase)
        tokens = word_tokenize(norm)
        docid_candidates = []
        for item in deepcopy(tokens):
            if item not in stopwords:
                stemming = stemmer.stem(item)
                ind = tokens.index(item)
                tokens = tokens[:ind]+[stemming]+tokens[ind+1:]
            else:
                tokens.remove(item)
        try:
            docid_candidates = list(pos_index[tokens[0]][1].keys())
            i = 1
            while  i < len(tokens):
                docid_candidates = merging(docid_candidates,list(pos_index[tokens[i]][1].keys()))
                i += 1
        except:
            docid_candidates = []
        all_terms.extend(tokens)
        counter = 0
        res = []
        while counter < len(tokens) - 1:
            for docid in docid_candidates:
                l1 = pos_index[tokens[counter]][1][docid]
                l2 = pos_index[tokens[counter+1]][1][docid]
                state = checkSeq(l1,l2)
                if state:
                    res.append(docid)
            counter += 1
        if first_phrase:
            first_phrase = False
            phrase_result = res
        phrase_result = sorted(list(set(phrase_result) & set(res))) 
    return phrase_result


def queryProcessing(query):
    global all_terms
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

    query_tfidf_weights = {}
    for term in counters.keys():
        n_t = len(pos_index[term][1])
        query_tfidf_weights[term] = (1 + math.log10(counters[term])) * math.log10(json_object_size/n_t)
    ########### /calculating tfidf_weights ###########

    ### and_terms ###
    try:
        tokens = list(dict.fromkeys(tokens))
        final_res = list(pos_index[tokens[0]][1].keys())
        i = 1
        while  i < len(tokens):
            final_res = softMerging(final_res,list(pos_index[tokens[i]][1].keys()))
            i += 1
    except:
        final_res = []
    # print("######### and_terms #########")  
    z = Counter(final_res)
    # print(z)
    final_res = []
    for key in z:
        if z[key] > len(tokens) - 2:
            final_res.append(key)
    final_res = sorted(final_res)
    ### /and_terms ###


    ######### /preprocessing #########

    ranked_doc,display_pos = ranking(all_terms=all_terms,final_res=final_res)
    f = open("result.txt", "w", encoding='utf-8')
    for i,doc in enumerate(ranked_doc):
        display_center = int(sum(display_pos[doc])/len(display_pos[doc]))
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
        f.write(str(i+1)+") \n\n")
        f.write("title:  "+title+"\n\n")
        f.write("url:  "+url+"\n\n")
        f.write(content[index-100:index+100]+"\n\n")
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
    global pos_index,tfidf_weights
    print("before cal len(pos_index)",len(pos_index))
    # for _ in range(0,1000):
    #     print(temp)
    #     print(temp[0])
    #     print(temp[1])
    for term in pos_index.keys():
        temp = pos_index[term]
        
        #### temp[1][1].keys: term,doc
        tfidf_weights[term] = {}
        n_t = len(temp[1])
        for keys_docs in temp[1].keys():
            doc_term = temp[1][keys_docs]
            doc_id = doc_term[0] 
            f_td =  len(doc_term[1])
            tfidf_weights[term][doc_id] = (1 + math.log10(f_td))*math.log10(json_object_size/n_t)
    print("after cal len(pos_index)",len(pos_index))




if __name__ == "__main__":
    loadStopwords()
    readData()
    # preprocessing()
    loadIndex()
    tfidfCal()
    while True:
        query = input('your query: ')
        queryProcessing(query)
