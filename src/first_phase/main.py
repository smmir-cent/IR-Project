# -*- coding: utf-8 -*-
from copy import deepcopy
import json
import time
import re
import pickle
from hazm import *


def loadIndex():
    global pos_index
    f = open("assets\index.dat", "rb")
    pos_index = pickle.load(f)
    f.close()
    # print(json.dumps(pos_index, sort_keys=False, indent=4))

normalizer =  Normalizer()
stemmer = Stemmer()
json_object = None
json_object_size = 12201
stopwords = []
all_terms = []
pos_index = {}


def readData():
    global json_object
    with open("assets\IR_data_news_12k.json", "r") as read_file:
        json_object = json.load(read_file)
    # docs = pd.read_json('assets\IR_data_news_12k.json', encoding='utf-8')
    # docs_contents = docs.T['content']
    # docs_titles = docs.T['title']



    # f = open("temp.txt", "w", encoding='utf-8')
    # f.write(docs_contents[6929])
    # f.close()

    # print(docs_contents[6929])



    # json_object = {}
    # json_object['0'] = {}
    # json_object['0']['content'] = "These are short, famous texts in English from classic sources like the Bible or Shakespeare. Some texts have word definitions and explanations to help you. Some of these texts are written in an old style of English. Try to understand them, because the English that we speak today is based on what our great, great, great, great grandparents spoke before! Of course, not all these texts were originally written in English. The Bible, for example, is a translation. But they are all well known in English today, and many of them express beautiful thoughts."

def preprocessing():
    global pos_index,json_object,stopwords,json_object_size
    start_time = time.time()
    for i in range(0,json_object_size):
        # print(i)
        content = json_object[str(i)]['content'].replace("\u200c", " ")
        # url = json_object[str(i)]['url']
        # title = json_object[str(i)]['title']
        norm = normalizer.normalize(content)
        tokens = word_tokenize(norm)
        position = 0
        for term in tokens:
            if term not in stopwords:
                term = stemmer.stem(term)
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
                position += 1
                

    end_time = time.time()
    print(f'index creation time: {end_time-start_time}')
    # print(json.dumps(pos_index,indent=4))
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
    # res = res + pl1[counter_1:] + pl2[counter_2:]
    return res



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
    # res = res + l1[counter_1:] + l2[counter_2:]
    return state

def ranking(all_terms,final_res):
    ranks = {}
    display_pos = {}
    for docid in final_res:
        ranks[docid] = 0
        display_pos[docid] = []
    # print("all_terms")
    # print(all_terms)
    # print("###########################")
    # print(pos_index[all_terms[0]][1])
    # print("###########################")
    # print(pos_index[all_terms[1]][1])
    for term in all_terms:
        for docid in final_res:
            ranks[docid] += len(pos_index[term][1][docid])
            display_pos[docid].append(int(sum(pos_index[term][1][docid])/len(pos_index[term][1][docid])))
    # print(ranks)
    return sorted(ranks, key=ranks.get, reverse=True)[:5],display_pos


def execPhrases(phrases):
    global all_terms
    phrase_result = []
    first_phrase = True
    # print(phrases)
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
        # print("docid_candidates")
        # print(docid_candidates) ### ok:6929
        # print("/docid_candidates")
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
    # bug: if '!' comes first does not work properly 
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
    for item in tokens:
        print(item)
    print("not_terms:")
    for item in not_terms:
        print(item)
    print("phrases:")
    for item in phrases:
        print(item)

    ### and_terms ###
    try:
        tokens = list(dict.fromkeys(tokens))
        final_res = list(pos_index[tokens[0]][1].keys())
        i = 1
        while  i < len(tokens):
            final_res = merging(final_res,list(pos_index[tokens[i]][1].keys()))
            i += 1
    except:
        final_res = []
    print("######### and_terms #########")  
    print(final_res)
    ### /and_terms ###
    all_terms.extend(tokens)


    # ### phrases ###
    phrase_result = []
    if len(phrases) != 0:
        phrase_result = execPhrases(phrases=phrases)
        print("######### phrase_result #########") 
        print(phrase_result)
        ## merge phrases and and_terms
        if len(tokens) != 0:
            final_res = sorted(list(set(phrase_result) & set(final_res)))
        else:
            final_res = deepcopy(phrase_result)
        ## /merge phrases and and_terms
    # ### /phrases ###

    ### not_terms ###
    not_terms = list(dict.fromkeys(not_terms))
    if len(final_res) != 0:
        for doc in deepcopy(final_res):
            for term in not_terms:
                if doc in list(pos_index[term][1].keys()):
                    print(doc)
                    final_res.remove(doc)  
    print("######################## final(without ranking) ########################")  
    print(final_res)  
    ## extract not from result
    ### /not_terms ###

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
        f.write(content[index-50:index+50]+"\n\n")
        f.write("############################################\n\n")
    f.close()



def loadStopwords():
    global stopwords
    textfile = open("assets\hazm_stopwords.txt", "r",encoding="utf-8")
    for line in textfile:
        currentPlace = line[:-1]
        stopwords.append(currentPlace.replace("\u200c", " "))
    textfile.close()        


if __name__ == "__main__":
    loadStopwords()
    readData()
    # preprocessing()
    loadIndex()
    while True:
        query = input('your query: ')
        queryProcessing(query)


    # print(stopwords)
    # textfile = open("assets\hazm_stopwords.txt", "w",encoding="utf-8")
    # for element in stopwords:
    #     textfile.write(element + "\n")
    # textfile.close()    
    # stopwords = []
    # print(stopwords)
    # print("###############################################################")