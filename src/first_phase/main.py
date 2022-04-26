import json
from hazm import *
import time
import re
import pickle


def loadIndex():
    global pos_index
    f = open("assets\index.dat", "rb")
    pos_index = pickle.load(f)
    f.close()
    # print(json.dumps(pos_index, sort_keys=False, indent=4))

normalizer = Normalizer()
# normalizer = parsivar.Normalizer()
stemmer = Stemmer()
# stemmer = parsivar.FindStems()
json_object = None
json_object_size = 12202
stopwords = list(set(stopwords_list()))
pos_index = {}


def emptyPositionalList(occurrence):
    if occurrence:
        return {"count":0,"occurrence":{}}
    else:
        return {"count":0,"occurrence":[]}

def readData():
    global json_object
    with open("assets\IR_data_news_12k.json", "r") as read_file:
        json_object = json.load(read_file)


def preprocessing():
    global pos_index,json_object,stopwords,json_object_size
    start_time = time.time()
    for i in range(0,json_object_size):
        # print(i)
        content = json_object[str(i)]['content']
        # url = json_object[str(i)]['url']
        # title = json_object[str(i)]['title']
        norm = normalizer.normalize(content)
        tokens = word_tokenize(norm)
        # print(tokens)
        position = 0
        for pos, term in enumerate(tokens):
            position += 1
            if term not in stopwords:
                term = stemmer.stem(term)
                if term in pos_index:
                    pos_index[term][0] = pos_index[term][0] + 1
                    if i in pos_index[term][1]:
                        pos_index[term][1][i].append(pos)
                         
                    else:
                        pos_index[term][1][i] = [pos]                    
                else:    
                   pos_index[term] = []
                   pos_index[term].append(1)
                   pos_index[term].append({})     
                   pos_index[term][1][i] = [pos]

    end_time = time.time()
    print(f'index creation time: {end_time-start_time}')
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


def mergingPhrases(pl1,pl2):
    counter_1, counter_2 = 0, 0
    size_1, size_2 = len(pl1), len(pl1)
    res = []
    ## todo
    while counter_1 < size_1 and counter_2 < size_2:
        if pl1[counter_1] == pl2[counter_2]:
            res.append(pl1[counter_1])
            counter_1 += 1
            counter_2 += 1
        if pl1[counter_1] < pl2[counter_2]:
            counter_1 += 1
        else:
            counter_2 += 1
    # res = res + pl1[counter_1:] + pl2[counter_2:]
    return res



def execPhrases(phrases):
    phrases_res = []
    phrases = list(dict.fromkeys(phrases))
    phrases_res = list(phrases[0]["occurrence"].keys())
    i = 1
    while  i < len(phrases):
        phrases_res = mergingPhrases(phrases_res,list(phrases[i]["occurrence"].keys()))
    


def queryProcessing(query):
    ######### preprocessing #########
    phrases = re.findall('"([^"]*)"', query)
    # print(phrase)
    terms = re.sub('"([^"]*)"', '', query)
    norm = normalizer.normalize(terms)
    # print(f'normalized = {norm}')
    tokens = word_tokenize(norm)
    # print(f'"tokens: {tokens}')
    _not = False
    not_terms = []
    # bug: if '!' comes first does not work properly 
    for item in tokens:
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
    tokens = list(dict.fromkeys(tokens))
    print("tokens")
    print(tokens)
    final_res = list(pos_index[tokens[0]][1].keys())
    print("final_res[0]")
    print(final_res)
    print("final_res[1]")
    print(list(pos_index[tokens[1]][1].keys()))

    i = 1
    while  i < len(tokens):
        final_res = merging(final_res,list(pos_index[tokens[i]][1].keys()))
        i += 1
    print("final_res")
    print(final_res)
    print("final_res")
    ### /and_terms ###

    # ### not_terms ###
    # not_res = []
    # not_terms = list(dict.fromkeys(not_terms))
    # not_res = list(not_terms[0]["occurrence"].keys())
    # i = 1
    # while  i < len(not_terms):
    #     not_res = merging(not_res,list(not_terms[i]["occurrence"].keys()))    
    # ### /not_terms ###

    # ### phrases ###
    # execPhrases(phrases=phrases)
    # ## todo: check position
    # ### /phrases ###


    ######### /preprocessing #########

    

            
    


if __name__ == "__main__":
    readData()
    # preprocessing()
    loadIndex()
    query = input('your query: ')
    queryProcessing(query)
