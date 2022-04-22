import json
from hazm import *
import time


normalizer = Normalizer()
# normalizer = parsivar.Normalizer()
stemmer = Stemmer()
# stemmer = parsivar.FindStems()
json_object = None
json_object_size = 12202
stopwords = list(set(stopwords_list()))
positional_index = {}


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
    global positional_index,json_object,stopwords,json_object_size
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
        for item in tokens:
            position += 1
            if item not in stopwords:
                stemming = stemmer.stem(item)
                if stemming not in positional_index:
                    positional_index[stemmer.stem(item)] = emptyPositionalList(True)
                positional_index[stemmer.stem(item)]["count"] += 1 
                if i not in positional_index[stemmer.stem(item)]["occurrence"]: 
                    positional_index[stemmer.stem(item)]["occurrence"][i] = emptyPositionalList(False)
                positional_index[stemmer.stem(item)]["occurrence"][i]["count"] += 1
                positional_index[stemmer.stem(item)]["occurrence"][i]["occurrence"].append(position)
    end_time = time.time()
    print(f'index creation time: {end_time-start_time}')
    with open('data.json', 'w', encoding="utf-8") as f:
        # json.dump(positional_index, f, indent=4, ensure_ascii=False)
        json.dump(positional_index, f, ensure_ascii=False)

if __name__ == "__main__":
    readData()
    # print(stopwords)
    preprocessing()
