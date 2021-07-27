import os
import json

# index merge

def get_data(corpus_path):
    with open('news_index/' + corpus_path, 'r') as js:
        dataset = json.load(js)
    return dataset

## main ##

file_name_list = os.listdir('news_index/')

data_sum = []
data_index = 0

for i in range(len(file_name_list)):
    data = get_data(file_name_list[i])
    data_sum = data_sum + data
    print("processing.." + file_name_list[i])
    print(len(data))
    print(len(data_sum))

    if len(data_sum) > 700000:
        with open("news_merge/news_" + str(data_index), 'w') as make:
            json.dump(data_sum, make)
        print("news_merge/news_" + str(data_index) + " save    len = " + str(len(data_sum)))
        data_sum=[]
        data_index += 1

if len(data_sum) > 0:
    with open("news_merge/news_" + str(data_index), 'w') as make:
        json.dump(data_sum, make)
    print("news_merge/news_" + str(data_index) + " save    len = " + str(len(data_sum)))

    data_sum = []
    data_index += 1

