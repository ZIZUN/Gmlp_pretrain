import json
from transformers import ElectraTokenizer
from kss import split_sentences
import tqdm
import os

def news_to_index(file_name, seq_len):
    out_of_length = 0
    with open("news/"+ file_name , 'r') as js:
        news = json.load(js)

        doc_index = 0
        doc_list_index = 0

        tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

        start_index = 2
        sep_index = 3

        input = [start_index]

        split_news = []
        dataset = []

        for i in tqdm.tqdm(range(len(news['document']))):
            doc = []
            for j in range(len(news['document'][i]['paragraph'])):

                split_sen = split_sentences(news['document'][i]['paragraph'][j]['form'])
                for k in range(len(split_sen)):
                    doc.append(split_sen[k])
            split_news.append(doc)

        #    split_news     -> all news(sentence splited)


        doc_len = len(split_news)

        exit = False
        while (True):
            if exit == True:
                break
            while (True):
                if doc_index % 10000 == 0:
                    print(str(doc_index) + "/" + str(doc_len))

                try:
                    doc_list = split_news[doc_index]
                except:
                    exit = True
                    break

                try:
                    text = tokenizer.tokenize(doc_list[doc_list_index])
                    if len(text) > seq_len-2:
                        doc_list_index += 1
                        out_of_length += 1
                        #print(text)
                        continue

                    if len(input) + len(text) < seq_len:
                        if len(input) + len(text) == seq_len - 1:
                            input += tokenizer.convert_tokens_to_ids(text) + [sep_index]
                            doc_list_index += 1
                            break  # complete
                        else:
                            input += tokenizer.convert_tokens_to_ids(text)
                            doc_list_index += 1
                            continue
                    else:
                        input = input + tokenizer.convert_tokens_to_ids(text)
                        input = input[:seq_len-1] + [sep_index]
                        break
                except:
                    doc_index += 1
                    doc_list_index = 0
                    if input[:-1] != sep_index:
                        input += [sep_index]
                    if len(input) == seq_len:
                        break
                    continue

            dict = {'indices': input}

            if len(input) != seq_len:
                print(len(input))
            else:
                dataset.append(dict)

            input = [start_index]

        with open("news_index/" + file_name, 'w') as make:
            json.dump(dataset, make)

    print('dataset len ' + str(len(dataset)))
    print('out tokens ' + str(out_of_length))



## main ##

file_name_list = os.listdir('news/')

for i in range(len(file_name_list)):
    news_to_index(file_name_list[i], 512)
    print('complete ' + file_name_list[i])

