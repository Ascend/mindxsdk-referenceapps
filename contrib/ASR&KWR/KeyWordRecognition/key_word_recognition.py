# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import jieba.analyse
import synonyms
import json


# Strictly match the text with the lexicon
def match(r):
    flag = 0
    with open('data/keyword.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        for i in range(len(json_data)):
            keyword = json_data[i]["name"]
            match = re.search((keyword), r)
            if match is not None:
                print("KeyWord：", match.group())
                flag = 1

    # Strictly match text synonyms with lexicon
    if(flag == 0):
        with open('data/TihuanWords.json', 'r', encoding='utf8')as fp:
            json_data1 = json.load(fp)
            for m in range(len(json_data1)):
                lists = []
                fields = json_data1[m]["TihuanWord"].strip()
                fields = fields.split(" ")
                lists.append(fields)
            for l in range(len(lists)):
                for i in range(len(lists[l])):
                    match = re.search((lists[l][i]), r)
                    if match is not None:
                        for j in range(len(lists[l])):
                            for k in range(len(json_data)):
                                keyword = json_data[k]["name"]
                                match = re.search((keyword), lists[l][j])
                                if match is not None:
                                    print("KeyWord：", match.group())
                                    flag = 1
    return flag


# TextRank keyword extraction
def getKeywords_textrank(data, topK):
    # Concatenating headings and abstracts
    text = '%s' % (data)
    # Loading custom stop word
    jieba.analyse.set_stop_words("data/stopWord.txt")
    # TextRank keyword extraction, part-of-speech filtering
    keywords = jieba.analyse.textrank(text, topK=topK, allowPOS=('n', 'nz', 'v',
                                                                 'vd', 'vn', 'l', 'a', 'd'))
    return keywords


# Extract synonyms of keywords
def Sy_KeyWord(r):
    keywords = getKeywords_textrank(r, 5)
    if (len(keywords) == 0):
        k = 0
    else:
        for i in range(len(keywords)):
            key = keywords[i]
            for l in range(5):
                synonyms.nearby(key)
                word = synonyms.nearby(key)[0][l + 1]
                flag2 = match(word)
                if (flag2 == 1):
                    k = 1
    return k

# Specific process function


def keyword(r):
    flag1 = match(r)
    k = 0
    if (flag1 == 1):
        k = 1
    if (flag1 == 0):
        k = Sy_KeyWord(r)
    if (k == 0):
        print("Keyword not matched!")


if __name__ == "__main__":
    # r='他去网吧。'
    r = '他经常上学打架。'
    keyword(r)
