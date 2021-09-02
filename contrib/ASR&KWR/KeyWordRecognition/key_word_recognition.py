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
                # Key words recognition success
                flag = 1
    # Strictly match text synonyms with lexicon
    if flag == 0:
        flag = match_2(r)
    return flag

# Matching after synonym substitution
def match_2(r):
    flag2 = 0
    # Open the custom synonyms table
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
                    flag2=match_3(l,match)
                    if flag2 == 1:
                        break
    return flag2

def match_3(l,match):
    flag3 = 0
    # Open the custom keyword table
    with open('data/keyword.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    # Open the custom synonyms table
    with open('data/TihuanWords.json', 'r', encoding='utf8')as fp:
        json_data1 = json.load(fp)
    for m in range(len(json_data1)):
        lists = []
        fields = json_data1[m]["TihuanWord"].strip()
        fields = fields.split(" ")
        lists.append(fields)
    # Matches successfully and outputs the keyword
    if match is not None:
        for j in range(len(lists[l])):
            for k in range(len(json_data)):
                keyword = json_data[k]["name"]
                match = re.search((keyword), lists[l][j])
                if match is not None:
                    print("KeyWord：", match.group())
                    # Key words recognition success
                    flag3 = 1
                    break
    return flag3

# TextRank keyword extraction
def getkeywords_textrank(data, topk):
    # Concatenating headings and abstracts
    text = '%s' % (data)
    # Loading custom stop word
    jieba.analyse.set_stop_words("data/stopWord.txt")
    # TextRank keyword extraction, part-of-speech filtering
    keywords = jieba.analyse.textrank(text, topK=topk, allowPOS=('n', 'nz', 'v',
                                                                 'vd', 'vn', 'l', 'a', 'd'))
    return keywords


# Extract synonyms of keywords
def sy_keyword(r):
    k1 = 0
    keywords = getkeywords_textrank(r, 5)
    if len(keywords) == 0:
        k1 = 0
    else:
        # Synonym matching after keyword extraction
        for i in range(len(keywords)):
            key = keywords[i]
            for l in range(5):
                # Perform synonym recognition in turn
                synonyms.nearby(key)
                word = synonyms.nearby(key)[0][l + 1]
                flag2 = match(word)
                if flag2 == 1:
                    k1 = 1
    return k1

# Specific process function
def keyword(r):
    flag1 = match(r)
    k = 0
    # Strict text matching
    if flag1 == 1:
        k = 1
    # Extract keywords for synonym matching
    if flag1 == 0:
        k = sy_keyword(r)
    # Keyword recognition failure
    if k == 0:
        print("Keyword not matched!")


if __name__ == "__main__":
    # Test text
    r = '推荐几部好看的影片给我吧。'
    keyword(r)
