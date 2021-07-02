# 语音转文本后关键词识别

## 第三方依赖库
> jieba  
> synonyms  
> numpy  

请使用pip3 install * 安装以上依赖
其中jieba提取关键词时需要词库，正常情况会自动获取，也可手动下载后放置于对应目录[github链接v3.15.0](https://github.com/chatopera/Synonyms/releases/download/3.15.0/words.vector.gz) ，或者使用自定义词库。词库大小影响匹配速度。
## 目录结构
```
.
|-------- data
|--------   |---- keyword.json        //自定义关键词语料库
|--------   |---- stopWord.txt        //TextRank关键词识别停词表
|--------   |---- TihuanWords.json    //自定义近义词表
|-------- key_word_recognition.py     //关键词识别
```

## 具体流程介绍如下：
1. 输入经过语音识别转化为的文本，根据keyword.json语料库给出的关键词与文本进行严格匹配，
     匹配成功输出语料库匹配关键词；匹配失败执行下一步。<br/>
2. 对文本进行Textrank关键词提取，提取Top3关键词；根据自定义近义词表TihuanWords.json查找关键词近义词后和语料库进行匹配。
     匹配成功输出语料库匹配关键词；匹配失败执行下一步。<br/>
3. 对Top3关键词，根据Synonyms近义词表查找关键词近义词后和语料库进行匹配。
     匹配成功输出语料库匹配关键词；匹配失败则未识别到关键词。<br/>
