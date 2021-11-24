//
// Created by 16663 on 2021/11/24.
//

#ifndef STREAM_PULL_SAMPLE_GETCONFIG_H
#define STREAM_PULL_SAMPLE_GETCONFIG_H
#define COMMENT_CHAR '#'
#include <string>
#include <map>

using namespace std;

bool ReadConfig(const string & filename, map<string, string> & m);

#endif //STREAM_PULL_SAMPLE_GETCONFIG_H
