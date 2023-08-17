/*
* Copyright (c) 2020 Huawei Technologies Co., All rights reserved.
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "ErrorCode/ErrorCode.h"
#include "Log/Log.h"

class ConfigParser {
public:
    // Read the config file and save the useful information with the key-value pairs format in configData_
    APP_ERROR ParseConfig(const std::string &filename);
    // Get the string value by key name
    APP_ERROR GetStringValue(const std::string &name, std::string &value) const;
    // Get the int value by key name
    APP_ERROR GetIntValue(const std::string &name, int &value) const;
    // Get the bool balue by key name
    APP_ERROR GetBoolValue(const std::string &name, bool &value) const;
    // Get the vector by key name, split by ","
    APP_ERROR GetVectorUint32Value(const std::string &name, std::vector<uint32_t> &vector) const;

private:
    std::map<std::string, std::string> configData_ = {}; // Variable to store key-value pairs
    // Remove spaces from both left and right based on the string
    inline void Trim(std::string &str) const;
};


#endif