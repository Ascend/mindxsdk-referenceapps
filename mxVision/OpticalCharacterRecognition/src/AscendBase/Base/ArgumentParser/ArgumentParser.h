/*
* Copyright (c) 2020 Huawei Technologies Co. Ltd, All rights reserved.
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

#ifndef ARGUMENT_PARSER_H
#define ARGUMENT_PARSER_H

#include <string>
#include <map>
#include <vector>
#include "Log/Log.h"

// Argument parser class
class ArgumentParser {
public:
    ArgumentParser();
    ArgumentParser(int argc, const char **argv);
    ~ArgumentParser() {};
    // Add arguments into the map
    void AddArgument(const std::string &argument, const std::string &defaults = "", const std::string &message="");
    // Parse the input arguments
    void ParseArgs(int argc, const char **argv);
    // Get the string argument value
    const std::string &GetStringArgumentValue(const std::string &argumentValue);
    // Get the int argument value
    const int GetIntArgumentValue(const std::string &argumentValue);
    // Get the bool argument value
    const bool GetBoolArgumentValue(const std::string &argumentValue);

private:
    std::map<std::string, std::pair<std::string, std::string>> argumentsMap_;
    // Show the help information, then exit
    void DisplayHelpInformation() const;
    bool IsInteger(std::string &str) const;
    bool IsDecimal(std::string &str) const;
};
#endif