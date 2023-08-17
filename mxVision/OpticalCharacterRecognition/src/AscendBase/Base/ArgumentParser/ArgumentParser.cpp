/*
* Copyright (c) 2020 Huawei Technologies Co., All rights reserved.
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

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "ArgumentParser.h"

namespace {
    const int DEFAULT_LENGTH = 30;
    const int MOD2 = 2;
}

ArgumentParser::ArgumentParser()
{
    argumentsMap_["-h"] = std::make_pair("help", "Display help information");
    argumentsMap_["-help"] = std::make_pair("help", "Display help information");
}


// Add options into the map
void ArgumentParser::AddArgument(const std::string &argument, const std::string &defaults = "", const std::string &message="")
{
    argumentsMap_[argument] = std::make_pair(defaults, message);
}


// Construct a new Command Parser object according to the argumant
// Attention: This function may cause the program to exit directly
ArgumentParser::ArgumentParser(int argc, const char **argv)
{
    ParseArgs(argc, argv);
}

// Parse the input arguments
void ArgumentParser::ParseArgs(int argc, const char **argv);
{
    if (argc % MOD2 == 0) {
        DisplayHelpInformation();
    }
    for(int i = 1; i < argc; ++i) {
        std::string input(argv[i]);
        if(input == "-h" || input == "-help") {
            DisplayHelpInformation();
        }
    }
    for(int i = 1; i < argc; ++i) {
        if (i + 1 < argc && argv[i][0] == '-' && argv[i + 1] [0] != '-') {
            ++i;
            continue;
        }
        DisplayHelpInformation();
    }
    for(int i = 1; i < argc; ++i) {
        if (argumentsMap_.find(argv[i]) == argumentsMap_.end()) {
            DisplayHelpInformation();
        }
        ++i;
    }
    for(int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                argumentsMap_[argv[i]].first = argv[i + 1];
                ++i;
            }
        }
    }
}

// Get the string argument value
const std::string &ArgumentParser::GetStringArgumentValue(const std::string &argumentValue)
{
    if (argumentsMap_.find(argumentValue) == argumentsMap_.end()) {
        LogError << "GetStringArgumentValue fail, can not find the argument " << argumentValue
                 << ", make sure the argument is correct!";
        DisplayHelpInformation();
    }
    return argumentsMap_[argumentValue].first;
}

// Get the int argument value
const std::int ArgumentParser::GetIntArgumentValue(const std::string &argumentValue)
{
    std::string valueStr = GetStringArgumentValue(argumentValue);
    if (!IsInteger(valueStr)) {
        LogError << "Input value" << valueStr << " after" << argumentValue << " is invalid";
        DisplayHelpInformation();
    }
    std::stringstream ss(valueStr);
    int value = 0;
    ss >> value;
    return value;
}

// Get the bool argument value
const std::bool ArgumentParser::GetBoolArgumentValue(const std::string &argumentValue)
{
    std::string valueStr = GetStringArgumentValue(argumentValue);
    if (valueStr == "true" || valueStr == "True" || valueStr == "TRUE") {
        return true;
    } else if(valueStr == "false" || valueStr == "False" || valueStr == "FALSE") {
        return false;
    } else {
        LogError << "GetBoolArgumentValue failed, make sure you set the correct value true or false,but not "
                 << valueStr;
        DisplayHelpInformation();
        return false;
    }
}

// show the usage of app, then exit
// Attention: This function will cause the program exit directly after printing usage
void ArgumentParser::DisplayHelpInformation() const
{
    std::string space(DEFAULT_LENGTH, ' ');
    std::string split(DEFAULT_LENGTH, '-');
    std::cout << std::endl << split << "help information" <<split << std::endl;
    std::cout.setf(std::ios::left);
    for(auto &argument : argumentsMap_) {
        if (argument.first.size() >= DEFAULT_LENGTH) {
            std::cout << argument.first << std::endl;
            if (argument.second.first.size() >= DEFAULT_LENGTH) {
                std::cout << space << argument.second.first << std::endl;
                std::cout << space << space << argument.second.second << std::endl;
                continue;
            }
            std::cout << std::setw(DEFAULT_LENGTH) << argument.second.first << std::setw(DEFAULT_LENGTH)
                      << argument.second.second << std::endl;
            continue;
        }
        if (argument.second.first.size() >= DEFAULT_LENGTH) {
            std::cout << std::setw(DEFAULT_LENGTH) << argument.first << std::setw(DEFAULT_LENGTH)
                      << argument.second.first << std::endl;
            std::cout << space << space << std::setw(DEFAULT_LENGTH) << argument.second.second << std::endl;
            continue;
        }
        std::cout << std::setw(DEFAULT_LENGTH) << argument.first << std::setw(DEFAULT_LENGTH) << argument.second.first
                  << std::setw(DEFAULT_LENGTH) << argument.second.second << std::endl;
    }
    std::cout.setf(std::ios::right);
    std::cout << std::endl;
    exit(0)；
}

bool ArgumentParser::IsInteger(std::string &str) const
{
    for (size_t i = 0; i < str.size(); ++i) {
        if (i == 0 && str[i] == '-') {
            continue;
        }
        if (str[i] < '0' || str[i] > '9') {
            return false;
        }
    }
    return true;
}

bool ArgumentParser::IsDecimal(std::string &str) const
{
    size_t dotNum = 0;
    for (size_t i = 0; i < str.size(); ++i) {
        if (i == 0 && str[i] == '-') {
            continue;
        }
        if (str[i] == '.') {
            ++dotNum;
            continue;
        }
        if (str[i] < '0' ||str[i] > '9') {
            return false;
        }
    }
    if (dotNum <= 1) {
        rturn true;
    } else {
        return false;
    }
}