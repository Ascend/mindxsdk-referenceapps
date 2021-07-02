#include "TextInfoPlugin.h"
#include <iostream>
#include "MxBase/Log/Log.h"
#include "MxTools/PluginToolkit/MxpiDataTypeWrapper/MxpiDataTypeConverter.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include <mutex>
#include <thread>
#include <map>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <codecvt>
#include <algorithm>
#include <cstdint>
#include <istream>
#include <sstream>
#include <boost/algorithm/string/join.hpp>
using namespace MxBase;
using namespace MxTools;
using namespace MxPlugins;
using namespace std;

APP_ERROR TextInfoPlugin::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "Begin to initialize TextInfoPlugin(" << pluginName_ << ").";

    dataSource_ = *std::static_pointer_cast<std::string>(configParamMap["dataSource"]);
    do_lower_case_ = false;
    never_split_ = { "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]" };
    map<string, int> vocab;
    vocab_ = vocab;
    unk_token_="[UNK]";
    LogInfo << "End to initialize MxpiFairmot(" << pluginName_ << ").";
    return APP_ERR_OK;
}

vector<float> TextInfoPlugin::convert_tokens_to_ids(vector<string> tokens, map<string, int> vocab, int maxlen_)
{
    vector<float> ids;
    vector<std::string>::iterator ptr;
    for (ptr = tokens.begin(); ptr < tokens.end(); ++ptr)
    {
        ids.push_back(int(vocab[*ptr]));
    }
    if (ids.size() > maxlen_){
        cout << "Token indices sequence length is longer than the specified maximum";
    }
    return ids;
}

APP_ERROR TextInfoPlugin::DeInit()
{
    LogInfo << "Begin to deinitialize MxpiFairmot(" << pluginName_ << ").";
    LogInfo << "End to deinitialize MxpiFairmot(" << pluginName_ << ").";
    return APP_ERR_OK;
}

void Covert(const std::shared_ptr<MxTools::MxpiTextsInfoList> &textsInfoList,
            std::vector<MxBase::TextsInfo> &textsInfoVec)
{
    for (uint32_t i = 0; i < textsInfoList->textsinfovec_size(); i++) {
        auto textsInfo = textsInfoList->textsinfovec(i);
        MxBase::TextsInfo text;
        for (uint32_t j = 0; j < textsInfo.text_size(); j++) {
            auto textInfo = textsInfo.text(j);
            if (textInfo == ""){
                continue;
            }
            LogInfo << "text:" << textInfo;
            text.text.push_back(textInfo);
        }
        textsInfoVec.push_back(text);
    }
}

void TextInfoPlugin::encode(std::vector<std::string> tokens_A, std::vector<float>& input_ids,
                            std::vector<float>& input_mask, std::vector<float>& segment_ids, int max_seq_length,
                            std::map<std::string, int>& vocab, int maxlen_){
    tokens_A.insert(tokens_A.begin(), "[CLS]");
    tokens_A.push_back("[SEP]");
    for (int i = 0; i < tokens_A.size(); i++)
    {
        segment_ids.push_back(0.0);
        input_mask.push_back(1.0);
    }

    input_ids = convert_tokens_to_ids(tokens_A, vocab, maxlen_);
    while (input_ids.size() < max_seq_length)
    {
        input_ids.push_back(0.0);
        input_mask.push_back(0.0);
        segment_ids.push_back(0.0);
    }

}

APP_ERROR TextInfoPlugin::Process(std::vector<MxpiBuffer *> &mxpiBuffer)
{
    /*
     * get the MxpiVisionList and MxpiTrackletList
     * */
    LogInfo<< "ProcessProcessProcess";
    LogInfo<< "Begin to process MxpiMotSimpleSort(" << elementName_ << ").";
    // Get MxpiVisionList and MxpiTrackletList from mxpibuffer
    MxpiBuffer *inputMxpiBuffer = mxpiBuffer[0];   // deviceID[0]
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer);

    // Get the metadata from buffer
    std::shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(dataSource_);
    auto textInfoList = std::static_pointer_cast<MxTools::MxpiTextsInfoList>(metadata);
    std::vector<MxBase::TextsInfo> texts;
    int length=0;
    Covert(textInfoList, texts);
    std::string input;
    for(int i=0; i<texts.size(); i++){
        for(int j=0; j<texts[i].text.size();j++){
            std::string temp=static_cast<string>(texts[i].text[j]);
            input = input + temp;
            input += " ";
        }
    }
    transform(input.begin(), input.end(), input.begin(), ::tolower);
    const char*  vocab_file = "./vocab.txt";
    add_vocab2(vocab_file);
    vector<std::string> tokens_a = tokenize3(input);
    map<string, int> vocab = read_vocab(vocab_file);

    vector<float> input_ids, input_mask, segment_ids;
    vector<int> res_input_ids, res_input_mask, res_input_segment;
    vector<int> res_input_length;

    vector<MxBase::TextsInfo> res_input_text;
    MxBase::TextsInfo textInfo = {};
    for (int i = 0; i < tokens_a.size(); ++i) {
        textInfo.text.push_back(tokens_a[i]);
    }
    res_input_text.push_back(textInfo);

    encode(tokens_a, input_ids, input_mask, segment_ids, 512, vocab, 512);

    for (int i = 0; i < input_mask.size(); ++i) {
        if((int)input_mask[i]==0){
            break;
        }
        length++;
    }

    res_input_length.push_back(length);

    for (int i = 0; i < input_ids.size(); ++i) {
        res_input_ids.push_back((int) input_ids[i]);
        res_input_mask.push_back((int) input_mask[i]);
        res_input_segment.push_back((int) segment_ids[i]);
    }

    const std::string key1 = elementName_+"_id";
    const std::string key2 = elementName_+"_mask";
    const std::string key3 = elementName_+"_segment";
    const std::string key4 = elementName_+"_length";

    const std::string key5 = elementName_+"_text";

    auto dataSize_id      = res_input_ids.size() * sizeof(int);
    auto dataSize_mask    = res_input_mask.size() * sizeof(int);
    auto dataSize_segment = res_input_segment.size() * sizeof(int);
    auto dataSize_length = res_input_length.size() * sizeof(int);

    auto dataPtr_id      = &res_input_ids[0];
    auto dataPtr_mask    = &res_input_mask[0];
    auto dataPtr_segment = &res_input_segment[0];
    auto dataPtr_length = &res_input_length[0];

    MxBase::MemoryData memorySrc_id(dataPtr_id, dataSize_id, MxBase::MemoryData::MEMORY_HOST_NEW);
    MxBase::MemoryData memorySrc_mask(dataPtr_mask, dataSize_mask, MxBase::MemoryData::MEMORY_HOST_NEW);
    MxBase::MemoryData memorySrc_segment(dataPtr_segment, dataSize_segment, MxBase::MemoryData::MEMORY_HOST_NEW);
    MxBase::MemoryData memorySrc_length(dataPtr_length, dataSize_length, MxBase::MemoryData::MEMORY_HOST_NEW);

    MxBase::MemoryData memoryDst_id(dataSize_id, MxBase::MemoryData::MEMORY_HOST_NEW);
    MxBase::MemoryData memoryDst_mask(dataSize_mask, MxBase::MemoryData::MEMORY_HOST_NEW);
    MxBase::MemoryData memoryDst_segment(dataSize_segment, MxBase::MemoryData::MEMORY_HOST_NEW);
    MxBase::MemoryData memoryDst_length(dataSize_length, MxBase::MemoryData::MEMORY_HOST_NEW);

    APP_ERROR res_id = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst_id, memorySrc_id);
    APP_ERROR res_mask = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst_mask, memorySrc_mask);
    APP_ERROR res_segment = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst_segment, memorySrc_segment);
    APP_ERROR res_length = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst_length, memorySrc_length);

    auto tensorPackageList_id = std::shared_ptr<MxTools::MxpiTensorPackageList>(new
            MxTools::MxpiTensorPackageList,
            MxTools::g_deleteFuncMxpiTensorPackageList);
    auto tensorPackageList_mask = std::shared_ptr<MxTools::MxpiTensorPackageList>(new
            MxTools::MxpiTensorPackageList,
            MxTools::g_deleteFuncMxpiTensorPackageList);
    auto tensorPackageList_segment = std::shared_ptr<MxTools::MxpiTensorPackageList>(new
            MxTools::MxpiTensorPackageList,
            MxTools::g_deleteFuncMxpiTensorPackageList);
    auto tensorPackageList_length = std::shared_ptr<MxTools::MxpiTensorPackageList>(new
            MxTools::MxpiTensorPackageList,
            MxTools::g_deleteFuncMxpiTensorPackageList);

    auto tensorPackage_id      = tensorPackageList_id->add_tensorpackagevec();
    auto tensorPackage_mask    = tensorPackageList_mask->add_tensorpackagevec();
    auto tensorPackage_segment = tensorPackageList_segment->add_tensorpackagevec();
    auto tensorPackage_length = tensorPackageList_length->add_tensorpackagevec();

    auto tensorVec_id   = tensorPackage_id->add_tensorvec();
    auto tensorVec_mask = tensorPackage_mask->add_tensorvec();
    auto tensorVec_segment = tensorPackage_segment->add_tensorvec();
    auto tensorVec_length = tensorPackage_length->add_tensorvec();

    tensorVec_id->set_tensordataptr((uint64_t)memoryDst_id.ptrData);
    tensorVec_mask->set_tensordataptr((uint64_t)memoryDst_mask.ptrData);
    tensorVec_segment->set_tensordataptr((uint64_t)memoryDst_segment.ptrData);
    tensorVec_length->set_tensordataptr((uint64_t)memoryDst_length.ptrData);

    tensorVec_id->set_tensordatasize(res_input_ids.size());
    tensorVec_mask->set_tensordatasize(res_input_mask.size());
    tensorVec_segment->set_tensordatasize(res_input_segment.size());
    tensorVec_length->set_tensordatasize(res_input_length.size());

    tensorVec_id->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);
    tensorVec_mask->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);
    tensorVec_segment->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);
    tensorVec_length->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);

    tensorVec_id->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);
    tensorVec_mask->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);
    tensorVec_segment->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);
    tensorVec_length->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);

    tensorVec_id->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);
    tensorVec_mask->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);
    tensorVec_segment->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);
    tensorVec_length->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);

    tensorVec_id->set_deviceid(0);
    tensorVec_mask->set_deviceid(0);
    tensorVec_segment->set_deviceid(0);
    tensorVec_length->set_deviceid(0);

    tensorVec_id->add_tensorshape(1);
    tensorVec_mask->add_tensorshape(1);
    tensorVec_segment->add_tensorshape(1);
    tensorVec_length->add_tensorshape(1);

    tensorVec_id->add_tensorshape(res_input_ids.size());
    tensorVec_mask->add_tensorshape(res_input_mask.size());
    tensorVec_segment->add_tensorshape(res_input_segment.size());
    tensorVec_length->add_tensorshape(res_input_length.size());

    APP_ERROR error_id = mxpiMetadataManager.AddProtoMetadata(key1,
              static_pointer_cast<google::protobuf::Message>(tensorPackageList_id));
    APP_ERROR error_mask = mxpiMetadataManager.AddProtoMetadata(key2,
              static_pointer_cast<google::protobuf::Message>(tensorPackageList_mask));
    APP_ERROR error_segment = mxpiMetadataManager.AddProtoMetadata(key3,
              static_pointer_cast<google::protobuf::Message>(tensorPackageList_segment));
    APP_ERROR error_length = mxpiMetadataManager.AddProtoMetadata(key4,
              static_pointer_cast<google::protobuf::Message>(tensorPackageList_length));

    shared_ptr<MxpiTextsInfoList> mxpiTextsInfoList = ConstructProtobuf(res_input_text, key5);

    error_length = mxpiMetadataManager.AddProtoMetadata(key5,mxpiTextsInfoList);

    // Send the data to downstream plugin
    for (size_t i = 0; i < 5; i++) {
        gst_buffer_ref((GstBuffer*)inputMxpiBuffer->buffer);
        auto tmpBuffer = new MxpiBuffer {inputMxpiBuffer->buffer};
        SendData(i, *tmpBuffer);
    }
    MxpiBufferManager::DestroyBuffer(inputMxpiBuffer);
    LogInfo << "End to process TextInfoPlugin(" << elementName_ << ").";
    return APP_ERR_OK;
}

std::string TextInfoPlugin::ltrim(std::string str)
{
    return regex_replace(str, regex("^\\s+"), std::string(""));
}

std::string TextInfoPlugin::rtrim(std::string str)
{
    return regex_replace(str, regex("\\s+$"), std::string(""));
}

std::string TextInfoPlugin::trim(std::string& str)
{
    return ltrim(rtrim(str));
}

vector<std::string> TextInfoPlugin::split(const std::string& str, char delimiter)
{
    vector<std::string> internal;
    std::stringstream ss(str); // Turn the std::string into a stream.
    std::string tok;

    while (getline(ss, tok, delimiter))
    {
        internal.push_back(tok);
    }
    return internal;
}

vector<std::string> TextInfoPlugin::whitespace_tokenize(std::string text)
{
    vector<std::string> result;
    char delimeter = ' ';
    text = trim(text);
    if (text == "")
    {
        return result;
    }
    result = split(text, delimeter);
    return result;
}

bool TextInfoPlugin::_is_whitespace(char letter)
{
    if (letter == ' ' or letter == '\t' or letter == '\n' or letter == '\r'){
        return true;
    }
    return false;
}

bool TextInfoPlugin::_is_punctuation(char letter)
{
    int cp = int(letter);
    if ((cp >= '!' and cp <= '/') or (cp >= ':' and cp <= '@') or
        (cp >= '[' and cp <= '`') or (cp >= '{' and cp <= '~')){
        return true;
    }
    return false;
}

bool TextInfoPlugin::_is_control(char letter)
{
    if (letter == '\t' or letter == '\n' or letter == '\r'){
        return false;
    }
    return false;
}

map<std::string, int> TextInfoPlugin::read_vocab(const char* filename)
{
    map<std::string, int> vocab;
    int index = 0;
    unsigned int line_count = 1;
    ifstream fs8(filename);
    if (!fs8.is_open())
    {
        cout << "Could not open " << filename << endl;
        return vocab;
    }
    std::string line;
    // Read all the lines in the file
    while (getline(fs8, line))
    {
        vocab.insert(pair<std::string, int>(std::string(line.begin(), line.end()), index));
        index++;
        line_count++;
    }
    return vocab;
}

std::string TextInfoPlugin::_clean_text(std::string text)
{
    std::string output;
    int len = 0;
    char* char_array = new char[text.length() + 1];
    strcpy(char_array, text.c_str());
    while (char_array[len] != '\0')
    {
        int cp = int(char_array[len]);
        if (cp == 0 or cp == 0xfffd or _is_control(char_array[len])){
            continue;
        }
        if (_is_whitespace(char_array[len])){
            output = output + " ";
        }
        else{
            output = output + char_array[len];
        }
        ++len;
    }
    return output;
}

vector<std::string> TextInfoPlugin::_run_split_on_punc(std::string text)
{
    if (find(never_split_.begin(), never_split_.end(), text) != never_split_.end())
    {
        vector<std::string> temp = { text };
        return temp;
    }
    int len_char_array = text.length();
    char* char_array = new char[text.length() + 1];
    strcpy(char_array, text.c_str());
    int i = 0;
    bool start_new_word = true;
    vector<vector<char>> output;
    while (i < len_char_array)
    {
        char letter = char_array[i];
        if (_is_punctuation(letter))
        {
            vector<char> temp = { letter };
            output.push_back(temp);
            start_new_word = true;
        }
        else
        {
            if (start_new_word)
            {
                vector<char> temp_2;
                output.push_back(temp_2);
            }
            start_new_word = false;
            output.back().push_back(letter);
        }
        i += 1;
    }
    vector<std::string> final_output;
    vector<vector<char>>::iterator ptr;
    for (ptr = output.begin(); ptr < output.end(); ++ptr)//
    {
        vector<char> out = *ptr;
        std::string word = "";
        vector<char>::iterator itr;
        for (itr = out.begin(); itr < out.end(); ++itr)//
        {
            word = word + *itr;
        }
        final_output.push_back(word);
    }
    delete char_array;
    return final_output;
}

std::vector<std::string> TextInfoPlugin::tokenize1(std::string& text)
{
    vector<std::string> orig_tokens = whitespace_tokenize(text);
    vector<std::string> split_tokens;
    vector<std::string>::iterator itr;
    for (itr = orig_tokens.begin(); itr < orig_tokens.end(); ++itr)//
    {
        std::string temp = *itr;
        if (do_lower_case_ and not bool(find(never_split_.begin(), never_split_.end(), *itr) != never_split_.end()))
        {
            transform(temp.begin(), temp.end(), temp.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
        }
        vector<std::string> split = _run_split_on_punc(temp);
        split_tokens.insert(split_tokens.end(), split.begin(), split.end());
    }
    std::string temp_text;
    vector<std::string>::iterator ptr;
    for (ptr = split_tokens.begin(); ptr < split_tokens.end(); ++ptr)//
    {
        temp_text = temp_text + " " + *ptr;
    }
    return whitespace_tokenize(temp_text);
}

void TextInfoPlugin::add_vocab1(std::map<std::string, int>& vocab)//
{
    vocab_ = vocab;
    unk_token_ = "[UNK]";
}

std::vector<std::string> TextInfoPlugin::tokenize2(std::string& text)
{
    vector<std::string> output_tokens;
    vector<std::string> whitespace_tokens = whitespace_tokenize(text);
    vector<std::string>::iterator ptr;
    for (ptr = whitespace_tokens.begin(); ptr < whitespace_tokens.end(); ++ptr)//
    {

        std::string token = *ptr;
        int len_char_array = token.length();

        char* char_array = new char[token.length() + 1];
        strcpy(char_array, token.c_str());
        if (len_char_array > max_input_chars_per_word_)
        {
            output_tokens.push_back(unk_token_);
            continue;
        }

        bool is_bad = false;
        int start = 0;
        vector<std::string> sub_tokens;
        while (start < len_char_array)
        {
            int end = len_char_array;
            std::string cur_substr = "";
            while (start < end)
            {
                std::string substr;
                for (int c = start; c < end; c++){
                    substr = substr + char_array[c];
                }
                if (start > 0){
                    substr = "##" + substr;
                }
                if (vocab_.count(substr) == 1)
                {
                    cur_substr = substr;
                    break;
                }
                end = end - 1;
            }
            if (cur_substr == "")
            {
                is_bad = true;
                break;
            }
            sub_tokens.push_back(cur_substr);
            start = end;
        }
        if (is_bad){
            output_tokens.push_back(unk_token_);
        }
        else
        {
            output_tokens.insert(output_tokens.end(), sub_tokens.begin(), sub_tokens.end());
        }
    }
    return output_tokens;
}

void TextInfoPlugin::add_vocab2(const char* vocab_file)
{
    vocab = read_vocab(vocab_file);
    for (map<std::string, int>::iterator i = vocab.begin(); i != vocab.end(); ++i){
        ids_to_tokens[i->second] = i->first;
    }
    do_basic_tokenize_ = true;
    do_lower_case_ = true;
    add_vocab1(vocab);
    //maxlen_ = 128;
}

std::vector<std::string> TextInfoPlugin::tokenize3(std::string& text)
{
    vector<std::string> split_tokens;
    if (!do_basic_tokenize_)
    {

        vector<std::string> temp_tokens = tokenize1(text);
        vector<std::string>::iterator ptr;
        for (ptr = temp_tokens.begin(); ptr < temp_tokens.end(); ++ptr)//
        {
            vector<std::string> subtokens = tokenize2(*ptr);
            split_tokens.insert(split_tokens.end(), subtokens.begin(), subtokens.end());
        }
    }
    else
    {
        split_tokens = tokenize2(text);
    }

    return split_tokens;
}

std::vector<std::shared_ptr<void>> TextInfoPlugin::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    auto datasource = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
            STRING,
            "dataSource",
            "dataSource",
            "the name of cropped image source",
            "default", "NULL", "NULL"
    });

    properties.push_back(datasource);
    return properties;
}

MxpiPortInfo TextInfoPlugin::DefineInputPorts()
{
    MxpiPortInfo inputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);
    return inputPortInfo;
}

MxpiPortInfo TextInfoPlugin::DefineOutputPorts()
{
    MxpiPortInfo outputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"},{"ANY"},{"ANY"},{"ANY"},{"ANY"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);
    return outputPortInfo;
}

namespace {
    MX_PLUGIN_GENERATE(TextInfoPlugin)
}