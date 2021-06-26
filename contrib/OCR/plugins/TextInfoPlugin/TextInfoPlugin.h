#ifndef MXPLUGINS_TEXTINFOPLUGIN_H
#define MXPLUGINS_TEXTINFOPLUGIN_H

#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/buffer/MxpiBufferManager.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include <string>
#include "MxBase/Log/Log.h"
#include "opencv2/opencv.hpp"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxTools/PluginToolkit/MxpiDataTypeWrapper/MxpiDataTypeDeleter.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include <chrono>

/**
 * This plugin is to stack frames based on detected objects.
*/

namespace MxPlugins {
    class TextInfoPlugin : public MxTools::MxPluginBase {
    public:
        /**
        * @description: Init configs.
        * @param configParamMap: config.
        * @return: APP_ERROR.
        */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) override;

        /**
        * @description: DeInit device.
        * @return: Error code.
        */
        APP_ERROR DeInit() override;

        /**
        * @description: MxpiFairmot plugin process.
        * @param mxpiBuffer: data receive from the previous.
        * @return: Error code.
        */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer) override;

        /**
        * @description: MxpiFairmot plugin define properties.
        * @return: properties.
        */
        static std::vector<std::shared_ptr<void>> DefineProperties();

        /**
        * @api
        * @brief Define the number and data type of input ports.
        * @return MxTools::MxpiPortInfo.
        */
        static MxTools::MxpiPortInfo DefineInputPorts();

        /**
        * @api
        * @brief Define the number and data type of output ports.
        * @return MxTools::MxpiPortInfo.
        */
        static MxTools::MxpiPortInfo DefineOutputPorts();
        std::string ltrim(std::string str);
        std::string rtrim(std::string str);
        std::string trim(std::string str);
        std::vector<std::string> split(const std::string& str, char delimiter);
        std::vector<std::string> whitespace_tokenize(std::string text);

        bool _is_whitespace(char letter);
        bool _is_punctuation(char letter);
        bool _is_control(char letter);

        std::map<std::string, int> read_vocab(const char* filename);
        std::string _clean_text(std::string text);
        vector<std::string> _run_split_on_punc(std::string text);
        vector<std::string> tokenize1(std::string text);
        void add_vocab1(map<std::string, int> vocab);
        vector<std::string> tokenize2(std::string text);

        void add_vocab2(const char* vocab_file);
        void encode(vector<std::string> tokens_A, vector<float>& input_ids, 
                    vector<float>& input_mask, vector<float>& segment_ids, int max_seq_length ,
                    map<string, int> vocab, int maxlen_);

        vector<float> convert_tokens_to_ids(vector<std::string> tokens, map<string, int> vocab, int maxlen_);
        vector<std::string> tokenize3(std::string text);


        /**
         * @api
         * @brief Get the >confidence objects and responding id feature
         * @param key
         * @param buffer
         * @return APP_ERROR
         */
    private:
        std::string dataSource_ = "";
        bool do_lower_case_;
        vector<string> never_split_;
        map<string, int> vocab_;
        string unk_token_;
        const int max_input_chars_per_word_ = 100;
        map<string, int> vocab;
        map<int, string> ids_to_tokens;
        bool do_basic_tokenize_;
        const int maxlen_ = 512;

    };
}

#endif