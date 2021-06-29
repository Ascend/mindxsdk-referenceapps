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
#include "MxTools/PluginToolkit/MxpiDataTypeWrapper/MxpiDataTypeConverter.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include <chrono>

/**
 * This plugin is to stack frames based on detected objects.
*/

namespace MxPlugins {
    class TextSimilarityPlugin : public MxTools::MxPluginBase {
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

        float scalar_product(std::vector<float> a, std::vector<float> b);

        float linalg(std::vector<float> a);
        float similarity(std::vector<float> a, std::vector<float> b);
        /**
         * @api
         * @brief Get the >confidence objects and responding id feature
         * @param key
         * @param buffer
         * @return APP_ERROR
         */
    private:
        std::string dataSource_ = "";

    };
}

#endif