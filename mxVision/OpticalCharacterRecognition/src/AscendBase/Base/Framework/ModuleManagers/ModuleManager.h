#ifndef INC_MODULE_MANEGER_H
#define INC_MODULE_MANEGER_H

#include "acl/acl.h"
#include "Log/Log.h"
#include "ModuleManagers/ModuleBase.h"
#include "ModuleManagers/ModuleFactory.h"

namespace ascendOCR {
    const std::string PIPELINE_DEFAULT = "DefaultPipeline";
    
    struct ModuleDesc {
        std::string moduleName;
        int moduleCount;
    };
    
    struct ModuleConnectDesc {
        std::string moduleSend;
        std::string moduleRecv;
        ModuleConnectType connectType;
    };

    struct ModulesInformation {
        std::vector<std::shared_ptr<ModuleBase>> moduleVec;
        std::vector<std::shared_pre<BlockingQueue<std::shared_ptr<void>>>> inputQueueVec;
    };

    using ModulesInfo = ModulesInformation;

    class ModuleManager {
    public:
        ModuleManager();
        ~ModuleManager();
        APP_ERROR Init(std::string &configPath, std::string &aclConfigPath);
        APP_ERROR DeInit(void);

        APP_ERROR RegisterModules(std::string pipelineName, ModuleDesc *moduleDesc, int moduleTypeCount, int defaultCount);
        APP_ERROR RegisterModuleConnects(std::string pipelineName, ModuleConnectDesc *connectDesc, int moduleConnectCount);
        APP_ERROR RegisterInputVec(std::string pipelineName, std::string moduleName,
            std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> inputQueVec);
        APP_ERROR RegisterOutputModule(std::string pipelineName, std::string moduleSend, std::string moduleRecv,
            ModuleConnectType connectType, std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec);

        APP_ERROR RunPipeline();
    
    private:
        APP_ERROR InitModuleInstance(std::shared_ptr<ModuleBase> moduleInstance, int instanceId, std::string pipelineName,
            std::string moduleName);
        APP_ERROR InitPipelineModule();
        APP_ERROR DeInitPipelineModule();
        static void StopModule(std::shared_ptr<ModuleBase> moduleInstance);

    private:
        int32_t deviceId_ = 0;
        std::map<std::string, std::map<std::string, ModulesInfo>> pipelineMap_ = {};
        ConfigParser configParser_ = {};
        int moduleTypeCount_ = 0;
        int moduleConnectCount_ = 0;
        ModuleConnectDesc *connectDesc_ = nullptr;
    };
}


