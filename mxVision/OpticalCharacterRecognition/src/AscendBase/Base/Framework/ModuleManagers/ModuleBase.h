#ifndef INC_MODULE_BASE_H
#define INC_MODULE_BASE_H

#include <thread>
#include <vector>
#include <map>
#include <atomic>
#include "ConfigParser/ConfigParser.h"
#include "BlockingQueue/BlockingQueue.h"
namespace ascendOCR{
	enum ModuleConnectType {
		MODULE_CONNECT_ONE = 0,
		MODULE_CONNECT_CHANNEL,
		MODULE_CONNECT_PAIR,
		MODULE_CONNECT_RANDOM
	};
	struct ModuleInitParameters {
		std::string pipelineName = {};
		std::string moduleName = {};
		int instanceId = -1;
		void *userData = nullptr;
	};
	struct ModuleOutputInformation {
		std::string moduleName = "";
		ModuleConnectType connectType = MODULE_CONNECT_RANDOM;
		std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec = {};
		uint32_t outputQueVecSize = 0;
	};

	using ModuleInitParams = ModuleInitParameters;
	using ModuleOutputInfo = ModuleOutputInformation;

	class ModuleBase {
	public:
		ModuleBase() {};
		virtual ~ModuleBase() {};
		virtual APP_ERROR Init(ConfigParser &configParser, ModuleInitParams &initParams) = 0;
		virtual APP_ERROR Deinit(void) = 0;
		APP_ERROR Run(void);
		APP_ERROR Stop(void);
		void SetInputVec(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue);
		void SetOutputInfo(std::string moduleName, ModuleConnectType connectType,
			std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec);
		void SendToNextModule(std::string moduleNext, std::shared_ptr<void> outputData, int channelId = 0);
		const std::string GetModuleName();
		const int GetInstanceId();

	protected:
		void ProcessThread();
		virtual APP_ERROR Process(std::shared_ptr<void> inputData) = 0;
		void CallProcess(const std::shared_ptr<void> &sendData);
		void InitParams(const ModuleInitParams &initParams);

	protected:
		int instanceId_ = -1;
		std::string pipelineName_ = {};
		std::string moduleName_ = {};
		int32_t deviceId_ = -1;
		std::thread processThr_ = {};
		std::atomic_bool isStop_ = {};
		bool withoutInputQueue_ = false;
		std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue_ = nullptr;
		std::map<std::string, ModuleOutputInfo> outputQueMap_ = {};
		int outputQueVecSize_ = 0;
		ModuleConnectType connectType_ = MODULE_CONNECT_RANDOM;
		int sendCount_ = 0;
	};
}
#endif
