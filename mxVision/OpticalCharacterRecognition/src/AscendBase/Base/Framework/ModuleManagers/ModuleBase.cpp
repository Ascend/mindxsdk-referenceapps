#include "ModuleBase.h"
#include <chrono>
#include "Log/Log.h"
#include "BlockingQueue/BlockingQueue.h"
#include "ErrorCode/ErrorCode.h"

namespace ascendOCR {
	const int QUEUE_MAX_SIZE = 32;
	const double TIME_COUNTS = 1000.0;
	void ModuleBase::InitParams(const ModuleInitParams &initParams)
	{
		pipelineName_ = initParams.pipelineName;
		moduleName_ = initParams.moduleName;
		instanceId_ = initParams.instanceId;
		isStop_ = false;
	}

	// create a new thread to run module instance
	APP_ERROR ModuleBase::Run()
	{
		LogDebug << moduleName_ << "[" << instanceId_ << "] Run";
		processThr_ = std::thread(&ModuleBase::ProcessThread, this);
		return APP_ERR_OK;
	}

	void ModuleBase::ProcessThread()
	{
		APP_ERROR ret;
		if (withoutInputQueue_ == true) {
			ret = Process(nullptr);
			if (ret != APP_ERROR_OK) {
				LogError << "Fail to process data for " << moduleName_ << "[" << instanceId_ << "]"
					<< ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
			}
			return;
		}
		if (inputQueue_ == nullptr) {
			LogError << "Invalid input queue of " << moduleName_ << "[" << instanceId_ << "].";
			return;
		}
		LogDebug << "Input queue for " << moduleName_ << "[" << instanceId_ << "], inputQueue=" << inputQueue_;
		while (!isStop_) {
			std::shared_ptr<void> frameInfo = nullptr;
			ret = inputQueue_->Pop(frameInfo);
			if (ret == APP_ERR_QUEUE_STOPED) {
				LogDebug << moduleName_ << "[" << instanceId_ << "] input queue stopped.";
				break;
			} else if (ret != APP_ERR_OK || frameInfo == nullptr) {
				LogError << "Fail to get data from input queue for " << moduleName_ << "[" << instanceId_ << "]"
						 << ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
				continue;
			}
			CallProcess(frameInfo);
		}
		LogInfo << moduleName_ << "[" << instanceId_ << "] process thread End;";
	}

	void ModuleBase::CallProcess(const std::shared_ptr<void> &sendData)
	{
		auto startTime = std::chrono::high_resolution_clock::now();
		APP_ERROR ret = APP_ERR_COMM_FAILURE;
		try {
			ret = Process(sendData);
		} catch (...) {
			LogError << "Error occurred in " << moduleName_ << "[" << instanceId_ << "]";
		}

		auto endTime = std::chrono::high_resolution_clock::now();
		double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
		int queueSize = inputQueue_->GetSize();
		if (queueSize > QUEUE_MAX_SIZE) {
			LogWarn << "[Statistic] [Module] [" << moduleName_ << "] [" << instanceId_ << "] [QueueSize] [" << queueSize <<
				"] [Process] [" << costMs << " ms]";
		}

		if (ret != APP_ERR_OK) {
			LogError << "Fail to process data for " << moduleName_ << "[" << instanceId_ << "]"
					 << ", ret=" << ret << "(" << GetAppErrCodeInfo(ret) << ").";
		}
	}

	void ModuleBase::SetOutputInfo(std::string moduleName, ModuleConnectType connectType,
		std::vector<std::shared_ptr<BlockingQueue<std::shared_ptr<void>>>> outputQueVec)
	{
		if (outputQueVec.size() == 0) {
			LogError << "outputQueVec is Empty!" << moduleName;
			return;
		}
		ModuleOutputInfo outputInfo;
		outputInfo.moduleName = moduleName;
		outputInfo.connectType = connectType;
		outputInfo.outputQueVec = outputQueVec;
		outputInfo.outputQueVecSize = outputQueVec.size();
		outputQueMap_[moduleName] = outputInfo;
	}

	const std::string ModuleBase::GetModuleName()
	{
		return moduleName_;
	}

	const int ModuleBase::GetInstanceId()
	{
		return instanceId_;
	}

	void ModuleBase::SetInputVec(std::shared_ptr<BlockingQueue<std::shared_ptr<void>>> inputQueue)
	{
		inputQueue_ = inputQueue;
	}

	void ModuleBase::SendToNextModule(std::string moduleName, std::shared_ptr<void> outputData, int channelId)
	{
		if (isStop_) {
			LogDebug << moduleName_ << "[" << instanceId_ << "] is stopped, can not send to next module.";
			return;
		}
		if (outputQueMap_.find(moduleName) == outputQueMap_.end()) {
			LogError << "No Next Module " << moduleName;
			return;
		}

		auto itr = outputQueMap_.find(moduleName);
		if (itr == outputQueMap_.end()) {
			LogError << "No Next Module " << moduleName;
			return;
		}
		ModuleOutputInfo outputInfo = itr->second;

		if (outputInfo.connectType == MODULE_CONNECT_ONE) {
			outputInfo.outputQueVec[0]->Push(outputData, true);
		} else if (outputInfo.connectType == MODULE_CONNECT_CHANNEL) {
			uint32_t ch = channelId % outputInfo.outputQueVecSize;
			if (ch >= outputInfo.outputQueVecSize) {
				LogError << "No Next Module!";
				return;
			}
			outputInfo.outputQueVec[ch]->Push(outputData, true);
		} else if (outputInfo.connectType == MODULE_CONNECT_PAIR) {
			outputInfo.outputQueVec[instanceId_]->Push(outputData, true);
		} else if (outputInfo.connectType == MODULE_CONNECT_RANDOM) {
			outputInfo.outputQueVec[sendCount_ % outputInfo.outputQueVecSize]->Push(outputData, true);
		}
		sendCount_++;
	}

	APP_ERROR ModuleBase::Stop {
		isStop_ = true;
		
		if (inputQueue_ != nullptr) {
			inputQueue_->Stop();
		}

		if (processThr_.joinable()) {
			processThr_.join();
		}

		return DeInit();
	}
}

