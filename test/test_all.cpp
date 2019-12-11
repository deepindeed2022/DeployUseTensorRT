#include <gtest/gtest.h>
#include <common/logger.h>
#include <cuda_runtime_api.h>
#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
int main(int argc, char *argv[]){
#ifdef CONFIG_USE_LOG
    setReportableSeverity(Logger::Severity::kVERBOSE);
#endif
    cudaSetDevice(0);
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
