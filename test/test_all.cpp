#include <gtest/gtest.h>
#include <common/logger.h>
#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
int main(int argc, char *argv[]){
#ifdef CONFIG_USE_LOG
    setReportableSeverity(Logger::Severity::kINFO);
#endif
    cudaSetDevice(0);
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
