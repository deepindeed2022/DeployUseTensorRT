#include <gtest/gtest.h>
#include <common/logger.h>
int main(int argc, char *argv[]){
#ifdef CONFIG_USE_LOG
    setReportableSeverity(Logger::Severity::kINFO);
#endif
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
