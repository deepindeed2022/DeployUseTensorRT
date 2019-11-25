#include "common/logger.h"
#include "common/logging.h"

//! gLogger is used to set default reportability level for TRT-API specific logging.
Logger gLogger{Logger::Severity::kWARNING};

// LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
// LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
// LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
// LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
// LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

void setReportableSeverity(Logger::Severity severity) {
    gLogger.setReportableSeverity(severity);
    LOG_VERBOSE(gLogger).setReportableSeverity(severity);
    LOG_INFO(gLogger).setReportableSeverity(severity);
    LOG_WARN(gLogger).setReportableSeverity(severity);
    LOG_ERROR(gLogger).setReportableSeverity(severity);
    LOG_FATAL(gLogger).setReportableSeverity(severity);
}
