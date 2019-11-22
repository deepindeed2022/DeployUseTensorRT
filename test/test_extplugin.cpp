#include <CaffeModel.h>
#include <gtest/gtest.h>
#include <chrono>
#include <PluginManager.h>
#include <NvCaffeParser.h>
using namespace std::chrono;
using namespace nvcaffeparser1;
#include "extplugin/regPlugin.h"

TEST(Plugin, extPlugin) {
    ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactoryExt(PluginManager::getInstance().createPlugin("interp"));
    PluginManager::getInstance().dumpPlugins();
    parser->destroy();
}