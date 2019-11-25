#include <map>
#include <utility>
#include "NvCaffeParser.h"
#include <common/logger.h>
class PluginManager {
public:
	using FCTOR_PLUGIN_CREATOR = std::function<nvcaffeparser1::IPluginFactoryExt*()>;
	static PluginManager& getInstance() {
		static PluginManager manger;
		return manger;
	}
	void registerPlugin(const char* type, FCTOR_PLUGIN_CREATOR ctor) {
		_plugin_map[std::string(type)] = ctor;
	}
	nvcaffeparser1::IPluginFactoryExt* createPlugin(const char* type) {
		if(_plugin_map.find(std::string(type)) != _plugin_map.end()) {
			return _plugin_map[std::string(type)]();
		} else {
			LOG_ERROR(gLogger) << "Plugin " << type << " not register!";
			return NULL;
		}
	}
	void dumpPlugins() {
		std::cout << "Dump Plugins as following:\n";
		for(auto iter = _plugin_map.begin(); iter != _plugin_map.end(); ++iter) {
			std::cout << "[ext] " << iter->first << std::endl;
		}
	}
private:
	std::map<std::string,  FCTOR_PLUGIN_CREATOR> _plugin_map;
	PluginManager() = default;
};

#define REGISTER_PLUGIN(type, pluginname) 	\
	namespace {								\
	static nvcaffeparser1::IPluginFactoryExt* pluginname##_pfnPlugin() { return new pluginname; }\
	class pluginname##ext_pFCtor{			\
	public:									\
		pluginname##ext_pFCtor(){			\
			PluginManager::getInstance().registerPlugin(type, pluginname##_pfnPlugin);	\
		} 									\
	}; 										\
	static pluginname##ext_pFCtor  pluginname##ext_pFCtor##obj;	\
	}
