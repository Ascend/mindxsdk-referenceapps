#ifndef INC_MODULE_FACTORY_H
#define INC_MODULE_FACTORY_H

#include <string>
#include <map>
#include <functional>

#define MODULE_REGIST(class_name)                                                         \
    namespace ascendOCR {                                                                 \
    class class_name##Helper {                                                            \
    public:                                                                               \
        class_name##Helper()                                                              \
        {                                                                                 \
            ModuleFactory::RegisterModule(#class_name, class_name##Helper::CreatObjFunc); \
        }                                                                                 \
        static class_name##Helper class_name##helper;                                     \
        static std::string MT_##class_name = #class_name;                                 \
        }
namespace ascendOCR {
    using Constructor = std::function<void *()>;
    class ModuleFactory
    {
    public:
        static void RegisterModule(std::string className, Constructor constructor)
        {
            Constructors()[className] = constructor;
        }

        static void *MakeModule(const std::string &className)
        {
            auto itr = Constructors().find(className);
            if (itr == Constructors().end()) {
                return nullptr;
            }
            return ((Constructor)itr->second)();
        }
    private:
        inline static std::map<std::string, Constructor> &Constructors()
        {
            static std::map<std::string, Constructor> instance;
            return instance;
        }
    };
}

#endif
