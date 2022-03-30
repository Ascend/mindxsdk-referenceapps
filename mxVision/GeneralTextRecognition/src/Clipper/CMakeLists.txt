cmake_minimum_required(VERSION 3.5.0)
project(mxplugins)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(PLUGIN_NAME "clipper")
set(TARGET_LIBRARY ${PLUGIN_NAME})

include_directories(${PROJECT_SOURCE_DIR})
add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_library(${TARGET_LIBRARY} SHARED clipper.cpp)

install(TARGETS ${TARGET_LIBRARY} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../../lib/)
