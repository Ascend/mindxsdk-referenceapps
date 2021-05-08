set(PLUGIN_NAME "mxpi_framealign")
set(TARGET_LIBRARY ${PLUGIN_NAME})

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
include_directories(${PROJECT_DIR}/Proto)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")

add_library(${TARGET_LIBRARY} SHARED MxpiFrameAlign.cpp)

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)
target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxbase mxpidatatype mxpiallobjectsstructuringdatatype mindxsdk_protobuf)
target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack)

install(TARGETS ${TARGET_LIBRARY} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/dist/lib)
