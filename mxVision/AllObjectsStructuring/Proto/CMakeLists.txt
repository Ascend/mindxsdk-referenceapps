cmake_minimum_required(VERSION 3.10.0)
project(ProtoFile)

set(TARGET_LIBRARY mxpiallobjectsstructuringdatatype)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)

if(EXISTS ${MX_SDK_HOME})
    set(PROTOBUF_FOUND TRUE)
    set(PROTOBUF_PROTOC_EXECUTABLE ${MX_SDK_HOME}/opensource/bin/protoc)
    set(PROTOBUF_INCLUDE_DIRS ${MX_SDK_HOME}/opensource/include)
    set(PROTOBUF_PROTOC_LIBRARIES ${MX_SDK_HOME}/opensource/lib/libprotoc.so -lpthread)
    set(PROTOBUF_LITE_LIBRARIES ${MX_SDK_HOME}/opensource/lib/libprotoc-lite.so -lpthread)
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    link_directories(${MX_SDK_HOME}/opensource/lib)
    find_library(PROTOBUF_LIB NAMES libprotobuf.so.22 PATHS ${MX_SDK_HOME}/opensource/lib)

endif()
if (PROTOBUF_FOUND)
    message(STATUS "protobuf library found")

else ()
    message(FATAL_ERROR "protobuf library is needed but cant be found")
endif ()

LIST(APPEND PROTO_FLAGS -I${PROJECT_SOURCE_DIR})

EXECUTE_PROCESS(
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --cpp_out=${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/MxpiAllObjectsStructuringDataType.proto
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --python_out=${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/MxpiAllObjectsStructuringDataType.proto
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cc)
add_library(${TARGET_LIBRARY} SHARED ${SOURCE_FILES})
target_link_libraries(${TARGET_LIBRARY} mindxsdk_protobuf)
target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack -s)

if(ENABLE_TEST)
    target_link_libraries(${TARGET_LIBRARY} gcov)
endif()

install(TARGETS ${TARGET_LIBRARY} LIBRARY DESTINATION ${PROJECT_DIR}/dist/lib)
install(FILES ${PROJECT_SOURCE_DIR}/MxpiAllObjectsStructuringDataType.pb.h DESTINATION ${PROJECT_DIR}/dist/include)
install(FILES ${PROJECT_SOURCE_DIR}/MxpiAllObjectsStructuringDataType_pb2.py DESTINATION ${PROJECT_DIR}/dist/python)
