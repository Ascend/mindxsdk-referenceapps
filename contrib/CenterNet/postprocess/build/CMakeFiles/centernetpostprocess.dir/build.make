# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/build

# Include any dependencies generated for this target.
include CMakeFiles/centernetpostprocess.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/centernetpostprocess.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/centernetpostprocess.dir/flags.make

CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.o: CMakeFiles/centernetpostprocess.dir/flags.make
CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.o: ../CenterNetPostProcess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.o -c /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/CenterNetPostProcess.cpp

CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/CenterNetPostProcess.cpp > CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.i

CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/CenterNetPostProcess.cpp -o CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.s

# Object files for target centernetpostprocess
centernetpostprocess_OBJECTS = \
"CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.o"

# External object files for target centernetpostprocess
centernetpostprocess_EXTERNAL_OBJECTS =

libcenternetpostprocess.so: CMakeFiles/centernetpostprocess.dir/CenterNetPostProcess.cpp.o
libcenternetpostprocess.so: CMakeFiles/centernetpostprocess.dir/build.make
libcenternetpostprocess.so: CMakeFiles/centernetpostprocess.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcenternetpostprocess.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/centernetpostprocess.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/centernetpostprocess.dir/build: libcenternetpostprocess.so

.PHONY : CMakeFiles/centernetpostprocess.dir/build

CMakeFiles/centernetpostprocess.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/centernetpostprocess.dir/cmake_clean.cmake
.PHONY : CMakeFiles/centernetpostprocess.dir/clean

CMakeFiles/centernetpostprocess.dir/depend:
	cd /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/build /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/build /home/xiubo2/ImageDetection/mindxsdk-referenceapps/contrib/CenterNet/postprocess/build/CMakeFiles/centernetpostprocess.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/centernetpostprocess.dir/depend

