cmake_minimum_required(VERSION 3.5)

project(OpenSeqSLAM)

if (NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif ()
if (NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif ()
if (NOT DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif ()

find_package(OpenCV REQUIRED)

add_library(
	OpenSeqSLAM SHARED 
	OpenSeqSLAM/OpenSeqSLAM.cpp)

target_link_libraries(
	OpenSeqSLAM
	${OpenCV_LIBS})

add_executable(
	iOpenSeqSLAM
	OpenSeqSLAM/main.cpp)

target_link_libraries(
	iOpenSeqSLAM
	OpenSeqSLAM
	${OpenCV_LIBS})