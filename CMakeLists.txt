cmake_minimum_required(VERSION 3.20)
project(BetterMotionBlur LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OFX_SDK_ROOT "C:/ProgramData/Blackmagic Design/DaVinci Resolve/Support/Developer/OpenFX" CACHE PATH "Path to OpenFX SDK")

set(OFX_SUPPORT_SOURCES
    "${OFX_SDK_ROOT}/Support/Library/ofxsCore.cpp"
    "${OFX_SDK_ROOT}/Support/Library/ofxsImageEffect.cpp"
    "${OFX_SDK_ROOT}/Support/Library/ofxsInteract.cpp"
    "${OFX_SDK_ROOT}/Support/Library/ofxsLog.cpp"
    "${OFX_SDK_ROOT}/Support/Library/ofxsMultiThread.cpp"
    "${OFX_SDK_ROOT}/Support/Library/ofxsParams.cpp"
    "${OFX_SDK_ROOT}/Support/Library/ofxsProperty.cpp"
    "${OFX_SDK_ROOT}/Support/Library/ofxsPropertyValidation.cpp"
)

set(ENGINE_SOURCES
    src/FlowCache.cpp
    src/NeuralFlowEngine.cpp
    src/CudaEngine.cpp
    src/CpuFallbackProcessor.cpp
)

add_library(BetterMotionBlur SHARED
    src/BetterMotionBlurPlugin.cpp
    ${ENGINE_SOURCES}
    ${OFX_SUPPORT_SOURCES}
)

set_target_properties(BetterMotionBlur PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
)

target_include_directories(BetterMotionBlur PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${OFX_SDK_ROOT}/OpenFX-1.4/include"
    "${OFX_SDK_ROOT}/Support/include"
)

target_compile_definitions(BetterMotionBlur PRIVATE
    WIN32
    WIN64
    _WINDOWS
    _USRDLL
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
)

if (MSVC)
    target_compile_options(BetterMotionBlur PRIVATE /W3 /MP)
endif()

add_custom_command(TARGET BetterMotionBlur POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:BetterMotionBlur>/BetterMotionBlur.ofx.bundle/Contents/Win64"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:BetterMotionBlur>/BetterMotionBlur.ofx.bundle/Contents/Resources"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:BetterMotionBlur>" "$<TARGET_FILE_DIR:BetterMotionBlur>/BetterMotionBlur.ofx.bundle/Contents/Win64/BetterMotionBlur.ofx"
    COMMENT "Packaging BetterMotionBlur.ofx.bundle"
)

add_executable(BetterMotionBlurTests
    tests/TestRunner.cpp
    ${ENGINE_SOURCES}
)

add_dependencies(BetterMotionBlurTests BetterMotionBlur)

target_include_directories(BetterMotionBlurTests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${OFX_SDK_ROOT}/OpenFX-1.4/include"
)

target_compile_definitions(BetterMotionBlurTests PRIVATE
    WIN32
    WIN64
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
)

if (MSVC)
    target_compile_options(BetterMotionBlurTests PRIVATE /W3 /MP)
endif()
