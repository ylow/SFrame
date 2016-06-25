project(unity)
set(EXTENSIONS_LIST)
macro(make_extension NAME)
  set(multi_value_args SOURCES REQUIRES)
  CMAKE_PARSE_ARGUMENTS(make_extension "" "" "${multi_value_args}" ${ARGN})
  list(APPEND EXTENSIONS_LIST "${NAME}")
  message(STATUS "Registering Extension: " ${NAME})
  make_library(${NAME} SOURCES ${make_extension_SOURCES} REQUIRES unity boost ${make_extension_REQUIRES} SHARED)
endmacro()
#----------------------------
make_extension(additional_sframe_utilities SOURCES additional_sframe_utilities.cpp)
make_extension(grouped_sframe SOURCES grouped_sframe.cpp)
make_extension(internal_demo SOURCES internal_demo.cpp)
make_extension(json SOURCES
  json/decoder.cpp
  json/encoder.cpp
  json/json.cpp
)
make_extension(random_sframe_generation SOURCES random_sframe_generation.cpp)
make_extension(glrpc SOURCES glrpc.cpp REQUIRES rpc)

#----------------------------
set_property(DIRECTORY PROPERTY EXTENSIONS_LIST "${EXTENSIONS_LIST}")
