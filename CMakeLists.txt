cmake_minimum_required(VERSION 3.6.0)
project(fp16)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/src
)
file(GLOB SRCS
	"${PROJECT_SOURCE_DIR}/include/*"
	"${PROJECT_SOURCE_DIR}/src/*"
	"${PROJECT_SOURCE_DIR}/src/fp16/*"
	"${PROJECT_SOURCE_DIR}/src/fp16/math/*"
)

add_library(float16 STATIC ${SRCS})

