CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(googletest-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googletest
	URL https://github.com/google/googletest/archive/release-1.8.0.zip
	URL_HASH SHA256=f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf
	SOURCE_DIR "${CMAKE_SOURCE_DIR}/deps/gtest/googletest"
	BINARY_DIR "${CMAKE_SOURCE_DIR}/deps/gtest/googletest"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)