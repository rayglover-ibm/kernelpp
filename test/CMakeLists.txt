cmake_minimum_required (VERSION 3.2)

include (CTest)
include ("${PROJECT_SOURCE_DIR}/third_party/DownloadProject/DownloadProject.cmake")

# download GoogleTest targets
download_project (
	PROJ            googletest
	GIT_REPOSITORY  https://github.com/google/googletest.git
	GIT_TAG         master
	UPDATE_DISCONNECTED 1
)

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# setup unit test libraries
add_subdirectory ("${googletest_SOURCE_DIR}" "${googletest_BINARY_DIR}")

# main test suite
add_executable (kernelpp_test
	"lib_test.cpp"
)
target_link_libraries (kernelpp_test
	kernelpp gtest gmock_main
)

add_test (
	NAME kernelpp_test_suite
	COMMAND kernelpp_test
)