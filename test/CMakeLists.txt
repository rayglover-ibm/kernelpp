cmake_minimum_required (VERSION 3.2)

include (CTest)

# retrieve DownloadProject
set (src "https://raw.githubusercontent.com/Crascit/DownloadProject/master")
set (dest "${CMAKE_CURRENT_BINARY_DIR}/tmp.DownloadProject")

foreach (file
	"DownloadProject.cmake"
	"DownloadProject.CMakeLists.cmake.in")
	file (DOWNLOAD "${src}/${file}" "${dest}/${file}" STATUS "retrieving ${file}")
endforeach ()

# download GoogleTest targets
include ("${dest}/DownloadProject.cmake")
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
target_compile_options (kernelpp_test PUBLIC -g)

add_test (
	NAME kernelpp_test_suite
	COMMAND kernelpp_test
)