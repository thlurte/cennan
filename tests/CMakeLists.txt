add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE cennan_lib)
target_include_directories(test_tensor PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_mmap_loader test_mmap_loader.cpp)
target_link_libraries(test_mmap_loader PRIVATE cennan_lib)
target_include_directories(test_mmap_loader PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_norm test_norm.cpp)
target_link_libraries(test_norm PRIVATE cennan_lib)
target_include_directories(test_norm PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE cennan_lib)
target_include_directories(test_gemm PRIVATE ${CMAKE_SOURCE_DIR}/include)

enable_testing()
add_test(NAME tensor_tests COMMAND test_tensor)
add_test(NAME mmap_loader_tests COMMAND test_mmap_loader)
add_test(NAME norm_tests COMMAND test_norm)
add_test(NAME gemm_tests COMMAND test_gemm)


add_executable(test_activations test_activations.cpp)
target_link_libraries(test_activations PRIVATE cennan_lib)
target_include_directories(test_activations PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME activations_tests COMMAND test_activations)
