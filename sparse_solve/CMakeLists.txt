add_library(krylov STATIC
    "${CMAKE_CURRENT_LIST_DIR}/src/cg.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/gmres.cpp"
)
find_package(GSL REQUIRED)
target_include_directories(krylov PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(krylov PUBLIC GSL::gslcblas GSL::gsl)
target_compile_features(krylov PUBLIC cxx_std_23)