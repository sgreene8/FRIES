add_library(my_lapack SHARED lapack_wrappers.cpp)
find_package(LAPACK REQUIRED)
target_link_libraries(my_lapack PRIVATE ${LAPACK_LIBRARIES})

set_target_properties(my_lapack PROPERTIES PUBLIC_HEADER "lapack_wrappers.hpp;lapacke.h" )

install(
    TARGETS
        my_lapack
    RUNTIME
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        COMPONENT bin
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        COMPONENT lib
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/FRIES
        COMPONENT dev
)
