add_library(dcmt SHARED
            check32.c
            prescr.c
            mt19937.c
            eqdeg.c
            genmtrand.c
            seive.c
            init.c)

install(
    FILES
        dc.h
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/FRIES/Ext_Libs/dcmt
    )

install(
    TARGETS
        dcmt
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        COMPONENT lib
)
