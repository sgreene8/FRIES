install(
    FILES
        argparse.h
        csvparser.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/FRIES/Ext_Libs
)

add_subdirectory(dcmt)
