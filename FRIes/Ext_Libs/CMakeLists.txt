
target_sources(FRIes PRIVATE
                argparse.c
                csvparser.c
                heap.c)

add_library(dcmt PRIVATE
            dcmt/check32.c
            dcmt/prescr.c
            dcmt/mt19937.c
            dcmt/eqdeg.c
            dcmt/genmtrand.c
            dcmt/seive.c
            dcmt/init.c)

add_subdirectory(dcmt)

target_link_libraries(fries PRIVATE dcmt)
