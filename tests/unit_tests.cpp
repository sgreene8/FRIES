#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "inputs.hpp"
#include <iostream>
#include <string>
#include <string.h>
#include <mpi.h>

std::string test_inputs::hf_path;
std::string test_inputs::out_path;

int main( int argc, char* argv[] )
{

    MPI_Init(NULL, NULL);
    Catch::Session session; // There must be exactly one instance
    
    std::string hf_path;
    std::string out_path;
    
    // Build a new parser on top of Catch's
    using namespace clara;
    auto cli
    = session.cli() // Get Catch's composite command line parser
    | Opt( hf_path, "Hartree-Fock path" ) // bind variable to a new option, with a hint string
    ["-h"]["--hf_path"]    // the option names it will respond to
    ("Path to the directory that contains the HF output for one molecular system to test on");        // description string for the help output
    cli |= Opt(out_path, "Output path")
    ["-r"]["--out_path"]
    ("Path to the directory where output files from the tests are saved");
    
    // Now pass the new composite back to Catch so it uses that
    session.cli( cli );
    
    // Let Catch (using Clara) parse the command line
    int returnCode = session.applyCommandLine( argc, argv );
    if( returnCode != 0 ) {// Indicates a command line error

        MPI_Finalize();
        return returnCode;
    }
    
    test_inputs::hf_path = hf_path;
    test_inputs::out_path = out_path;

    int result = session.run();

    MPI_Finalize();
    return result;
}
