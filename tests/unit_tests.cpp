//#define CATCH_CONFIG_MAIN // automatically provideds a main() function
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "inputs.hpp"
#include <iostream>
#include <string>
#include <string.h>

std::string test_inputs::hf_path;

int main( int argc, char* argv[] )
{
    Catch::Session session; // There must be exactly one instance
    
    std::string hf_path;
    
    // Build a new parser on top of Catch's
    using namespace clara;
    auto cli
    = session.cli() // Get Catch's composite command line parser
    | Opt( hf_path, "Hartree-Fock path" ) // bind variable to a new option, with a hint string
    ["-d"]["--hf_path"]    // the option names it will respond to
    ("Path to the directory that contains the HF output for one molecular system to test on");        // description string for the help output
    
    // Now pass the new composite back to Catch so it uses that
    session.cli( cli );
    
    // Let Catch (using Clara) parse the command line
    int returnCode = session.applyCommandLine( argc, argv );
    if( returnCode != 0 ) // Indicates a command line error
        return returnCode;
    
    if(hf_path.size() > 0)
        std::cout << "hf_path: " << hf_path << std::endl;
    
    test_inputs::hf_path = hf_path;
    
    return session.run();
}

