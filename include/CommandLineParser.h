#include <iostream>
#include <string>
#include <vector>

class CommandLineParser {
public:
    CommandLineParser(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-i" && i + 1 < argc) {
                inputFileName = argv[i + 1];
                i++;  // Skip the next argument since it's the input file name.
            }
            else if (arg == "-o" && i + 1 < argc) {
                outputFileName = argv[i + 1];
                i++;  // Skip the next argument since it's the output file name.
            }
            else {
                // Handle unknown or invalid arguments here.
                std::cerr << "Unknown or invalid argument: " << arg << std::endl;
                PrintUsage();
                exit(1);
            }
        }
    }

    std::string GetInputFileName() const {
        return inputFileName;
    }

    std::string GetOutputFileName() const {
        return outputFileName;
    }

    void PrintUsage() const {
        std::cout << "Usage: ./test -i input.txt -o output.txt" << std::endl;
    }

private:
    std::string inputFileName;
    std::string outputFileName;
};
