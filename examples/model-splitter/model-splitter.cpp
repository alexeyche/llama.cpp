
#include "llama.h"
#include <string>


int main(int argc, char ** argv) {
    assert(argc == 5);
    int first_layer = std::stoi(argv[1]);
    int last_layer = std::stoi(argv[2]);
    std::string fname_in = argv[3];
    std::string fname_out = argv[4];

    llama_model_split(fname_in.c_str(), fname_out.c_str(), first_layer, last_layer);
}