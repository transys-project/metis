# Routing Simulation

## Installation

### Requirements

* OMNET++ [Download Link](https://omnetpp.org/download/)
* Boost C++ Library [Download Link](https://www.boost.org/users/download/)

### Run the simulation

* Make sure binary directory `/path/to/omnetpp/bin` has been included in PATH. If you use mingw terminal on Windows, it's included automatically.

* Run `opp_makemake --deep -f -I{path}` and replace `{path}` with path of your boost library.

* Run `make` to build the simulation program, you can use `make MODE=Release` and `make MODE=Debug` to select the compilation target.

* Run `./simulation -c NetBuilder` to start simulation, Add `-u Cmdenv` to close Qt frontend. Otherwise it will open a visulization interface and display
the simulation precess.