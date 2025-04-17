#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace orbit_fit {

struct gauss_soln {
    double root; 
    double epoch;

    double x; 
    double y;
    double z;

    double vx; 
    double vy;
    double vz;

};
    
}
