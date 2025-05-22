#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <sstream>

#include "lib/orbit_fit/orbit_fit.cpp"

// autodiff include
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

std::string hello_world()
{
    std::cout << "Hello hello" << std::endl;
    return "Hello, World!";
}

// autodiff hello world functions
// these are only around for unit testing the autodiff import
// taken from https://autodiff.github.io/tutorials/
// The single-variable function for which derivatives are needed
dual f(dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

std::string autodiff_hello_world()
{
    dual x = 2.0;                                 // the input variable x
    dual u = f(x);                                // the output variable u

    double dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx

    // std::cout << "u = " << u << std::endl;        // print the evaluated output u
    // std::cout << "du/dx = " << dudx << std::endl; // print the evaluated derivative du/dx
    std::stringstream ss;
    ss << "u = " << u << ";du/dx = " << dudx;
    std::string result = ss.str();
    return result;
}

namespace py = pybind11;

PYBIND11_MODULE(_core, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("hello_world", &hello_world, R"pbdoc(
        Say 'Hello, World!'
    )pbdoc");
    m.def("autodiff_hello_world", &autodiff_hello_world, R"pbdoc(
        Test autodiff hello world
    )pbdoc");

    orbit_fit::detection_bindings(m);
    orbit_fit::gauss_bindings(m);
    orbit_fit::orbit_fit_bindings(m);
    orbit_fit::orbit_fit_result_bindings(m);
    orbit_fit::predict_bindings(m);
    orbit_fit::predict_result_bindings(m);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
