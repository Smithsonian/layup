#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "lib/orbit_fit/orbit_fit.cpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

std::string hello_world()
{
    std::cout << "Hello hello" << std::endl;
    return "Hello, World!";
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
