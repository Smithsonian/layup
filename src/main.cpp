#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "lib/orbit_fit/orbit_fit.cpp"
#include "lib/orbit_fit/orbit_fit_result.cpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

std::string hello_world() {
    return "Hello, World!";
}

namespace py = pybind11;


PYBIND11_MODULE(_core, m) {
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

    // m.def("gauss", &gauss, R"pbdoc(
    //     Gauss' method of initial orbit determination
    // )pbdoc");

    // Bind the OrbfitResult struct as a Python class
    py::class_<OrbfitResult>(m, "OrbfitResult")
        .def(py::init<>())  // Expose the default constructor
        .def_readwrite("csq", &OrbfitResult::csq, "Chi-square value")
        .def_readwrite("ndof", &OrbfitResult::ndof, "Number of degrees of freedom")
        .def_readwrite("state", &OrbfitResult::state, "State vector")
        .def_readwrite("epoch", &OrbfitResult::epoch, "Epoch")
        .def_readwrite("cov", &OrbfitResult::cov, "Covariance matrix")
        .def_readwrite("niter", &OrbfitResult::niter, "Number of iterations");


    // Bind AstrometryObservation type.
    py::class_<AstrometryObservation>(m, "AstrometryObservation")
        .def(py::init<double, double>(),
             py::arg("ra"), py::arg("dec"))
        .def_readonly("rho_hat", &AstrometryObservation::rho_hat,
                      "Computed unit direction vector (rho_hat)");


    // Bind StreakObservation type.
    py::class_<StreakObservation>(m, "StreakObservation")
        .def(py::init<double, double, double, double>(),
             py::arg("ra"), py::arg("dec"), py::arg("ra_rate"), py::arg("dec_rate"))
        .def_readonly("ra_rate", &StreakObservation::ra_rate)
        .def_readonly("dec_rate", &StreakObservation::dec_rate)
        .def_readonly("rho_hat", &StreakObservation::rho_hat,
                      "Computed unit direction vector (rho_hat)");

    // Bind the main Observation type with multiple overloaded constructors.
    py::class_<Observation>(m, "Observation")
        // Constructor for an Astrometry observation.
        // .def(py::init<double, double, double, const Eigen::Vector3d &, const Eigen::Vector3d &>(),
        .def(py::init<double, double, double, const std::array<double, 3> &, const std::array<double, 3> &>(),
             py::arg("ra"), py::arg("dec"), py::arg("epoch"),
             py::arg("observer_position"), py::arg("observer_velocity"),
             "Construct an Astrometry observation")
        // Constructor for a Streak observation.
        // .def(py::init<double, double, double, double, double, const Eigen::Vector3d &, const Eigen::Vector3d &>(),
        .def(py::init<double, double, double, double, double, const std::array<double, 3> &, const std::array<double, 3> &>(),
             py::arg("ra"), py::arg("dec"), py::arg("ra_rate"), py::arg("dec_rate"),
             py::arg("epoch"), py::arg("observer_position"), py::arg("observer_velocity"),
             "Construct a Streak observation")
        // Expose common members.
        // .def_readonly("rho_hat", &Observation::observation_type::rho_hat, "Computed unit direction vector (rho_hat)")
        .def_readonly("epoch", &Observation::epoch, "Observation epoch (as a double)")
        .def_readonly("observation_type", &Observation::observation_type, "Variant holding the observation data")
        .def_readonly("observer_position", &Observation::observer_position, "Observer position as a 3D vector")
        .def_readonly("observer_velocity", &Observation::observer_velocity, "Observer velocity as a 3D vector")
        .def_readonly("inverse_covariance", &Observation::inverse_covariance, "Optional inverse covariance matrix")
        .def_readonly("mag", &Observation::mag, "Optional magnitude")
        .def_readonly("mag_err", &Observation::mag_err, "Optional magnitude error");

    // m.def("orbit_fit", &orbit_fit, R"pbdoc(
    //     Nonlinear orbit fit using levenberg-marquardt
    // )pbdoc");
    orbit_fit::orbit_fit_bindings(m);


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}