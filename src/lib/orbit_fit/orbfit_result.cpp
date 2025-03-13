#include <pybind11/pybind11.h>
namespace py = pybind11;

struct OrbfitResult {
    float csq;   // Chi-square value
    int ndof;    // Number of degrees of freedom
    float state[6];  // State vector
    float epoch;  // Epoch
    float cov[36];  // Covariance matrix
    int niter;  // Number of iterations

};

PYBIND11_MODULE(orbfit, m) {
    m.doc() = "Module exposing the OrbfitResult structure";

    // Bind the OrbfitResult struct as a Python class
    py::class_<OrbfitResult>(m, "OrbfitResult")
        .def(py::init<>())  // Expose the default constructor
        .def_readwrite("csq", &OrbfitResult::csq, "Chi-square value")
        .def_readwrite("ndof", &OrbfitResult::ndof, "Number of degrees of freedom");
        .def_readwrite("state", &OrbfitResult::state, "State vector")
        .def_readwrite("epoch", &OrbfitResult::epoch, "Epoch")
        .def_readwrite("cov", &OrbfitResult::cov, "Covariance matrix")
        .def_readwrite("niter", &OrbfitResult::niter, "Number of iterations");
}