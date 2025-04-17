#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace orbit_fit
{

    struct OrbfitResult
    {
        double csq;                  // Chi-square value
        int ndof;                    // Number of degrees of freedom
        std::array<double, 6> state; // State vector
        double epoch;                // Epoch
        std::array<double, 36> cov;  // Covariance matrix
        int niter;                   // Number of iterations
    };

    static void orbit_fit_result_bindings(py::module &m)
    {
        // Bind the OrbfitResult struct as a Python class
        py::class_<OrbfitResult>(m, "OrbfitResult")
            .def(py::init<>()) // Expose the default constructor
            .def_readwrite("csq", &orbit_fit::OrbfitResult::csq, "Chi-square value")
            .def_readwrite("ndof", &orbit_fit::OrbfitResult::ndof, "Number of degrees of freedom")
            .def_readwrite("state", &orbit_fit::OrbfitResult::state, "State vector")
            .def_readwrite("epoch", &orbit_fit::OrbfitResult::epoch, "Epoch")
            .def_readwrite("cov", &orbit_fit::OrbfitResult::cov, "Covariance matrix")
            .def_readwrite("niter", &orbit_fit::OrbfitResult::niter, "Number of iterations");
    }

} // namespace orbit_fit
