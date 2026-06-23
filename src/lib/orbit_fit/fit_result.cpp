#include <string>

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

namespace py = pybind11;

namespace orbit_fit
{
    using namespace Eigen;

    typedef struct
    {
        double csq;                    // Chi-square value
        int ndof;                      // Number of degrees of freedom
        double epoch;                  // Epoch
        double root;                   // Root value (gauss)
        std::array<double, 6> state;   // State vector
        std::array<double, 36> cov;    // Covariance matrix
        int niter;                     // Number of iterations
        std::string method;            // Method used for fitting
        int flag;                      // Flag indicating the success of the fit
        // Non-gravitational A2 (transverse Marsden term, au/day^2). Used both as
        // a seed (when fitting non-gravs) and as the fitted result. fit_a2 marks
        // whether A2 was a fitted parameter; a2_unc is its 1-sigma formal
        // uncertainty (sqrt of the A2 diagonal of the joint 7x7 covariance).
        // Default-zero, so the 6-parameter path is unchanged.
        bool fit_a2 = false;
        double a2 = 0.0;
        double a2_unc = 0.0;
    } FitResult;

    static void orbit_fit_result_bindings(py::module &m)
    {
        // Bind the OrbfitResult struct as a Python class
        py::class_<FitResult>(m, "FitResult")
            .def(py::init<>()) // Expose the default constructor
            .def_readwrite("csq", &orbit_fit::FitResult::csq, "Chi-square value")
            .def_readwrite("ndof", &orbit_fit::FitResult::ndof, "Number of degrees of freedom")
            .def_readwrite("state", &orbit_fit::FitResult::state, "State vector")
            .def_readwrite("epoch", &orbit_fit::FitResult::epoch, "Epoch")
            .def_readwrite("cov", &orbit_fit::FitResult::cov, "Covariance matrix")
            .def_readwrite("niter", &orbit_fit::FitResult::niter, "Number of iterations")
            .def_readwrite("method", &orbit_fit::FitResult::method, "Method used for fitting")
            .def_readwrite("flag", &orbit_fit::FitResult::flag, "Flag indicating the success of the fit")
            .def_readwrite("fit_a2", &orbit_fit::FitResult::fit_a2, "Whether the non-grav A2 was fitted")
            .def_readwrite("a2", &orbit_fit::FitResult::a2, "Non-grav A2 transverse term (au/day^2)")
            .def_readwrite("a2_unc", &orbit_fit::FitResult::a2_unc, "1-sigma uncertainty on A2 (au/day^2)");
    }

} // namespace orbit_fit
