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
        // Non-gravitational Marsden parameters A1 (radial), A2 (transverse),
        // A3 (normal), au/day^2. Used both as seeds and as fitted results.
        // nongrav_mask is a bitmask of which were fitted (bit 1=A1, 2=A2, 4=A3;
        // 0 = pure 6-parameter fit). a{1,2,3}_unc are the 1-sigma formal
        // uncertainties (sqrt of the corresponding diagonal of the joint
        // covariance). All default-zero, so the 6-parameter path is unchanged.
        int nongrav_mask = 0;
        double a1 = 0.0, a2 = 0.0, a3 = 0.0;
        double a1_unc = 0.0, a2_unc = 0.0, a3_unc = 0.0;
        // Per-arc (piecewise-constant) non-grav amplitudes for the two-apparition
        // comet-linkage fit. With per_arc on, a1/a2/a3 above are the EARLIER arc
        // (arc A) and a{1,2,3}_arc2 the LATER arc (arc B); the state and g(r) are
        // shared. per_arc=false leaves the _arc2 fields zero (unchanged 6/7-param
        // behavior). Comparing arc A vs arc B amplitudes labels the outcome:
        // equal -> stable, changed -> recovered, orbit+g(r)-shared/A-differs -> split.
        bool per_arc = false;
        double a1_arc2 = 0.0, a2_arc2 = 0.0, a3_arc2 = 0.0;
        double a1_arc2_unc = 0.0, a2_arc2_unc = 0.0, a3_arc2_unc = 0.0;
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
            .def_readwrite("nongrav_mask", &orbit_fit::FitResult::nongrav_mask,
                           "Bitmask of fitted non-grav params (1=A1, 2=A2, 4=A3)")
            .def_readwrite("a1", &orbit_fit::FitResult::a1, "Non-grav A1 radial term (au/day^2)")
            .def_readwrite("a2", &orbit_fit::FitResult::a2, "Non-grav A2 transverse term (au/day^2)")
            .def_readwrite("a3", &orbit_fit::FitResult::a3, "Non-grav A3 normal term (au/day^2)")
            .def_readwrite("a1_unc", &orbit_fit::FitResult::a1_unc, "1-sigma uncertainty on A1 (au/day^2)")
            .def_readwrite("a2_unc", &orbit_fit::FitResult::a2_unc, "1-sigma uncertainty on A2 (au/day^2)")
            .def_readwrite("a3_unc", &orbit_fit::FitResult::a3_unc, "1-sigma uncertainty on A3 (au/day^2)")
            .def_readwrite("per_arc", &orbit_fit::FitResult::per_arc,
                           "Whether per-arc (piecewise-constant) non-grav amplitudes were fit")
            .def_readwrite("a1_arc2", &orbit_fit::FitResult::a1_arc2, "Later-arc (arc B) A1 (au/day^2)")
            .def_readwrite("a2_arc2", &orbit_fit::FitResult::a2_arc2, "Later-arc (arc B) A2 (au/day^2)")
            .def_readwrite("a3_arc2", &orbit_fit::FitResult::a3_arc2, "Later-arc (arc B) A3 (au/day^2)")
            .def_readwrite("a1_arc2_unc", &orbit_fit::FitResult::a1_arc2_unc, "1-sigma uncertainty on arc-B A1")
            .def_readwrite("a2_arc2_unc", &orbit_fit::FitResult::a2_arc2_unc, "1-sigma uncertainty on arc-B A2")
            .def_readwrite("a3_arc2_unc", &orbit_fit::FitResult::a3_arc2_unc, "1-sigma uncertainty on arc-B A3");
    }

} // namespace orbit_fit
