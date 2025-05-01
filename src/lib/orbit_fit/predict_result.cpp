#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace orbit_fit
{

    typedef struct {
        // rho_x, rho_y, rho_z: Topocentric coordinates
        // obs_cov: Covariance matrix of the observation (2,2)
        // time
        std::array<double, 3> rho;
        std::array<double, 4> obs_cov;
        double epoch;
    } PredictResult;

    static void predict_result_bindings(py::module &m)
    {
        // Bind the PredictResult struct as a Python class
        py::class_<PredictResult>(m, "PredictResult")
            .def(py::init<>()) // Expose the default constructor
            .def_readwrite("rho", &orbit_fit::PredictResult::rho, "rho unit vector")
            .def_readwrite("obs_cov", &orbit_fit::PredictResult::obs_cov, "Covariance matrix of the observation")
            .def_readwrite("epoch", &orbit_fit::PredictResult::epoch, "Epoch");
    }

}