#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <pybind11/pybind11.h>

extern "C"
{
#include "rebound.h"
#include "assist.h"
}

#include "predict_result.cpp"

using std::cout;
using namespace Eigen;

double AU_M = 149597870700;
double SPEED_OF_LIGHT = 2.99792458e8 * 86400.0 / AU_M;

namespace orbit_fit
{
    // Does this need to report possible failures?
    int integrate_light_time(struct assist_extras *ax, int np, double t, reb_vec3d r_obs, double lt0, size_t iter, double speed_of_light)
    {

        struct reb_simulation *r = ax->sim;
        struct assist_ephem *ephem = ax->ephem;
        double lt = lt0;
        // Performs the light travel time correction between object and observatory iteratively for the object at a given reference time
        for (int i = 0; i < iter; i++)
        {
            assist_integrate_or_interpolate(ax, t - lt);

            double dx = r->particles[np].x - r_obs.x;
            double dy = r->particles[np].y - r_obs.y;
            double dz = r->particles[np].z - r_obs.z;
            double rho_mag = sqrt(dx * dx + dy * dy + dz * dz);
            if (r->status == REB_STATUS_GENERIC_ERROR)
            {
                printf("barf %lf %le %lf\n", t, lt, rho_mag);
                return 1;
            }

            lt = rho_mag / speed_of_light;
        }

        return 0;
    }

    void add_variational_particles(struct reb_simulation *r, size_t np, int *var)
    {

        int varx = reb_simulation_add_variation_1st_order(r, np);
        r->particles[varx].x = 1.;

        int vary = reb_simulation_add_variation_1st_order(r, np);
        r->particles[vary].y = 1.;

        int varz = reb_simulation_add_variation_1st_order(r, np);
        r->particles[varz].z = 1.;

        int varvx = reb_simulation_add_variation_1st_order(r, np);
        r->particles[varvx].vx = 1.;

        int varvy = reb_simulation_add_variation_1st_order(r, np);
        r->particles[varvy].vy = 1.;

        int varvz = reb_simulation_add_variation_1st_order(r, np);
        r->particles[varvz].vz = 1.;

        *var = varx;
    }

    PredictResult predict(struct assist_ephem *ephem,
                          struct reb_particle p0, double epoch,
                          Observation this_det,
                          Eigen::MatrixXd &cov,
                          Eigen::MatrixXd &obs_cov)
    {

        // Takes an ephemeris object, a simulation,
        // a detection/observation (only the time and observatory
        // state matter), and an orbit covariance
        // as arguments.
        // Returns the model observation, the partials, and
        // the observation covariance.

        // Pass in this simulation stuff to keep it flexible
        struct reb_simulation *r = reb_simulation_create();
        struct assist_extras *ax = assist_attach(r, ephem);

        // 0. Set initial time, relative to ephem->jd_ref
        r->t = epoch - ephem->jd_ref;

        // 1. Add the main particle to the REBOUND simulation.
        reb_simulation_add(r, p0);

        // 2. incorporate the variational particles
        int var;
        add_variational_particles(r, 0, &var);

        double jd_tdb = this_det.epoch;

        double xe = this_det.observer_position[0];
        double ye = this_det.observer_position[1];
        double ze = this_det.observer_position[2];

        reb_vec3d r_obs = {xe, ye, ze};
        double t_obs = jd_tdb - ephem->jd_ref;

        int j = 0;
        // Make the number of iterations flexible
        // Could make integrate_light_time return
        // rho and its components, since those are
        // already computed for the light time iteration.

        integrate_light_time(ax, j, t_obs, r_obs, 0.0, 4, SPEED_OF_LIGHT);

        double rho_x = r->particles[j].x - xe;
        double rho_y = r->particles[j].y - ye;
        double rho_z = r->particles[j].z - ze;
        double rho = sqrt(rho_x * rho_x + rho_y * rho_y + rho_z * rho_z);

        rho_x /= rho;
        rho_y /= rho;
        rho_z /= rho;

        Eigen::Vector3d rho_matrix;
        rho_matrix.x() = rho_x;
        rho_matrix.y() = rho_y;
        rho_matrix.z() = rho_z;

        Eigen::Vector3d Av = a_vec_from_rho_hat(rho_matrix);
        Eigen::Vector3d Dv = d_vec_from_rho_hat(rho_matrix);

        double Ax = Av.x();
        double Ay = Av.y();
        double Az = Av.z();

        double Dx = Dv.x();
        double Dy = Dv.y();
        double Dz = Dv.z();

        // Calculate the partial deriviatives of the model
        // observations with respect to the initial conditions

        double dxk[6], dyk[6], dzk[6];
        double drho_x[6], drho_y[6], drho_z[6];

        for (size_t i = 0; i < 6; i++)
        {
            size_t vn = var + i;
            dxk[i] = r->particles[vn].x;
            dyk[i] = r->particles[vn].y;
            dzk[i] = r->particles[vn].z;
        }

        double invd = 1. / rho;

        double ddist[6];
        double dx_resid[6];
        double dy_resid[6];
        for (size_t i = 0; i < 6; i++)
        {
            // Derivative of topocentric distance w.r.t. parameters
            ddist[i] = rho_x * dxk[i] + rho_y * dyk[i] + rho_z * dzk[i];

            drho_x[i] = dxk[i] * invd - rho_x * invd * ddist[i] - r->particles[0].vx * ddist[i] * invd / SPEED_OF_LIGHT;
            drho_y[i] = dyk[i] * invd - rho_y * invd * ddist[i] - r->particles[0].vy * ddist[i] * invd / SPEED_OF_LIGHT;
            drho_z[i] = dzk[i] * invd - rho_z * invd * ddist[i] - r->particles[0].vz * ddist[i] * invd / SPEED_OF_LIGHT;

            dx_resid[i] = -(drho_x[i] * Ax + drho_y[i] * Ay + drho_z[i] * Az);
            dy_resid[i] = -(drho_x[i] * Dx + drho_y[i] * Dy + drho_z[i] * Dz);
        }

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
        B.resize(2, 6);

        for (size_t j = 0; j < 6; j++)
        {
            B(0, j) = dx_resid[j];
            B(1, j) = dy_resid[j];
        }

        // Eigen::MatrixXd obs_cov(6, 6);
        obs_cov = B * cov * B.transpose();

        assist_free(ax);
        reb_simulation_free(r);

        PredictResult result;
        result.rho[0] = rho_x;
        result.rho[1] = rho_y;
        result.rho[2] = rho_z;
        result.obs_cov[0] = obs_cov(0, 0);
        result.obs_cov[1] = obs_cov(0, 1);
        result.obs_cov[2] = obs_cov(1, 0);
        result.obs_cov[3] = obs_cov(1, 1);
        result.epoch = epoch;

        return result;
    }

    PredictResult predict_from_fit_result(struct assist_ephem *ephem,
                                          FitResult fit,
                                          Observation obs_position,
                                          Eigen::MatrixXd &cov,
                                          Eigen::MatrixXd &obs_cov)
    {
        struct reb_particle particle;
        particle.x = fit.state[0];
        particle.y = fit.state[1];
        particle.z = fit.state[2];
        particle.vx = fit.state[3];
        particle.vy = fit.state[4];
        particle.vz = fit.state[5];

        PredictResult res = predict(
            ephem,
            particle,
            fit.epoch,
            obs_position,
            cov,
            obs_cov);

        return res;
    }

    std::vector<PredictResult> predict_sequence(struct assist_ephem *ephem,
                                                FitResult fit,
                                                std::vector<Observation> &detections,
                                                Eigen::MatrixXd &cov)
    {
        std::vector<PredictResult> results;

        Eigen::MatrixXd obs_cov(6, 6);

        for (int i = 0; i < detections.size(); i++)
        {
            double this_epoch = detections[i].epoch;
            Observation this_det = detections[i];
            PredictResult this_result = predict_from_fit_result(ephem, fit, this_det, cov, obs_cov);
            results.push_back(this_result);
        }

        return results;
    }

    Eigen::MatrixXd numpy_to_eigen(std::vector<double> arr, size_t m, size_t n)
    {
        // Utility function to convert a numpy array to an Eigen matrix
        // Convert a 1D array to a 2D Eigen matrix
        // Assuming arr is of size m*n
        if (arr.size() != m * n)
        {
            throw std::invalid_argument("Array size does not match the specified dimensions.");
        }

        Eigen::MatrixXd result(m, n);
        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                result(i, j) = arr[i * n + j];
            }
        }
        return result;
    }

    static void predict_bindings(py::module &m)
    {
        py::class_<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(m, "MatrixXd")
            .def(py::init<>());
        m.def("predict", &orbit_fit::predict, R"pbdoc(predict)pbdoc");
        m.def("predict_sequence", &orbit_fit::predict_sequence, R"pbdoc(predict_sequence)pbdoc");
        m.def("numpy_to_eigen", &orbit_fit::numpy_to_eigen, R"pbdoc(numpy_to_eigen)pbdoc");
    }
} // namespace orbit_fit
