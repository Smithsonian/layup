// This code requires the following:
// 1. The rebound library
// 2. The assist library
// 3. The Eigen header-only library
//
// It will probably need either pybind11, eigenpy, or something like
// that to connect it to python

// To Do:
// 1. Set up a make file, or something like that.
// 2. Make sure that the physical constants are consistent
//    with the rest of the code
// 3. Work on getting better data weights.
// 4. Work on outlier rejection.
// 5. Write code to handle doppler and range observations
// 6. Write code to handle shift+stack observations (RA/Dec + rates)

// Try gauss's method with an unevenly spaced triplet.  DONE

// Put in a check on the residual as part of the convergence
// criteria.

// Use gauss to get a decent start for fitting a segment of the data.
// Then use the results of that as an initial guess for fitting a
// larger segment of data.

// Important issues:
// 1. Obtaining reliable initial orbit determination for the nonlinear fits.
// 2. Making sure the weight matrix is as good as it can be
// 3. Identifying and removing outliers

// Later issues:
// 1. Deflection of light

// Compile with something like this
// g++ -std=c++11 -I../..//src/ -I../../../rebound/src/ -I/Users/mholman/eigen-3.4.0 -Wl,-rpath,./ -Wpointer-arith -D_GNU_SOURCE -O3 -fPIC -I/usr/local/include -Wall -g  -Wno-unknown-pragmas -D_APPLE -DSERVER -DGITHASH=e99d8d73f0aa7fb7bf150c680a2d581f43d3a8be orbit_fit.cpp -L. -lassist -lrebound -L/usr/local/lib -o orbit_fit

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <pybind11/pybind11.h>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <variant>
#include <optional>

#include "orbit_fit.h"
#include "../gauss/gauss.cpp"
#include "predict.cpp"

using std::cout;
namespace py = pybind11;

namespace orbit_fit
{
    struct reb_particle read_initial_conditions(const char *ic_file_name, double *epoch)
    {

        FILE *ic_file;
        ic_file = fopen(ic_file_name, "r");

        struct reb_particle p0;

        double epch;
        double x0, y0, z0;
        double vx0, vy0, vz0;

        fscanf(ic_file, "%lf %lf %lf %lf %lf %lf %lf\n",
               &epch, &x0, &y0, &z0, &vx0, &vy0, &vz0);

        *epoch = epch;
        p0.x = x0;
        p0.y = y0;
        p0.z = z0;
        p0.vx = vx0;
        p0.vy = vy0;
        p0.vz = vz0;

        return p0;
    }

    void print_initial_condition(struct reb_particle p0, double epoch)
    {

        printf("%lf %.16le %.16le %.16le %.16le %.16le %.16le\n", epoch, p0.x, p0.y, p0.z, p0.vx, p0.vy, p0.vz);

        return;
    }

    void read_detections(const char *data_file_name,
                         std::vector<detection> &detections,
                         std::vector<double> &times)
    {

        FILE *detections_file;
        detections_file = fopen(data_file_name, "r");

        char obsCode[10];
        char objID[20];
        char mag_str[6];
        char filt[6];
        double theta_x, theta_y, theta_z;
        double xe, ye, ze;
        double jd_tdb;
        double ast_unc;

        // Define an observation structure or class
        // Write a function to read observations

        while (fscanf(detections_file, "%s %s %s %s %lf %lf %lf %lf %lf %lf %lf %lf\n",
                      objID, obsCode, mag_str, filt, &jd_tdb, &theta_x, &theta_y, &theta_z, &xe, &ye, &ze, &ast_unc) != EOF)
        {
            detection this_det = detection();

            this_det.jd_tdb = jd_tdb;
            this_det.theta_x = theta_x;
            this_det.theta_y = theta_y;
            this_det.theta_z = theta_z;

            this_det.xe = xe;
            this_det.ye = ye;
            this_det.ze = ze;

            this_det.obsCode = obsCode;

            // Read these from the file
            this_det.ra_unc = (ast_unc / 206265);
            this_det.dec_unc = (ast_unc / 206265);

            // compute A and D vectors, the local tangent plane trick from Danby.

            double Ax =  -theta_y;
            double Ay =   theta_x;
            double Az =   0.0;

            double A = sqrt(Ax * Ax + Ay * Ay + Az * Az);
            Ax /= A;
            Ay /= A;
            Az /= A;

            this_det.Ax = Ax;
            this_det.Ay = Ay;
            this_det.Az = Az;

            double sd = theta_z;
            double cd = sqrt(1-theta_z*theta_z);
            double ca = theta_x/cd;
            double sa = theta_y/cd;
        
            double Dx = -sd*ca;
            double Dy = -sd*sa;
            double Dz = cd;        

            double D = sqrt(Dx * Dx + Dy * Dy + Dz * Dz);
            Dx /= D;
            Dy /= D;
            Dz /= D;

            this_det.Dx = Dx;
            this_det.Dy = Dy;
            this_det.Dz = Dz;

            detections.push_back(this_det);
            times.push_back(jd_tdb);
        }
    }

    void compute_single_residuals(struct assist_ephem *ephem,
                                  struct assist_extras *ax,
                                  int var,
                                  Observation this_det,
                                  residuals &resid,
                                  partials &parts)
    {

        // Takes a detection/observation and
        // a simulation as arguments.
        // Returns the model observation, the
        // residuals, and the partial derivatives.

        struct reb_simulation *r = ax->sim;

        double jd_tdb = this_det.epoch;

        double xe = this_det.observer_position[0];
        double ye = this_det.observer_position[1];
        double ze = this_det.observer_position[2];

        Eigen::Vector3d Av = this_det.a_vec;
        Eigen::Vector3d Dv = this_det.d_vec;

        double Ax = Av.x();
        double Ay = Av.y();
        double Az = Av.z();

        double Dx = Dv.x();
        double Dy = Dv.y();
        double Dz = Dv.z();

        // 5. compare the model result to the observation.
        //   This means dotting the model unit vector with the
        //   A and D vectors of the observation

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

        // Put the residuals into a struct, so that the
        // struct can be easily managed in a vector.
        resid.x_resid = -(rho_x * Ax + rho_y * Ay + rho_z * Az);
        resid.y_resid = -(rho_x * Dx + rho_y * Dy + rho_z * Dz);

        // 6. Calculate the partial deriviatives of the model
        //    observations with respect to the initial conditions

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

        // Likewise, put the partials into a struct so that they can be more easily
        // managed.

        for (size_t i = 0; i < 6; i++)
        {
            parts.x_partials.push_back(dx_resid[i]);
        }
        for (size_t i = 0; i < 6; i++)
        {
            parts.y_partials.push_back(dy_resid[i]);
        }
    }

    // This routine requires that resid_vec and partials_vec are
    // preallocated because the indices in out_seq will be used for
    // random access of locations in those vectors.
    // Also, in_seq and out_seq must be of the same length.
    void compute_residuals_sequence(struct assist_ephem *ephem,
                                    struct reb_particle p0, double epoch,
                                    std::vector<Observation> &detections,
                                    std::vector<residuals> &resid_vec,
                                    std::vector<partials> &partials_vec,
                                    std::vector<size_t> &in_seq,
                                    std::vector<size_t> &out_seq)
    {

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

        // 3. iterate over a sequence of detections.
        for (size_t i = 0; i < in_seq.size(); ++i)
        {

            size_t j = in_seq[i];
            size_t k = out_seq[i];

            residuals resids;
            partials parts;
            Observation this_det = detections[j];
            compute_single_residuals(ephem, ax, var,
                                     this_det,
                                     resids,
                                     parts);
            resid_vec[k] = resids;
            partials_vec[k] = parts;
        }

        assist_free(ax);
        reb_simulation_free(r);

        return;
    }

    void compute_residuals(struct assist_ephem *ephem,
                           struct reb_particle p0, double epoch,
                           std::vector<Observation> &detections,
                           std::vector<residuals> &resid_vec,
                           std::vector<partials> &partials_vec,
                           std::vector<size_t> &forward_in_seq,
                           std::vector<size_t> &forward_out_seq,
                           std::vector<size_t> &reverse_in_seq,
                           std::vector<size_t> &reverse_out_seq)
    {

        compute_residuals_sequence(ephem, p0, epoch,
                                   detections,
                                   resid_vec,
                                   partials_vec,
                                   forward_in_seq,
                                   forward_out_seq);

        compute_residuals_sequence(ephem, p0, epoch,
                                   detections,
                                   resid_vec,
                                   partials_vec,
                                   reverse_in_seq,
                                   reverse_out_seq);
    }

    // Consider having this accept Eigen matrices as
    // input
    void compute_dX(std::vector<residuals> &resid_vec,
                    std::vector<partials> &partials_vec,
                    Eigen::SparseMatrix<double> W,
                    Eigen::MatrixXd &dX,
                    Eigen::MatrixXd &C,
                    Eigen::MatrixXd &chi2,
                    Eigen::Matrix<double, 6, 1> &grad,
                    double lambda)
    {

        const int mlength = (int)partials_vec.size();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
        Eigen::MatrixXd eye = MatrixXd::Identity(6, 6);
        B.resize(mlength * 2, 6);

        for (size_t i = 0; i < partials_vec.size(); i++)
        {
            for (size_t j = 0; j < 6; j++)
            {
                B(2 * i, j) = partials_vec[i].x_partials[j];
                B(2 * i + 1, j) = partials_vec[i].y_partials[j];
            }
        }

        Eigen::MatrixXd resid_v(mlength * 2, 1);
        for (size_t i = 0; i < resid_vec.size(); i++)
        {
            resid_v(2 * i) = resid_vec[i].x_resid;
            resid_v(2 * i + 1) = resid_vec[i].y_resid;
        }

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Bt = B.transpose();
        C = Bt * W * B + lambda * eye; // This is where the extra term for LM is.

        grad = Bt * W * resid_v;

        dX = C.colPivHouseholderQr().solve(-grad);
        // An alternative looks like this.  It's probably less stable.
        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G = C.inverse();
        // dX = -G * Bt * W * resid_v;

        chi2 = resid_v.transpose() * W * resid_v;
        // reset for inverse covariance
        C -= lambda * eye;
    }

    void identify_outliers(std::vector<detection> &detections,
                           std::vector<residuals> &resid_vec,
                           std::vector<bool> &reject)
    {
    }

    void print_residuals(std::vector<detection> &detections,
                         std::vector<residuals> &resid_vec,
                         std::vector<partials> &partials_vec)
    {

        for (size_t j = 0; j < resid_vec.size(); j++)
        {

            detection this_det = detections[j];

            printf("%lf %s ", this_det.jd_tdb, this_det.obsCode.c_str());

            residuals resids = resid_vec[j];
            printf("%7.3lf %7.3lf ", resids.x_resid * 206265, resids.y_resid * 206265);

            partials parts = partials_vec[j];

            for (size_t i = 0; i < 6; i++)
            {

                printf("%7.3le %7.3le ", parts.x_partials[i], parts.y_partials[i]);
            }

            printf("%7.3le %7.3le ", this_det.ra_unc * 206265, this_det.dec_unc * 206265);

            printf("\n");
        }
    }

    int converged(Eigen::MatrixXd dX, double eps, double chi2, size_t ndof, double thresh)
    {

        if ((chi2 > ndof) > thresh)
        {
            return 2;
        }
        for (size_t i = 0; i < dX.size(); i++)
        {
            if (abs(dX(i)) > eps)
            {
                return 0;
            }
        }
        return 1;
    }

    typedef Eigen::Triplet<double> T;

    Eigen::SparseMatrix<double> get_weight_matrix(std::vector<Observation> &detections)
    {
        const int mlength = (int)detections.size();
        std::vector<T> tripletList;
        tripletList.reserve(2 * mlength);
        Eigen::SparseMatrix<double> W(mlength * 2, mlength * 2);

        for (size_t i = 0; i < mlength; i++)
        {
            double x_unc = *detections[i].ra_unc;
            double w2_x = 1.0 / (x_unc * x_unc);
            tripletList.push_back(T(2 * i, 2 * i, w2_x));
            double y_unc = *detections[i].dec_unc;
            double w2_y = 1.0 / (y_unc * y_unc);
            tripletList.push_back(T(2 * i + 1, 2 * i + 1, w2_y));
        }
        W.setFromTriplets(tripletList.begin(), tripletList.end());

        return W;
    }

    void create_sequences(std::vector<double> &times,
                          double epoch,
                          std::vector<size_t> &reverse_in_seq,
                          std::vector<size_t> &reverse_out_seq,
                          std::vector<size_t> &forward_in_seq,
                          std::vector<size_t> &forward_out_seq)
    {

        for (size_t i = 0; i < times.size(); i++)
        {
            if (times[i] > epoch)
            {
                forward_in_seq.push_back(i);
                forward_out_seq.push_back(i);
            }
            else
            {
                reverse_in_seq.push_back(i);
                reverse_out_seq.push_back(i);
            }
        }

        std::reverse(reverse_in_seq.begin(), reverse_in_seq.end());
        std::reverse(reverse_out_seq.begin(), reverse_out_seq.end());
    }

    int orbit_fit(struct assist_ephem *ephem,
                  struct reb_particle &p0,
                  double epoch,                            // Result from gauss
                  std::vector<Observation> &detections, // In Observation
                  std::vector<residuals> &resid_vec,    // Compute as part of run
                  std::vector<partials> &partials_vec,    // Compute as part of run
                  size_t &iters,                        // runtime
                  double &chi2_final,                    // result
                  Eigen::MatrixXd &cov,
                  double eps, // constant
                  size_t iter_max)
    { // runtime

        std::vector<size_t> reverse_in_seq;
        std::vector<size_t> reverse_out_seq;

        std::vector<size_t> forward_in_seq;
        std::vector<size_t> forward_out_seq;

        std::vector<double> times(detections.size());

        for (int i = 0; i < detections.size(); i++)
        {
            times[i] = detections[i].epoch;
        }

        create_sequences(times, epoch,
                         reverse_in_seq, reverse_out_seq,
                         forward_in_seq, forward_out_seq);

        Eigen::SparseMatrix<double> W = get_weight_matrix(detections);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> chi2;
        Eigen::MatrixXd dX;

        int flag = 1;

        double chi2_prev = HUGE_VAL;

        double rho_accept = 0.1;

        // Do an initial step
        double lambda = (206265.0 * 206265.0) / 1000;
        for (iters = 0; iters < iter_max; iters++)
        {

            compute_residuals(ephem, p0, epoch,
                              detections,
                              resid_vec,
                              partials_vec,
                              forward_in_seq,
                              forward_out_seq,
                              reverse_in_seq,
                              reverse_out_seq);

            Eigen::Matrix<double, 6, 1> grad;
            compute_dX(resid_vec, partials_vec, W, dX, C, chi2, grad, lambda);

            double chi2_d = chi2(0, 0);

            double rho = (chi2_prev - chi2_d) / (dX.transpose() * (lambda * dX - grad)).norm();

            if (rho > rho_accept)
            {

                // Accept the step
                // Reduce lambda
                // Update the state
                // Repeat, unless too many iterations

                lambda *= 0.5; // reduce lambda and update the state
                p0.x += dX(0);
                p0.y += dX(1);
                p0.z += dX(2);
                p0.vx += dX(3);
                p0.vy += dX(4);
                p0.vz += dX(5);
                chi2_prev = chi2_d;
            }
            else
            {

                // Reject the step
                // leave the state
                // increase lambda
                // repeat, unless too many iterations

                lambda *= 2.0;
            }

            size_t ndof = detections.size() * 2 - 6;
            double thresh = 10;
            int cflag = converged(dX, eps, chi2_d, ndof, thresh);
            if (cflag)
            {
                flag = 0;
                chi2_final = chi2_d;
                break;
            }
        }

        size_t ndof = detections.size() * 2 - 6;
        double thresh = 10.0;
        if ((chi2_final / ndof) > thresh)
        {
            flag = 2;
        }

        cov = C.inverse();

        return flag;
    }

    // Go through the detections in reverse order, looking for
    // a set of three detections such that each adjacent pair is
    // separated by more than interval_min and less than interval_max.
    std::vector<std::vector<size_t>> IOD_indices(std::vector<Observation> &detections,
                                                 double interval0_min,
                                                 double interval0_max,
                                                 double interval1_min,
                                                 double interval1_max,
                                                 size_t max_count,
                                                 size_t i_start)
    {

        size_t cnt = 0;
        std::vector<std::vector<size_t>> res;
        for (int i = i_start; i >= 2; i--)
        {

            size_t idx_i = (size_t)i;
            Observation d2 = detections[idx_i];
            double t2 = d2.epoch;

            for (int j = i - 1; j >= 1; j--)
            {

                size_t idx_j = (size_t)j;
                Observation d1 = detections[j];
                double t1 = d1.epoch;

                if (fabs(t2 - t1) < interval0_min || fabs(t2 - t1) >= interval0_max)
                    continue;

                for (int k = j - 1; k >= 0; k--)
                {

                    size_t idx_k = (size_t)k;
                    Observation d0 = detections[idx_k];
                    double t0 = d0.epoch;

                    if (fabs(t1 - t0) < interval1_min || fabs(t1 - t0) >= interval1_max)
                        continue;

                    cnt++;

                    if (cnt > max_count)
                        return res;

                    res.push_back({idx_k, idx_j, idx_i});
                    // printf("%lu %lu %lu %lf %lf %lf\n", idx_k, idx_j, idx_i, t0, t1, t2);
                }
            }
        }

        return res;
    }

        // Go through the detections in reverse order, looking for
    // a set of three detections such that each adjacent pair is
    // separated by more than interval_min and less than interval_max.
    std::vector<std::vector<size_t>> IOD_indices_forward(std::vector<Observation> &detections,
							 double interval0_min,
							 double interval0_max,
							 double interval1_min,
							 double interval1_max,
							 size_t max_count,
							 size_t i_start)
    {

        size_t cnt = 0;
        std::vector<std::vector<size_t>> res;
	size_t ndets = detections.size();
        for (int i = i_start; i <= ndets-3; i++)
        {

            size_t idx_i = (size_t)i;
            Observation d0 = detections[idx_i];
            double t0 = d0.epoch;

            for (int j = i + 1; j <= ndets-2; j++)
            {

                size_t idx_j = (size_t)j;
                Observation d1 = detections[j];
                double t1 = d1.epoch;

                if (fabs(t1 - t0) < interval0_min || fabs(t1 - t0) >= interval0_max)
                    continue;

                for (int k = j + 1; k <= ndets-1; k++)
                {

                    size_t idx_k = (size_t)k;
                    Observation d2 = detections[idx_k];
                    double t2 = d2.epoch;

                    if (fabs(t2 - t1) < interval1_min || fabs(t2 - t1) >= interval1_max)
                        continue;

                    cnt++;

                    if (cnt > max_count)
                        return res;

                    res.push_back({idx_i, idx_j, idx_k});
                    // printf("%lu %lu %lu %lf %lf %lf\n", idx_k, idx_j, idx_i, t0, t1, t2);
                }
            }
        }

        return res;
    }

    // Initial orbit and covariance matrix
    // Table of times, observations and uncertainties:
    // x, y, z unit vectors; dRA, dDec (and possibly covariance)
    // Table of observatory positions at the observation times

    // Incorporate these types of observations:
    // astrometry (unit vector)
    // radar: range and doppler
    // shift+stack

    struct assist_ephem * get_ephem(std::string cache_dir)
    {
        std::string ephem_kernel = cache_dir + "/linux_p1550p2650.440";
        std::string small_bodies_kernel = cache_dir + "/sb441-n16.bsp";

        // Allocate space for the string plus the null-terminator
        char *ephem_kernel_char = new char[ephem_kernel.length() + 1];
        std::strncpy(ephem_kernel_char, ephem_kernel.c_str(), ephem_kernel.length());
        ephem_kernel_char[ephem_kernel.length()] = '\0'; // Ensure null-termination

        char *small_bodies_kernel_char = new char[small_bodies_kernel.length() + 1];
        std::strncpy(small_bodies_kernel_char, small_bodies_kernel.c_str(), small_bodies_kernel.length());
        small_bodies_kernel_char[small_bodies_kernel.length()] = '\0'; // Ensure null-termination

        struct assist_ephem *ephem = assist_ephem_create(ephem_kernel_char, small_bodies_kernel_char);
        if (!ephem)
        {
            printf("Cannot create ephemeris structure.\n");
            exit(-1);
        }
        return ephem;
    }

    FitResult run_from_vector_prev(struct assist_ephem *ephem, std::vector<Observation> &detections_full)
    {

        size_t start_i = detections_full.size() - 1;

        // First find a triple of detections in the full data set.
        std::vector<std::vector<size_t>> idx = IOD_indices(detections_full, 0.5, 100.0, 0.01, 100.0, 3, start_i);

        int success = 1;
        size_t iters;
        double chi2_final;
        double result_epoch;
        size_t dof;
        struct reb_particle p1;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> final_cov;

        for (size_t i = 0; i < idx.size(); i++)
        {

            std::vector<size_t> indices = idx[i];
            size_t id0 = indices[0];
            size_t id1 = indices[1];
            size_t id2 = indices[2];

            printf("gauss indices: %lu %lu %lu\n", id0, id1, id2);
            fflush(stdout);

            Observation d0 = detections_full[id0];
            Observation d1 = detections_full[id1];
            Observation d2 = detections_full[id2];

            // Put this in a better place
            double GMtotal = 0.00029630927487993194;

            std::optional<std::vector<FitResult>> res = gauss(GMtotal, d0, d1, d2, 0.0001, SPEED_OF_LIGHT);
            if (!res.has_value())
            {
                printf("gauss failed.  Try another triplet.\n");
                fflush(stdout);
                continue;
            }

	    printf("gauss rho: %le\n", res.value()[0].root);

	    // This bit is for testing TNOs.
	    double inner_root_thresh = 0.1;
	    double outer_root_thresh = 1000.;	    
	    if(res.value()[0].root>outer_root_thresh){
		printf("Too far!\n");
		continue;
	    }

	    if(res.value()[0].root<=inner_root_thresh){
		printf("Too close!\n");
		continue;
	    }
	    
            // Now get a segment of data that spans the triplet and that uses up to
            // a certain number of points, if they are available in a time window.
            // This should be a parameter
            size_t num = 1000;

            size_t curr_num = id2 - id1 + 1;

            size_t start_index;
            size_t end_index;
            if (curr_num >= num)
            {
                // We already have enough data and
                // don't need to move the limits
                start_index = id0;
                end_index = id1;
            }
            else if ((detections_full.size() - id0) >= num)
            {
                // There are enough points if we include
                // only the later points, we only need to
                // later limit.
                start_index = id0;
                end_index = id0 + num;
            }
            else if (detections_full.size() >= num)
            {
                end_index = detections_full.size();
                start_index = detections_full.size() - num;
            }
            else
            {
                start_index = 0;
                end_index = detections_full.size();
            }

            std::vector<Observation>::const_iterator first_d = detections_full.begin() + start_index;
            std::vector<Observation>::const_iterator last_d = detections_full.begin() + end_index;
            std::vector<Observation> detections(first_d, last_d);

            std::vector<residuals> resid_vec(detections.size());
            std::vector<partials> partials_vec(detections.size());

            // Make these parameters flexible.
            size_t iter_max = 100;
            double eps = 1e-12;

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov;
            int flag;

            p1.x = res.value()[0].state[0];
            p1.y = res.value()[0].state[1];
            p1.z = res.value()[0].state[2];
            p1.vx = res.value()[0].state[3];
            p1.vy = res.value()[0].state[4];
            p1.vz = res.value()[0].state[5];

            flag = orbit_fit(
                ephem,
                p1,
                res.value()[0].epoch,
                detections,
                resid_vec,
                partials_vec,
                iters,
                chi2_final,
                cov,
                eps,
                iter_max);

            dof = 2 * detections.size() - 6;
	    /*
            if (flag == 0)
            {
                printf("flag: %d iters: %lu dof: %lu chi2: %lf %lu\n", flag, iters, dof, chi2_final, i);
                print_initial_condition(p1, res.value()[0].epoch);
            }
            else
            {
                printf("flag: %d iters: %lu, stage 2 failed.  Try another triplet %lu\n", flag, iters, i);
                continue;
            }
	    */

            Eigen::MatrixXd obs_cov(6, 6);
            PredictResult pred_first = predict(
                ephem,
                p1,
                res.value()[0].epoch,
                detections_full[0],
                cov,
                obs_cov);

            size_t last = detections_full.size() - 1;
            PredictResult pred_last = predict(
                ephem,
                p1,
                res.value()[0].epoch,
                detections_full[last],
                cov,
                obs_cov);

            num = detections_full.size();

            first_d = detections_full.begin() + detections_full.size() - num;
            last_d = detections_full.end();
            std::vector<Observation> detections2(first_d, last_d);

            std::vector<residuals> resid_vec2(detections2.size());
            std::vector<partials> partials_vec2(detections2.size());

            dof = 2 * detections2.size() - 6;

            flag = orbit_fit(
                ephem,
                p1,
                res.value()[0].epoch,
                detections2,
                resid_vec2,
                partials_vec2,
                iters,
                chi2_final,
                final_cov,
                eps,
                iter_max);

            if (flag == 0)
            {
                printf("flag: %d iters: %lu dof: %lu chi2: %lf\n", flag, iters, dof, chi2_final);
                Eigen::MatrixXd obs_final_cov(6, 6);
                result_epoch = res.value()[0].epoch;
                PredictResult pred_first2 = predict(
                    ephem,
                    p1,
                    res.value()[0].epoch,
                    detections_full[0],
                    final_cov,
                    obs_final_cov);

                last = detections_full.size() - 1;
                PredictResult pred_last2 = predict(
                    ephem,
                    p1,
                    res.value()[0].epoch,
                    detections_full[last],
                    final_cov,
                    obs_final_cov);
                success = 0;
                break;
            }
            else
            {
                // At this point there is a good gauss solution and a good
                // short interval solution, but the full data set is still failing.
                // That probably means that the data set should be gradually enlarged.
                printf("flag: %d iters: %lu, stage 3 failed.  Try another triplet %lu\n", flag, iters, i);
                continue;
            }
        }
        if (success != 0)
        {
            printf("fully failed\n");
            fflush(stdout);
        }

        FitResult result;

        result.method = "orbit_fit";
        result.flag = success;

        result.epoch = result_epoch;
        result.csq = chi2_final;
        result.ndof = dof;
        result.niter = iters;
        result.state[0] = p1.x;
        result.state[1] = p1.y;
        result.state[2] = p1.z;
        result.state[3] = p1.vx;
        result.state[4] = p1.vy;
        result.state[5] = p1.vz;

        if (success == 0) {
            // Populate our covariance matrix
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    // flatten the covariance matrix
                    result.cov[(i * 6) + j] = final_cov(i, j);
                }
            }
        }
        // Important issues:
        // 1. Obtaining reliable initial orbit determination for the nonlinear fits.
        // 2. Making sure the weight matrix is as good as it can be
        // 3. Identifying and removing outliers

        // Later issues:
        // 1. Deflection of light

        return result;
    }

    std::optional<FitResult> run_from_vector_with_initial_guess(struct assist_ephem *ephem, FitResult initial_guess, std::vector<Observation> &detections)
    {
        int success = 1;
        size_t iters;
        double chi2_final;
        size_t dof;

        std::vector<residuals> resid_vec(detections.size());
        std::vector<partials> partials_vec(detections.size());

	printf("run_from_vector_with_initial_guess %lu\n", detections.size());	

        // Make these parameters flexible.
        size_t iter_max = 100;
        double eps = 1e-12;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov;
        int flag;

        reb_particle p1;

        p1.x = initial_guess.state[0];
        p1.y = initial_guess.state[1];
        p1.z = initial_guess.state[2];
        p1.vx = initial_guess.state[3];
        p1.vy = initial_guess.state[4];
        p1.vz = initial_guess.state[5];
        double epoch = initial_guess.epoch;

        flag = orbit_fit(
            ephem,
            p1,
            epoch,
            detections,
            resid_vec,
            partials_vec,
            iters,
            chi2_final,
            cov,
            eps,
            iter_max);

        dof = 2 * detections.size() - 6;

        if (flag == 0)
        {

            Eigen::MatrixXd obs_cov(2, 2);
            predict(
                ephem,
                p1,
                epoch,
                detections[0],
                cov,
                obs_cov);
            std::cout << "obs_cov: \n"
                      << obs_cov << std::endl;

            size_t last = detections.size() - 1;
            predict(
                ephem,
                p1,
                epoch,
                detections[last],
                cov,
                obs_cov);
            std::cout << "obs_cov: \n"
                      << obs_cov << std::endl;

            FitResult result;

            result.niter = iters;
            result.csq = chi2_final;
            result.ndof = dof;
            result.flag = flag;
            result.method = "orbit_fit";

            result.epoch = epoch;
            result.state = {p1.x, p1.y, p1.z, p1.vx, p1.vy, p1.vz};

            for(int i = 0; i < 6; i++)
            {
                for(int j = 0; j < 6; j++)
                {
                    // flatten the covariance matrix
                    result.cov[(i * 6) + j] = cov(i,j);
                }
            }

            return result;
        }
        else
        {
            return std::nullopt;
        }
    }

    std::pair<size_t, size_t> adjust_limits(size_t curr_num, size_t num, size_t dfs, size_t id0, size_t id1, size_t id2){
	    
	size_t start_index;
	size_t end_index;
	if (curr_num >= num)
            {
                // We already have enough data and
                // don't need to move the limits
                start_index = id0;
                end_index = id1;
            }
            else if ((dfs - id0) >= num)
		{
		    // There are enough points if we include
		    // only the later points, we only need to
		    // later limit.
		    start_index = id0;
		    end_index = id0 + num;
		}
            else if (dfs >= num)
		{
		    end_index = dfs;
		    start_index = dfs - num;
            }
            else
		{
		    start_index = 0;
		    end_index = dfs;
		}
	return std::make_pair(start_index, end_index);
    }

    std::vector<Observation> select_detections(std::vector<Observation>& detections_full, size_t start_index, size_t end_index){
	std::vector<Observation>::const_iterator first_d = detections_full.begin() + start_index;
	std::vector<Observation>::const_iterator last_d = detections_full.begin() + end_index;
	std::vector<Observation> selected_detections(first_d, last_d);
	return selected_detections;
    }
    
    FitResult run_from_vector(struct assist_ephem *ephem, std::vector<Observation> &detections_full)
    {

	size_t start_i = detections_full.size() - 1;

	std::string objID = detections_full[0].objID;

        // First find a triple of detections in the full data set.
        std::vector<std::vector<size_t>> idx = IOD_indices(detections_full, 2.0, 10.0, 2.0, 20.0, 10, start_i);
        //std::vector<std::vector<size_t>> idx = IOD_indices_forward(detections_full, 2.0, 10.0, 2.0, 20.0, 10, start_i);	
        //std::vector<std::vector<size_t>> idx_bwd = IOD_indices(detections_full, 2.0, 10.0, 2.0, 20.0, 10, end_i);
	//idx.insert(idx.end(), idx_bwd.begin(), idx_bwd.end());
	
        int success = 1;
        size_t iters;
        double chi2_final;
        double result_epoch;
        size_t dof;
        struct reb_particle p1;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> final_cov;
        size_t iter_max = 100;
        double eps = 1e-12;

        for (size_t i = 0; i < idx.size(); i++)
        {

            std::vector<size_t> indices = idx[i];
            size_t id0 = indices[0];
            size_t id1 = indices[1];
            size_t id2 = indices[2];

            printf("gauss indices: %lu %lu %lu\n", id0, id1, id2);
            fflush(stdout);

            Observation d0 = detections_full[id0];
            Observation d1 = detections_full[id1];
            Observation d2 = detections_full[id2];

            // Put this in a better place
            double GMtotal = 0.00029630927487993194;

            std::optional<std::vector<FitResult>> res = gauss(GMtotal, d0, d1, d2, 0.0001, SPEED_OF_LIGHT);
            if (!res.has_value())
            {
                printf("gauss failed.  Try another triplet.\n");
                fflush(stdout);
                continue;
            }

	    printf("gauss rho: %le\n", res.value()[0].root);

	    // This bit is for testing TNOs.
	    double inner_root_thresh = 0.1;
	    double outer_root_thresh = 1000.;	    
	    if(res.value()[0].root>outer_root_thresh){
		printf("Too far!\n");
		continue;
	    }

	    if(res.value()[0].root<=inner_root_thresh){
		printf("Too close!\n");
		continue;
	    }
	    
	    // Now get a segment of data that spans the triplet and that uses up to
            // a certain number of points, if they are available in a time window.
            // This should be a parameter
            size_t num = 1000;

	    size_t dfs = detections_full.size();
            size_t curr_num = id2 - id1 + 1;

	    std::pair<size_t, size_t> limits;

	    FitResult guess = res.value()[0];
	    std::optional<FitResult> fit_res;

	    limits = adjust_limits(curr_num, num, dfs, id0, id1, id2);
            size_t start_index = limits.first;
            size_t end_index = limits.second;

	    std::vector<Observation> detections = select_detections(detections_full, start_index, end_index);
	    fit_res = run_from_vector_with_initial_guess(ephem, guess, detections);
            if (!fit_res.has_value()){
		continue;
	    }

	    guess = fit_res.value();

	    curr_num = num;
            num = 50;
	    limits = adjust_limits(curr_num, num, dfs, id0, id1, id2);
	    start_index = limits.first;
            end_index = limits.second;

	    printf("limits %lu %lu\n", start_index, end_index);

	    std::vector<Observation> detections2 = select_detections(detections_full, start_index, end_index);

	    fit_res = run_from_vector_with_initial_guess(ephem, guess, detections2);
            if (!fit_res.has_value()){
		continue;
	    }

	    guess = fit_res.value();

	    curr_num = num;
            num = dfs;
	    limits = adjust_limits(curr_num, num, dfs, id0, id1, id2);
	    start_index = limits.first;
            end_index = limits.second;

	    std::vector<Observation> detections3 = select_detections(detections_full, start_index, end_index);

	    fit_res = run_from_vector_with_initial_guess(ephem, guess, detections3);
            if (!fit_res.has_value()){
		continue;
	    }
	    
	    FitResult sfr = fit_res.value();

	    int flag = sfr.flag;
	    
            if (flag != 0)
            {
                // At this point there is a good gauss solution and a good
                // short interval solution, but the full data set is still failing.
                // That probably means that the data set should be gradually enlarged.
                printf("flag: %d iters: %lu, stage 3 failed.  Try another triplet %lu\n", flag, sfr.niter, i);
                continue;
            }else{
		printf("SUCCESS\n\n");
		success = 0;
		break;
	    }
        }
        if (success != 0)
        {
            printf("fully failed\n\n");
            fflush(stdout);
        }

        FitResult result;

        result.method = "orbit_fit";
        result.flag = success;

        result.epoch = result_epoch;
        result.csq = chi2_final;
        result.ndof = dof;
        result.niter = iters;
        result.state[0] = p1.x;
        result.state[1] = p1.y;
        result.state[2] = p1.z;
        result.state[3] = p1.vx;
        result.state[4] = p1.vy;
        result.state[5] = p1.vz;

        if (success == 0) {
            // Populate our covariance matrix
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    // flatten the covariance matrix
                    result.cov[(i * 6) + j] = final_cov(i, j);
                }
            }
        }
        // Important issues:
        // 1. Obtaining reliable initial orbit determination for the nonlinear fits.
        // 2. Making sure the weight matrix is as good as it can be
        // 3. Identifying and removing outliers

        // Later issues:
        // 1. Deflection of light

        return result;
    }
    

#ifdef Py_PYTHON_H
    static void orbit_fit_bindings(py::module &m)
    {
        py::class_<assist_ephem>(m, "assist_ephem");
        m.def("orbit_fit", &orbit_fit::orbit_fit, R"pbdoc(Core orbit fit function.)pbdoc");
        m.def("get_ephem", &orbit_fit::get_ephem, R"pbdoc(get ephemeris)pbdoc");
        m.def("run_from_vector", &orbit_fit::run_from_vector, R"pbdoc(
                Takes an assist_ephem object and a vector of observations and runs orbit fit.
            )pbdoc");
        m.def("run_from_vector_with_initial_guess", &orbit_fit::run_from_vector_with_initial_guess, R"pbdoc(
                Takes an assist_ephem object, a vector of observations and an initial guess
                and runs orbit fit.
            )pbdoc");
    }
#endif /* Py_PYTHON_H */

} // namespace: orbit_fit
