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

// Compile with something like this
// g++ -std=c++17 -I../..//src/ -I../../../rebound/src/ -I/Users/mholman/eigen-3.4.0 -Wl,-rpath,./ -Wpointer-arith -D_GNU_SOURCE -O3 -fPIC -I/usr/local/include -Wall -g  -Wno-unknown-pragmas -D_APPLE -DSERVER -DGITHASH=e99d8d73f0aa7fb7bf150c680a2d581f43d3a8be gauss.cpp -L. -lassist -lrebound -L/usr/local/lib -o gauss

#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>

#include <chrono>

#include "orbit_fit.h"
#include "gauss.h"

using namespace Eigen;
using std::cout;

// template for gauss
// pass in three detections
std::optional<std::vector<gauss_soln>> gauss(double MU_BARY, detection &o1_in, detection &o2_in, detection &o3_in, double min_distance, double SPEED_OF_LIGHT) {
    // Create a vector of pointers to observations for sorting by epoch
    std::vector<detection> triplet = { o1_in, o2_in, o3_in };
    std::sort(triplet.begin(), triplet.end(), [](const detection a, const detection b) {
        return a.jd_tdb < b.jd_tdb;
    });

    // Get the observer positions and pointing directions
    Eigen::Vector3d o1 = {triplet[0].xe, triplet[0].ye, triplet[0].ze};
    Eigen::Vector3d o2 = {triplet[1].xe, triplet[1].ye, triplet[1].ze};
    Eigen::Vector3d o3 = {triplet[2].xe, triplet[2].ye, triplet[2].ze};        

    Eigen::Vector3d rho1 = {triplet[0].theta_x, triplet[0].theta_y, triplet[0].theta_z};
    Eigen::Vector3d rho2 = {triplet[1].theta_x, triplet[1].theta_y, triplet[1].theta_z};
    Eigen::Vector3d rho3 = {triplet[2].theta_x, triplet[2].theta_y, triplet[2].theta_z};        

    double t1 = triplet[0].jd_tdb;
    double t2 = triplet[1].jd_tdb;
    double t3 = triplet[2].jd_tdb;

    double tau1 = t1 - t2;
    double tau3 = t3 - t2;
    double tau  = t3 - t1;

    // Compute cross products
    Eigen::Vector3d p1 = rho2.cross(rho3);
    Eigen::Vector3d p2 = rho1.cross(rho3);
    Eigen::Vector3d p3 = rho1.cross(rho2);

    double d0 = rho1.dot(p1);

    // Construct the 3x3 d matrix
    Eigen::Matrix3d d;
    d(0, 0) = o1.dot(p1); d(0, 1) = o1.dot(p2); d(0, 2) = o1.dot(p3);
    d(1, 0) = o2.dot(p1); d(1, 1) = o2.dot(p2); d(1, 2) = o2.dot(p3);
    d(2, 0) = o3.dot(p1); d(2, 1) = o3.dot(p2); d(2, 2) = o3.dot(p3);

    double a = (1.0 / d0) * (-d(0, 1) * (tau3 / tau) + d(1, 1) + d(2, 1) * (tau1 / tau));
    double b = (1.0 / (6.0 * d0)) * (d(0, 1) * ((tau3 * tau3) - (tau * tau)) * (tau3 / tau) +
                                     d(2, 1) * ((tau * tau) - (tau1 * tau1)) * (tau1 / tau));
    double e = o2.dot(rho2);
    double o2sq = o2.dot(o2);

    double aa = -((a * a) + 2.0 * a * e + o2sq);
    double bb = -2.0 * MU_BARY * b * (a + e);
    double cc = -std::pow(MU_BARY, 2) * (b * b);

    // Construct the 8x8 companion matrix for the polynomial
    Eigen::Matrix<double, 8, 8> mat = Eigen::Matrix<double, 8, 8>::Zero();
    mat(0, 1) = 1.0;
    mat(1, 2) = 1.0;
    mat(2, 3) = 1.0;
    mat(3, 4) = 1.0;
    mat(4, 5) = 1.0;
    mat(5, 6) = 1.0;
    mat(6, 7) = 1.0;
    mat(7, 0) = -cc;
    mat(7, 3) = -bb;
    mat(7, 6) = -aa;

    // Compute the eigenvalues of the companion matrix
    Eigen::ComplexEigenSolver<Eigen::Matrix<double, 8, 8>> ces(mat);
    if (ces.info() != Eigen::Success) {
        return std::nullopt;
    }

    // Filter eigenvalues: select those with (nearly) zero imaginary part and real part > min_distance
    std::vector<double> roots;
    for (int i = 0; i < ces.eigenvalues().size(); ++i) {
        std::complex<double> eig = ces.eigenvalues()[i];
        if (std::abs(eig.imag()) < 1e-10 && eig.real() > min_distance) {
            roots.push_back(eig.real());
        }
    }
    if (roots.empty()) {
        return std::nullopt;
    }

    std::sort(roots.begin(), roots.end(), [](const double a, const double b) {
        return a>b;
    });
    

    std::vector<gauss_soln> res;
    for (double root : roots) {
	std::cout << root << std::endl;
        double root3 = std::pow(root, 3);

        // Compute a1
        double num1 = 6.0 * (d(2, 0) * (tau1 / tau3) + d(1, 0) * (tau / tau3)) * root3 +
                      MU_BARY * d(2, 0) * ((tau * tau) - (tau1 * tau1)) * (tau1 / tau3);
        double den1 = 6.0 * root3 + MU_BARY * ((tau * tau) - (tau3 * tau3));
        double a1 = (1.0 / d0) * ((num1 / den1) - d(0, 0));

        // Compute a2 and a3
        double a2 = a + (MU_BARY * b) / root3;
        double num3 = 6.0 * (d(0, 2) * (tau3 / tau1) - d(1, 2) * (tau / tau1)) * root3 +
                      MU_BARY * d(0, 2) * ((tau * tau) - (tau3 * tau3)) * (tau3 / tau1);
        double den3 = 6.0 * root3 + MU_BARY * ((tau * tau) - (tau1 * tau1));
        double a3 = (1.0 / d0) * ((num3 / den3) - d(2, 2));

        // Compute position vectors
        Eigen::Vector3d r1 = o1 + a1 * rho1;
        Eigen::Vector3d r2 = o2 + a2 * rho2;
        Eigen::Vector3d r3 = o3 + a3 * rho3;

        // Calculate f and g functions
        double f1 = 1.0 - 0.5 * (MU_BARY / root3) * (tau1 * tau1);
        double f3 = 1.0 - 0.5 * (MU_BARY / root3) * (tau3 * tau3);
        double g1 = tau1 - (1.0 / 6.0) * (MU_BARY / root3) * (tau1 * tau1 * tau1);
        double g3 = tau3 - (1.0 / 6.0) * (MU_BARY / root3) * (tau3 * tau3 * tau3);

        // Solve for the velocity at t2
        Eigen::Vector3d v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1);

        // Extract components for clarity
        double x  = r2.x();
        double y  = r2.y();
        double z  = r2.z();
        double vx = v2.x();
        double vy = v2.y();
        double vz = v2.z();

        // Apply light-time correction
        double ltt = r2.norm() / SPEED_OF_LIGHT;
        double corrected_t = triplet[1].jd_tdb;
        corrected_t -= ltt;

	gauss_soln soln = gauss_soln(root, corrected_t, x, y, z, vx, vy, vz);
	res.push_back(soln);

    }

    return res;
}


