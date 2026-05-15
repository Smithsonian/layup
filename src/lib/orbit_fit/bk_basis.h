#pragma once

#include <Eigen/Dense>
#include <vector>

namespace orbit_fit
{

    // Bernstein-Khushalani parameters at epoch.  Origin: barycenter.
    // (alpha, beta) are gnomonic tangent-plane coordinates of the
    // line-of-sight direction rho_hat at a fiducial direction n0,
    // gamma = 1/|r_helio|, and (adot, bdot, gdot) are their time
    // derivatives at the epoch.
    struct BKState
    {
        double alpha = 0.0;
        double beta = 0.0;
        double gamma = 0.0;
        double adot = 0.0;
        double bdot = 0.0;
        double gdot = 0.0;
    };

    // Orthonormal frame defining the BK gnomonic tangent plane.
    // {a, b, n0} form a right-handed orthonormal basis; n0 is the
    // fiducial line-of-sight, (a, b) span its tangent plane.
    struct BKFiducial
    {
        Eigen::Vector3d n0;
        Eigen::Vector3d a;
        Eigen::Vector3d b;
    };

    // Choose a fiducial frame from a list of line-of-sight unit vectors.
    // n0 := normalize(sum(rho_hats)); (a, b) constructed by Gram-Schmidt
    // against ICRS z (or ICRS x if n0 is near the z-axis).  This is one
    // of many valid choices -- the BK fit is gauge-invariant under any
    // rotation of (a, b) about n0.
    BKFiducial choose_fiducial(const std::vector<Eigen::Vector3d> &rho_hats);

    // Forward transform: BK -> barycentric Cartesian (position + velocity).
    //   r_vec = (1/gamma) * rho_hat(alpha, beta)
    //   v_vec = (1/gamma) [adot * rho_hat_alpha + bdot * rho_hat_beta]
    //           - (gdot/gamma^2) * rho_hat(alpha, beta)
    Eigen::Matrix<double, 6, 1> bk_to_cartesian(
        const BKState &bk, const BKFiducial &fid);

    // Inverse transform: barycentric Cartesian -> BK.  Well-defined for
    // any state with r_vec . n0 > 0 (object on the n0-facing hemisphere)
    // and gamma > 0.
    BKState cartesian_to_bk(
        const Eigen::Matrix<double, 6, 1> &cart, const BKFiducial &fid);

    // 6x6 Jacobian d(r_vec, v_vec) / d(alpha, beta, gamma, adot, bdot, gdot).
    // Block structure (each block is 3x3):
    //   [ d r / d (alpha,beta,gamma)     0                              ]
    //   [ d v / d (alpha,beta,gamma)     d v / d (adot,bdot,gdot)       ]
    // Top-left and bottom-right blocks are identical (both arise from the
    // (1/gamma) * tangent-vector structure).
    Eigen::Matrix<double, 6, 6> dcart_dbk(
        const BKState &bk, const BKFiducial &fid);

    // Variance of the bound-orbit gdot prior:
    //   sigma_gdot^2 = gamma^2 * (2 * mu * gamma^3 - adot^2 - bdot^2)
    // Returns +infinity when the tangential rates already exceed escape
    // (the right-hand side would be non-positive), signalling "no prior."
    // The caller's precision is 1 / sigma_gdot_sq, so +inf -> 0 precision
    // -> no contribution, which is the correct behavior.
    double sigma_gdot_sq(const BKState &bk, double mu);

} // namespace orbit_fit
