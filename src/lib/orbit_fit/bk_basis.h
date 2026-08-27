#pragma once

#include <Eigen/Dense>
#include <vector>

namespace orbit_fit
{

    // Bernstein-Khushalani parameters at epoch.  Origin: barycenter.
    // (alpha, beta) are gnomonic tangent-plane coordinates of the
    // line-of-sight direction rho_hat at a fiducial direction n0,
    // gamma = 1/|r| with r measured from the SAME origin as the struct --
    // the barycenter (issue #444; the comment here previously said
    // r_helio, which contradicted the line above and the implementation).
    //
    // (adot, bdot, gdot) are NOT the plain time derivatives of
    // (alpha, beta, gamma).  They are the velocity components in the
    // orthonormal fiducial basis, scaled by gamma (issue #445):
    //
    //     adot = gamma (v . a),  bdot = gamma (v . b),  gdot = gamma (v . n0)
    //
    // This is the scaling that makes all three carry the same units and the
    // same 1/gamma weighting as the position, so the 6x6 Jacobian below has
    // no second-derivative terms and the energy prior collapses to its
    // familiar form.  See the derivation note accompanying issue #445.
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
    //   v_vec = (1/gamma) [adot * a + bdot * b + gdot * n0]
    // The velocity is diagonal in the fiducial basis: under the scaled
    // convention the dots ARE its components there (issue #445).
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
    // Under the scaled convention (issue #445) v depends on alpha and beta
    // only through nothing at all -- the first two columns of the lower-left
    // block are exactly zero, and the lower-right block is (1/gamma) times
    // the orthonormal fiducial basis.  The second-derivative terms that the
    // unscaled convention required are gone.
    Eigen::Matrix<double, 6, 6> dcart_dbk(
        const BKState &bk, const BKFiducial &fid);

    // Variance of the bound-orbit gdot prior:
    //   sigma_gdot^2 = 2 mu gamma^3 - adot^2 - bdot^2
    //
    // The bound-orbit condition is |v|^2 < 2 mu gamma, and under the scaled
    // convention gamma^2 |v|^2 = adot^2 + bdot^2 + gdot^2 exactly, because
    // (a, b, n0) is orthonormal.  So the bound becomes
    //   gdot^2 < 2 mu gamma^3 - adot^2 - bdot^2
    // with no dependence on (alpha, beta) at all.  The (alpha, beta)-dependent
    // tangent-vector norms that the unscaled convention carried were an
    // artifact of that convention, not physics (issue #445).
    //
    // Returns +infinity when the tangential rates already exceed escape (the
    // right-hand side would be non-positive), signalling "no prior."  The
    // caller's precision is 1 / sigma_gdot_sq, so +inf -> 0 precision ->
    // no contribution, which is the correct behavior.
    double sigma_gdot_sq(const BKState &bk, double mu);

} // namespace orbit_fit
