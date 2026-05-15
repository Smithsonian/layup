// Bernstein-Khushalani parameterization primitives for the layup-internal
// universal BK fitter (feat/bk-everywhere).
//
// The math layer is pure C++/Eigen -- no ASSIST or REBOUND dependencies --
// so this translation unit can be reasoned about and tested in isolation
// of the dynamics path.  pybind11 bindings at the bottom expose the
// primitives to Python so Layer 1 tests (round-trip, finite-difference
// Jacobian, mixed-partial symmetry, etc.) can run via pytest.  The design
// and math derivation live in the project memory file
// bk_everywhere_design.md.

#include "bk_basis.h"

#include <cmath>
#include <limits>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace orbit_fit
{

    namespace
    {
        // Internal cached quantities at the BK position (alpha, beta).
        //
        //   p = n0 + alpha*a + beta*b
        //   s_sq = 1 + alpha^2 + beta^2 = |p|^2
        //   rho_hat = p / sqrt(s_sq)
        //   rho_hat_alpha = (a - (a . rho_hat) * rho_hat) / sqrt(s_sq)
        //   rho_hat_beta  = (b - (b . rho_hat) * rho_hat) / sqrt(s_sq)
        //
        // rho_hat_alpha and rho_hat_beta are the gnomonic-projection tangent
        // vectors at rho_hat; they are NOT unit length in general (they
        // scale as 1/sqrt(s_sq) times the projection of a/b onto T_{rho_hat}).
        struct RhoFrame
        {
            double s_sq;
            double s;
            Eigen::Vector3d rho_hat;
            Eigen::Vector3d rho_hat_alpha;
            Eigen::Vector3d rho_hat_beta;
        };

        RhoFrame compute_rho_frame(double alpha, double beta, const BKFiducial &fid)
        {
            RhoFrame f;
            f.s_sq = 1.0 + alpha * alpha + beta * beta;
            f.s = std::sqrt(f.s_sq);
            const Eigen::Vector3d p = fid.n0 + alpha * fid.a + beta * fid.b;
            f.rho_hat = p / f.s;
            const double rho_dot_a = f.rho_hat.dot(fid.a);
            const double rho_dot_b = f.rho_hat.dot(fid.b);
            f.rho_hat_alpha = (fid.a - rho_dot_a * f.rho_hat) / f.s;
            f.rho_hat_beta = (fid.b - rho_dot_b * f.rho_hat) / f.s;
            return f;
        }
    } // namespace

    BKFiducial choose_fiducial(const std::vector<Eigen::Vector3d> &rho_hats)
    {
        BKFiducial fid;
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        for (const auto &r : rho_hats)
            mean += r;
        if (mean.norm() < 1e-12)
        {
            // Pathological: observation directions cancel out.  Pick ICRS x
            // as a fallback; the fit is gauge-invariant under fiducial choice
            // anyway, so any nonzero direction works.
            mean = Eigen::Vector3d::UnitX();
        }
        fid.n0 = mean.normalized();

        // Gram-Schmidt against the ICRS axis least parallel to n0 so we
        // don't divide by something tiny.
        const Eigen::Vector3d seed = std::abs(fid.n0.z()) < 0.9
                                         ? Eigen::Vector3d::UnitZ()
                                         : Eigen::Vector3d::UnitX();
        fid.a = (seed - seed.dot(fid.n0) * fid.n0).normalized();
        fid.b = fid.n0.cross(fid.a);
        return fid;
    }

    Eigen::Matrix<double, 6, 1> bk_to_cartesian(
        const BKState &bk, const BKFiducial &fid)
    {
        const RhoFrame f = compute_rho_frame(bk.alpha, bk.beta, fid);
        const double inv_g = 1.0 / bk.gamma;
        const double inv_g2 = inv_g * inv_g;

        const Eigen::Vector3d r = inv_g * f.rho_hat;
        const Eigen::Vector3d v = inv_g * (bk.adot * f.rho_hat_alpha + bk.bdot * f.rho_hat_beta)
                                  - bk.gdot * inv_g2 * f.rho_hat;

        Eigen::Matrix<double, 6, 1> cart;
        cart << r, v;
        return cart;
    }

    BKState cartesian_to_bk(
        const Eigen::Matrix<double, 6, 1> &cart, const BKFiducial &fid)
    {
        const Eigen::Vector3d r = cart.head<3>();
        const Eigen::Vector3d v = cart.tail<3>();

        const double r_norm = r.norm();
        const double gamma = 1.0 / r_norm;
        const Eigen::Vector3d rho_hat = gamma * r;

        // Gnomonic tangent-plane coordinates of rho_hat at n0.
        const double u = rho_hat.dot(fid.n0);
        const double alpha = rho_hat.dot(fid.a) / u;
        const double beta = rho_hat.dot(fid.b) / u;

        // gdot = d/dt (1/|r|) = -(r . v) / |r|^3 = -gamma^2 * (rho_hat . v)
        const double gdot = -gamma * gamma * rho_hat.dot(v);

        // d(rho_hat)/dt = gamma * (v - (v . rho_hat) * rho_hat)  -- v's component perp to rho_hat,
        // scaled to a sphere tangent vector.  alpha-dot, beta-dot come from
        // applying the quotient rule to alpha = (rho_hat . a) / (rho_hat . n0)
        // and similarly for beta.
        const Eigen::Vector3d rho_dot = gamma * (v - v.dot(rho_hat) * rho_hat);
        const double rho_dot_n0 = rho_dot.dot(fid.n0);
        const double adot = (rho_dot.dot(fid.a) - alpha * rho_dot_n0) / u;
        const double bdot = (rho_dot.dot(fid.b) - beta * rho_dot_n0) / u;

        BKState bk;
        bk.alpha = alpha;
        bk.beta = beta;
        bk.gamma = gamma;
        bk.adot = adot;
        bk.bdot = bdot;
        bk.gdot = gdot;
        return bk;
    }

    Eigen::Matrix<double, 6, 6> dcart_dbk(
        const BKState &bk, const BKFiducial &fid)
    {
        const double alpha = bk.alpha;
        const double beta = bk.beta;
        const double gamma = bk.gamma;
        const double adot = bk.adot;
        const double bdot = bk.bdot;
        const double gdot = bk.gdot;

        const RhoFrame f = compute_rho_frame(alpha, beta, fid);
        const double inv_g = 1.0 / gamma;
        const double inv_g2 = inv_g * inv_g;
        const double inv_g3 = inv_g2 * inv_g;
        const double inv_s2 = 1.0 / f.s_sq;
        const double inv_s4 = inv_s2 * inv_s2;

        Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();

        // Top-left and bottom-right 3x3 blocks: d(r)/d(alpha,beta,gamma) and
        // d(v)/d(adot,bdot,gdot) -- identical shape.
        const Eigen::Vector3d dr_dalpha = inv_g * f.rho_hat_alpha;
        const Eigen::Vector3d dr_dbeta = inv_g * f.rho_hat_beta;
        const Eigen::Vector3d dr_dgamma = -inv_g2 * f.rho_hat;

        J.block<3, 1>(0, 0) = dr_dalpha;
        J.block<3, 1>(0, 1) = dr_dbeta;
        J.block<3, 1>(0, 2) = dr_dgamma;

        J.block<3, 1>(3, 3) = dr_dalpha;
        J.block<3, 1>(3, 4) = dr_dbeta;
        J.block<3, 1>(3, 5) = dr_dgamma;

        // Bottom-left 3x3 block: d(v)/d(alpha,beta,gamma).  Needs second
        // derivatives of rho_hat with respect to (alpha, beta):
        //   d rho_hat_alpha / d alpha = -(1+beta^2)/s^4 * rho_hat
        //                                - 2 alpha / s^2 * rho_hat_alpha
        //   d rho_hat_alpha / d beta  = (alpha*beta)/s^4 * rho_hat
        //                                - alpha/s^2 * rho_hat_beta
        //                                - beta/s^2 * rho_hat_alpha
        //   d rho_hat_beta  / d beta  = -(1+alpha^2)/s^4 * rho_hat
        //                                - 2 beta / s^2 * rho_hat_beta
        // and d rho_hat_beta / d alpha == d rho_hat_alpha / d beta by mixed-partial symmetry.
        const Eigen::Vector3d d_rha_dalpha = -(1.0 + beta * beta) * inv_s4 * f.rho_hat
                                             - 2.0 * alpha * inv_s2 * f.rho_hat_alpha;
        const Eigen::Vector3d d_rha_dbeta = (alpha * beta) * inv_s4 * f.rho_hat
                                            - alpha * inv_s2 * f.rho_hat_beta
                                            - beta * inv_s2 * f.rho_hat_alpha;
        const Eigen::Vector3d d_rhb_dbeta = -(1.0 + alpha * alpha) * inv_s4 * f.rho_hat
                                            - 2.0 * beta * inv_s2 * f.rho_hat_beta;

        // d v / d alpha
        const Eigen::Vector3d dv_dalpha = inv_g * (adot * d_rha_dalpha + bdot * d_rha_dbeta)
                                          - gdot * inv_g2 * f.rho_hat_alpha;
        // d v / d beta
        const Eigen::Vector3d dv_dbeta = inv_g * (adot * d_rha_dbeta + bdot * d_rhb_dbeta)
                                         - gdot * inv_g2 * f.rho_hat_beta;
        // d v / d gamma
        const Eigen::Vector3d dv_dgamma = -inv_g2 * (adot * f.rho_hat_alpha + bdot * f.rho_hat_beta)
                                          + 2.0 * gdot * inv_g3 * f.rho_hat;

        J.block<3, 1>(3, 0) = dv_dalpha;
        J.block<3, 1>(3, 1) = dv_dbeta;
        J.block<3, 1>(3, 2) = dv_dgamma;

        return J;
    }

    double sigma_gdot_sq(const BKState &bk, double mu)
    {
        // Bound-orbit constraint: 0.5 |v|^2 <= mu / |r| = mu * gamma.
        // Substituting |v|^2 = gdot^2 / gamma^4 + |adot rho_hat_alpha + bdot rho_hat_beta|^2 / gamma^2
        // (cross-terms with rho_hat vanish since rho_hat_alpha, rho_hat_beta are tangent to rho_hat),
        //
        //   gdot^2 <= gamma^2 * (2 mu gamma^3 - |adot rho_hat_alpha + bdot rho_hat_beta|^2)
        //
        // The tangential term expands via
        //   |rho_hat_alpha|^2 = (1+beta^2)/s^4,  |rho_hat_beta|^2 = (1+alpha^2)/s^4,
        //   rho_hat_alpha . rho_hat_beta       = -alpha*beta/s^4,
        // so
        //   |adot rho_hat_alpha + bdot rho_hat_beta|^2 =
        //       [adot^2 (1+beta^2) - 2 adot bdot alpha beta + bdot^2 (1+alpha^2)] / s^4 .
        // At alpha = beta = 0 (the fiducial direction) this reduces to adot^2 + bdot^2.
        const double alpha = bk.alpha;
        const double beta = bk.beta;
        const double gamma = bk.gamma;
        const double adot = bk.adot;
        const double bdot = bk.bdot;
        const double s_sq = 1.0 + alpha * alpha + beta * beta;
        const double s4 = s_sq * s_sq;
        const double v_tan_sq = (adot * adot * (1.0 + beta * beta)
                                 - 2.0 * adot * bdot * alpha * beta
                                 + bdot * bdot * (1.0 + alpha * alpha)) / s4;
        const double rhs = 2.0 * mu * gamma * gamma * gamma - v_tan_sq;
        if (rhs <= 0.0)
        {
            // Tangential rates already exceed escape velocity for this
            // gamma; the energy bound provides no constraint on gdot.
            return std::numeric_limits<double>::infinity();
        }
        return gamma * gamma * rhs;
    }

    static void bk_basis_bindings(pybind11::module &m)
    {
        namespace py = pybind11;

        py::class_<BKState>(m, "BKState")
            .def(py::init<>())
            .def(py::init([](double alpha, double beta, double gamma,
                             double adot, double bdot, double gdot)
                          {
                BKState bk;
                bk.alpha = alpha; bk.beta = beta; bk.gamma = gamma;
                bk.adot = adot;   bk.bdot = bdot; bk.gdot = gdot;
                return bk; }),
                 py::arg("alpha") = 0.0, py::arg("beta") = 0.0, py::arg("gamma") = 0.0,
                 py::arg("adot") = 0.0, py::arg("bdot") = 0.0, py::arg("gdot") = 0.0)
            .def_readwrite("alpha", &BKState::alpha)
            .def_readwrite("beta", &BKState::beta)
            .def_readwrite("gamma", &BKState::gamma)
            .def_readwrite("adot", &BKState::adot)
            .def_readwrite("bdot", &BKState::bdot)
            .def_readwrite("gdot", &BKState::gdot)
            .def("__repr__", [](const BKState &b)
                 {
                std::ostringstream s;
                s << "<BKState alpha=" << b.alpha << " beta=" << b.beta
                  << " gamma=" << b.gamma << " adot=" << b.adot
                  << " bdot=" << b.bdot << " gdot=" << b.gdot << ">";
                return s.str(); });

        py::class_<BKFiducial>(m, "BKFiducial")
            .def(py::init<>())
            .def_readwrite("n0", &BKFiducial::n0)
            .def_readwrite("a", &BKFiducial::a)
            .def_readwrite("b", &BKFiducial::b);

        m.def("bk_choose_fiducial", &choose_fiducial, py::arg("rho_hats"),
              "Construct a BKFiducial frame from a list of unit line-of-sight vectors.");
        m.def("bk_to_cartesian", &bk_to_cartesian,
              py::arg("bk"), py::arg("fid"),
              "Forward transform: BK state -> 6-vector of barycentric Cartesian (r, v).");
        m.def("cartesian_to_bk", &cartesian_to_bk,
              py::arg("cart"), py::arg("fid"),
              "Inverse transform: 6-vector barycentric Cartesian -> BK state.");
        m.def("dcart_dbk", &dcart_dbk,
              py::arg("bk"), py::arg("fid"),
              "6x6 Jacobian d(r, v) / d(alpha, beta, gamma, adot, bdot, gdot).");
        m.def("sigma_gdot_sq", &sigma_gdot_sq,
              py::arg("bk"), py::arg("mu"),
              "Variance of the bound-orbit energy prior on gdot.");
    }

} // namespace orbit_fit
