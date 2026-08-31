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
        // p, n0, a, b, rho_hat, rho_hat_alpha and rho_hat_beta are all
        // 3-vectors in the ICRS frame.  (n0, a, b) is the right-handed
        // orthonormal fiducial basis (struct BKFiducial in bk_basis.h):
        // n0 is the fiducial line-of-sight direction and (a, b) span its
        // tangent plane.  (alpha, beta) are the gnomonic tangent-plane
        // coordinates of the object's line-of-sight direction in that basis.
        //
        //   p             = n0 + alpha*a + beta*b     (unnormalized LOS dir)
        //   s_sq          = 1 + alpha^2 + beta^2 = |p|^2
        //   rho_hat       = p / sqrt(s_sq)            (unit LOS direction)
        //   rho_hat_alpha = (a - (a . rho_hat) * rho_hat) / sqrt(s_sq)
        //   rho_hat_beta  = (b - (b . rho_hat) * rho_hat) / sqrt(s_sq)
        //
        // rho_hat_alpha and rho_hat_beta are d(rho_hat)/d(alpha) and
        // d(rho_hat)/d(beta) -- the gnomonic-projection tangent vectors at
        // rho_hat.  They are NOT unit length in general (they scale as
        // 1/sqrt(s_sq) times the projection of a/b onto T_{rho_hat}).
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

        const Eigen::Vector3d r = inv_g * f.rho_hat;

        // Under the 1/gamma-scaled convention (issue #445) the dots ARE the
        // velocity components in the orthonormal fiducial basis, scaled by gamma:
        //   adot = gamma (v . a),  bdot = gamma (v . b),  gdot = gamma (v . n0)
        // so the inverse is three terms in an orthonormal basis:
        //   v = (1/gamma) (adot a + bdot b + gdot n0)
        //
        // The gnomonic tangent vectors rho_hat_alpha, rho_hat_beta -- which are
        // deliberately not unit length and were a recurring source of confusion --
        // drop out of the velocity path entirely.  All three dots now share units
        // of inverse time.
        const Eigen::Vector3d v = inv_g * (bk.adot * fid.a + bk.bdot * fid.b + bk.gdot * fid.n0);

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

        // Under the 1/gamma-scaled convention (issue #445) the dots are simply the
        // velocity components in the orthonormal fiducial basis, scaled by gamma.
        // No quotient rule, no tangent-vector norms.
        const double adot = gamma * v.dot(fid.a);
        const double bdot = gamma * v.dot(fid.b);
        const double gdot = gamma * v.dot(fid.n0);

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

        Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();

        // Position rows are unchanged by the convention change: r = rho_hat / gamma
        // does not involve the dots.
        J.block<3, 1>(0, 0) = inv_g * f.rho_hat_alpha;   // d r / d alpha
        J.block<3, 1>(0, 1) = inv_g * f.rho_hat_beta;    // d r / d beta
        J.block<3, 1>(0, 2) = -inv_g2 * f.rho_hat;       // d r / d gamma

        // Velocity rows.  Under the 1/gamma-scaled convention (issue #445),
        //   v = (1/gamma) (adot a + bdot b + gdot n0)
        // so v does NOT depend on alpha or beta at all, and the whole
        // second-derivative block that used to live here is gone.
        //
        //   d v / d alpha = d v / d beta = 0
        //   d v / d gamma = -(1/gamma^2) (adot a + bdot b + gdot n0) = -v / gamma
        //   d v / d (adot, bdot, gdot) = (1/gamma) [a b n0]
        const Eigen::Vector3d v_dir = adot * fid.a + bdot * fid.b + gdot * fid.n0;
        J.block<3, 1>(3, 2) = -inv_g2 * v_dir;           // d v / d gamma
        J.block<3, 1>(3, 3) = inv_g * fid.a;             // d v / d adot
        J.block<3, 1>(3, 4) = inv_g * fid.b;             // d v / d bdot
        J.block<3, 1>(3, 5) = inv_g * fid.n0;            // d v / d gdot

        return J;
    }

    double sigma_gdot_sq(const BKState &bk, double mu)
    {
        // Bound-orbit constraint: |v|^2 < 2 mu / |r|.
        //
        // Under the 1/gamma-scaled convention (issue #445) all three dots are
        // velocity components in an ORTHONORMAL basis, so
        //   |v|^2 = (adot^2 + bdot^2 + gdot^2) / gamma^2,   |r| = 1 / gamma
        // and the bound reduces exactly to
        //   gdot^2 < 2 mu gamma^3 - adot^2 - bdot^2.
        //
        // This is the form the old header called "the familiar" one and could only
        // claim at the fiducial direction (alpha = beta = 0).  It is now exact
        // everywhere: the gamma^2 prefactor, the (1+alpha^2)/(1+beta^2) terms, the
        // alpha*beta cross term and the s^4 denominator all existed only because
        // the old gdot had different units from adot/bdot and the gnomonic tangent
        // vectors are not unit length.
        //
        // Returns +infinity when the tangential rates already exceed escape, so the
        // caller's precision 1/sigma_gdot_sq is 0 and the prior contributes nothing.
        const double rhs = 2.0 * mu * bk.gamma * bk.gamma * bk.gamma
                           - bk.adot * bk.adot - bk.bdot * bk.bdot;
        if (!(rhs > 0.0))
            return std::numeric_limits<double>::infinity();
        return rhs;
    }

    static void bk_basis_bindings(pybind11::module &m)
    {
        namespace py = pybind11;

        py::class_<BKState>(m, "BKState",
                            "Bernstein-Khushalani parameters at epoch, barycentric.\n\n"
                            "(alpha, beta) are gnomonic tangent-plane coordinates of the\n"
                            "line-of-sight direction at a fiducial direction n0, and\n"
                            "gamma = 1/|r| with r from the barycenter.\n\n"
                            "The dots are NOT d(alpha)/dt, d(beta)/dt, d(gamma)/dt.  They are\n"
                            "the velocity components in the orthonormal fiducial basis,\n"
                            "scaled by gamma:\n\n"
                            "    adot = gamma * (v . a)\n"
                            "    bdot = gamma * (v . b)\n"
                            "    gdot = gamma * (v . n0)\n\n"
                            "In particular gdot is the line-of-sight velocity -- the\n"
                            "parameter angles-only astrometry constrains least, and the one\n"
                            "the bound-orbit energy prior regularizes.  The coordinate rates,\n"
                            "if you want them, are\n\n"
                            "    d(alpha)/dt = s * (adot - alpha * gdot)\n"
                            "    d(beta)/dt  = s * (bdot - beta  * gdot)\n"
                            "    d(gamma)/dt = -(gamma/s) * (alpha*adot + beta*bdot + gdot)\n\n"
                            "with s = sqrt(1 + alpha^2 + beta^2).  See issue #445.")
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
