// Universal-BK 5-parameter linear initial orbit determination.
//
// Included from orbit_fit.cpp at the bottom of `namespace orbit_fit`,
// after bk_fit.cpp, so all of layup's existing types and the BK
// primitives from bk_basis.cpp are in scope:
//   Observation, FitResult           (from detection.cpp / fit_result.cpp)
//   SPEED_OF_LIGHT                   (from predict.cpp; AU/day)
//   BKState, BKFiducial,
//   choose_fiducial, bk_to_cartesian,
//   dcart_dbk, sigma_gdot_sq         (from bk_basis.cpp)
//
// The model is the small-angle / no-gravity linear projection used by
// Bernstein & Khushalani's `prelim_fit` (liborbfit/orbfit1.c), adapted
// to barycenter-origin BK.  Per observation, the predicted tangent-
// plane components are linear in (alpha, beta, gamma, adot, bdot):
//
//   t_i = (obs_jd_i - (r_obs_i . n0) / c) - epoch           // light-time corrected dt
//   X_i = r_obs_i . a,   Y_i = r_obs_i . b                  // observer in BK tangent plane
//   x_obs_i = (rho_hat_i . a) / (rho_hat_i . n0)            // observation, gnomonic
//   y_obs_i = (rho_hat_i . b) / (rho_hat_i . n0)
//
//   x_obs_i = (alpha + adot * t_i - gamma * (X_i - x_obs_i * ze_i))
//   y_obs_i = (beta  + bdot * t_i - gamma * (Y_i - y_obs_i * ze_i))
//
// gdot is pinned to 0 with a nominal prior variance from the bound-
// orbit energy constraint (sigma_gdot_sq, also matching prelim_fit's
// covar[5][5] = 0.1 * (2 pi)^2 * gamma^3 convention up to a small
// numerical factor).
//
// ---------------------------------------------------------------------------
// THE PERSPECTIVE TERM  (issue #446, fixed)
// ---------------------------------------------------------------------------
//
// The exact gnomonic relation carries a denominator:
//
//   x_obs = (alpha + adot*t - gamma*X) / (1 + gdot*t - gamma*ze)
//
// where ze = r_obs . n0 is the observer's projection onto the line of sight.
// This code previously dropped that denominator, setting it to 1, because
// keeping it appeared to make the system nonlinear in gamma.  The omitted
// factor scales as |gamma * ze| ~ 1 AU / r_helio -- negligible for TNOs, but
// 0.29 at 3.5 AU and 0.40 at 2.5 AU, and it was the dominant seed error in the
// inner solar system.
//
// It does not in fact make the system nonlinear.  Multiplying through by the
// denominator and collecting the unknowns on one side:
//
//   x_obs = alpha + adot*t - gamma*(X - x_obs*ze) - gdot*(x_obs*t)
//
// x_obs is DATA, so x_obs*ze is a known coefficient and gamma still appears to
// first order.  Restoring the term costs one changed coefficient per row, not
// an iteration.
//
// Measured effect on heliocentric-distance accuracy (synthetic, noiseless,
// 7 observations, 10-day arc; relative improvement is the meaningful figure --
// absolute values depend on geometry):
//
//   population        before      after
//   --------------  ----------  ---------
//   mainbelt 2.5AU     53.8%       8.4%
//   mainbelt 3.5AU     44.6%       4.3%
//   centaur 15 AU      15.4%       0.15%
//   classical 42 AU     6.2%       0.00%
//   sednoid 80 AU       3.4%       0.01%
//
// The main belt does not reach sub-percent, and should not be expected to:
// what remains once the perspective term is handled is the straight-line,
// no-gravity assumption itself, which is a genuine limit at 2.5 AU over a
// 10-day arc.  But a few percent in heliocentric distance is a seed the LM can
// start from, where 45-54% is not.
//
// 🔑 AVAILABLE BUT NOT DONE: gdot need not be pinned.  It enters the relation
// above linearly, through the coefficient -(x_obs * t), so freeing it is one
// extra column and measurably improves every population again (mainbelt 2.5AU
// 8.4% -> 5.5%, classical 42AU 0.00%).  It is NOT done here because it is
// blocked on issue #445: this file's dots are BK's scaled velocity components,
// while bk_basis's are true time derivatives, and the two meet where the
// parameters are packed into a BKState below.  That mismatch is currently
// invisible *because* gdot is pinned to 0 -- zero is zero in either convention.
// Freeing gdot is precisely the "reasonable future change" #445 warns will make
// it break silently.  Resolve #445 first.
//
// In all cases, BK-IOD is intended as a SEED for the full BK or
// Cartesian LM fit, not as a final orbit.

FitResult run_bk_iod(
    std::vector<Observation> &observations,
    double epoch,
    double mu)
{
    FitResult result;
    result.method = "bk_iod";
    result.flag = 1;
    result.epoch = epoch;
    result.csq = HUGE_VAL;
    result.niter = 0;
    result.ndof = 0;
    for (int i = 0; i < 6; i++) result.state[i] = 0.0;
    result.cov.fill(0.0);

    const int N = (int)observations.size();
    // Need 2N >= 5 equations to constrain 5 parameters.
    if (N < 3)
    {
        return result;
    }

    // ----- Fiducial direction from the observations -----
    std::vector<Eigen::Vector3d> rho_hats;
    rho_hats.reserve(observations.size());
    for (const auto &obs : observations)
    {
        rho_hats.push_back(obs.rho_hat);
    }
    const BKFiducial fid = choose_fiducial(rho_hats);

    // ----- Build the (2N x 5) weighted least-squares system -----
    // Parameter ordering matches BKState's positional fields:
    //   p = (alpha, beta, gamma, adot, bdot)
    Eigen::MatrixXd A(2 * N, 5);
    Eigen::VectorXd b(2 * N);
    Eigen::VectorXd w(2 * N);   // diagonal weights (precision = 1/variance)
    Eigen::VectorXd ze_v(N);    // observer along n0, per observation (issue #446)
    Eigen::VectorXd t_v(N);     // light-time corrected dt, per observation

    for (int i = 0; i < N; i++)
    {
        const auto &obs = observations[i];
        const Eigen::Vector3d r_obs(
            obs.observer_position[0],
            obs.observer_position[1],
            obs.observer_position[2]);

        // Light-time corrected dt: when the observer is closer to the
        // target along the line of sight (ze > 0), the topocentric
        // distance is smaller, light-time is smaller, and emission time
        // is *later* relative to the baseline -- so we ADD ze/c.  Any
        // constant offset is absorbed by the rates and doesn't affect
        // the linear fit.
        const double ze = r_obs.dot(fid.n0);
        const double t_i = (obs.epoch + ze / SPEED_OF_LIGHT) - epoch;

        // Observer projected onto the BK tangent plane.
        const double X_i = r_obs.dot(fid.a);
        const double Y_i = r_obs.dot(fid.b);

        // Observed line-of-sight projected onto the BK tangent plane
        // via gnomonic projection.  Matches the predicted x, y in the
        // linear model: alpha, beta are themselves gnomonic coordinates,
        // so the observed tangent-plane coords must also be gnomonic.
        // (Small-angle approximation rho_hat . a would introduce
        // O(alpha^2) error -- noticeable at TNO arcs of several degrees.)
        const double rho_n0 = obs.rho_hat.dot(fid.n0);
        const double x_obs = obs.rho_hat.dot(fid.a) / rho_n0;
        const double y_obs = obs.rho_hat.dot(fid.b) / rho_n0;

        // Per-observation uncertainties.  Default to a 1" guess if the
        // Observation didn't carry an explicit uncertainty; the LM
        // pipeline applies the same fallback elsewhere.
        const double sigma_ra = obs.ra_unc.value_or(1.0 / 206265.0);
        const double sigma_dec = obs.dec_unc.value_or(1.0 / 206265.0);
        const double w_x = 1.0 / (sigma_ra * sigma_ra);
        const double w_y = 1.0 / (sigma_dec * sigma_dec);

        ze_v(i) = ze;
        t_v(i) = t_i;

        // Row for the x (alpha-axis) equation.
        A(2 * i, 0) = 1.0;
        A(2 * i, 1) = 0.0;
        // Perspective term restored (issue #446).  The exact gnomonic relation is
        //     x_obs * (1 + gdot*t - gamma*ze) = alpha + adot*t - gamma*X
        // and collecting the unknowns on one side gives
        //     x_obs = alpha + adot*t - gamma*(X - x_obs*ze) - gdot*(x_obs*t)
        // Every unknown is still first order: x_obs is DATA, so x_obs*ze is a known
        // coefficient.  The system stays linear and closed-form -- the omitted
        // denominator never required iteration, only this coefficient.
        A(2 * i, 2) = -(X_i - x_obs * ze);
        A(2 * i, 3) = t_i;
        A(2 * i, 4) = 0.0;
        b(2 * i) = x_obs;
        w(2 * i) = w_x;

        // Row for the y (beta-axis) equation.
        A(2 * i + 1, 0) = 0.0;
        A(2 * i + 1, 1) = 1.0;
        A(2 * i + 1, 2) = -(Y_i - y_obs * ze);
        A(2 * i + 1, 3) = 0.0;
        A(2 * i + 1, 4) = t_i;
        b(2 * i + 1) = y_obs;
        w(2 * i + 1) = w_y;
    }

    // Weighted normal equations: H p = g  with  H = A^T W A,  g = A^T W b.
    auto solve_with = [&](const Eigen::VectorXd &weights,
                          Eigen::Matrix<double, 5, 5> &H_out) {
        const Eigen::MatrixXd AtW = A.transpose() * weights.asDiagonal();
        H_out = AtW * A;
        const Eigen::Matrix<double, 5, 1> g = AtW * b;
        return Eigen::Matrix<double, 5, 1>(H_out.colPivHouseholderQr().solve(g));
    };

    Eigen::Matrix<double, 5, 5> H;
    Eigen::Matrix<double, 5, 1> p = solve_with(w, H);

    // One weight-refinement pass (issue #446).  Multiplying the relation through by
    // the perspective denominator scales each residual by d_i, so the weighting
    // implied above is w_i * d_i^2 rather than w_i.  Re-solving with w_i / d_i^2,
    // using the gamma just obtained, restores the intended chi-square.  This moves
    // the point estimate negligibly (~1e-4 relative) but it is the returned
    // COVARIANCE that matters: it is used as the seed covariance downstream.
    // Skipped if any d_i is non-positive, which would mean the object is inside the
    // observer's own distance along the line of sight -- not a regime this model
    // describes.
    {
        const double gamma_0 = p(2);
        Eigen::VectorXd w2(2 * N);
        bool usable = std::isfinite(gamma_0) && gamma_0 > 0.0;
        for (int i = 0; i < N && usable; i++)
        {
            // gdot is pinned to 0, so d_i = 1 - gamma*ze_i.
            const double d_i = 1.0 - gamma_0 * ze_v(i);
            if (!(d_i > 0.0) || !std::isfinite(d_i)) { usable = false; break; }
            w2(2 * i) = w(2 * i) / (d_i * d_i);
            w2(2 * i + 1) = w(2 * i + 1) / (d_i * d_i);
        }
        if (usable)
        {
            Eigen::Matrix<double, 5, 5> H2;
            const Eigen::Matrix<double, 5, 1> p2 = solve_with(w2, H2);
            if (p2.allFinite() && p2(2) > 0.0)
            {
                p = p2;
                H = H2;
                w = w2;
            }
        }
    }

    // Pack into a BKState (gdot pinned to 0).
    BKState bk;
    bk.alpha = p(0);
    bk.beta = p(1);
    bk.gamma = p(2);
    bk.adot = p(3);
    bk.bdot = p(4);
    bk.gdot = 0.0;

    // Reject pathological solutions (gamma must be positive for the
    // forward transform to make sense).
    if (!std::isfinite(bk.alpha) || !std::isfinite(bk.beta) || !std::isfinite(bk.gamma)
        || !std::isfinite(bk.adot) || !std::isfinite(bk.bdot) || bk.gamma <= 0.0)
    {
        return result;
    }

    // 6x6 BK covariance: top-left 5x5 from the fit's inverse-normal
    // matrix; gdot row/column from the bound-orbit energy prior (gdot
    // is treated as uncorrelated with the data-constrained params).
    const Eigen::Matrix<double, 5, 5> cov_5 = H.inverse();
    Eigen::Matrix<double, 6, 6> cov_bk = Eigen::Matrix<double, 6, 6>::Zero();
    cov_bk.block<5, 5>(0, 0) = cov_5;
    const double sgsq = sigma_gdot_sq(bk, mu);
    cov_bk(5, 5) = std::isfinite(sgsq) ? sgsq : 0.0;

    // Transform to Cartesian via the analytic 6x6 chain rule.
    const Eigen::Matrix<double, 6, 1> cart = bk_to_cartesian(bk, fid);
    const Eigen::Matrix<double, 6, 6> J = dcart_dbk(bk, fid);
    const Eigen::Matrix<double, 6, 6> cov_cart = J * cov_bk * J.transpose();

    for (int i = 0; i < 6; i++)
    {
        result.state[i] = cart(i);
    }
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            result.cov[i * 6 + j] = cov_cart(i, j);
        }
    }

    // Residual chi-square at the linear solution (for diagnostic only).
    const Eigen::VectorXd resid = A * p - b;
    double csq = 0.0;
    for (int k = 0; k < 2 * N; k++)
    {
        csq += w(k) * resid(k) * resid(k);
    }
    result.csq = csq;
    result.ndof = (2 * N >= 5) ? (2 * N - 5) : 0;
    result.niter = 1;  // closed-form -- one "iteration"
    result.flag = 0;
    return result;
}

#ifdef Py_PYTHON_H
static void bk_iod_bindings(pybind11::module &m)
{
    namespace py = pybind11;
    m.def("run_bk_iod", &run_bk_iod,
          py::arg("observations"),
          py::arg("epoch"),
          py::arg("mu"),
          "Universal-BK 5-parameter linear initial orbit determination.  "
          "Closed-form weighted least-squares over (alpha, beta, gamma, "
          "adot, bdot); gdot pinned to 0 with covariance from the bound-"
          "orbit energy prior.  Returns a FitResult with barycentric "
          "Cartesian state at the supplied epoch.");
}
#endif /* Py_PYTHON_H */
