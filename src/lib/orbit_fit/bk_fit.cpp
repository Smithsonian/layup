// Levenberg-Marquardt driver for the universal-BK fitter
// (feat/bk-everywhere).  Included from orbit_fit.cpp at the bottom of
// its `namespace orbit_fit { ... }` block, so all of layup's existing
// Cartesian-side machinery is in scope without forward declarations:
//
//   Observation, FitResult (from detection.cpp / fit_result.cpp)
//   residuals, partials   (from orbit_fit.h)
//   compute_residuals, create_sequences, get_weight_matrix, converged
//                          (from orbit_fit.cpp)
//   bk_basis::*            (from bk_basis.cpp -- included earlier in TU)
//
// The driver mirrors `orbit_fit()` and `run_from_vector_with_initial_guess()`
// but works in BK basis throughout: chain-rules the per-observation
// Cartesian partials produced by `compute_residuals` through the 6x6
// dcart_dbk Jacobian into BK basis, then assembles and solves
// C = B^T W B + lambda I + P_prior in BK coordinates.

// FitResult run_bk_native_fit
//
// See bk_everywhere_design.md for the algorithmic design.

FitResult run_bk_native_fit(
    struct assist_ephem *ephem,
    FitResult initial_guess,
    std::vector<Observation> &detections,
    double mu)
{
    FitResult result;
    result.method = "bk_native";
    result.flag = 1;
    result.epoch = initial_guess.epoch;
    result.csq = HUGE_VAL;
    result.niter = 0;
    result.ndof = (int)(2 * detections.size() - 6);
    result.state = initial_guess.state;
    result.cov.fill(0.0);

    if (detections.size() < 3)
    {
        // Not enough observations to constrain a 6-parameter fit.
        return result;
    }

    // ----- Pick a fiducial direction from the observations -----
    std::vector<Eigen::Vector3d> rho_hats;
    rho_hats.reserve(detections.size());
    for (const auto &obs : detections)
    {
        rho_hats.push_back(obs.rho_hat);
    }
    const BKFiducial fid = choose_fiducial(rho_hats);

    // ----- Convert the seed to BK -----
    Eigen::Matrix<double, 6, 1> cart_seed;
    for (int i = 0; i < 6; i++) cart_seed(i) = initial_guess.state[i];
    BKState bk = cartesian_to_bk(cart_seed, fid);
    const BKState bk_seed = bk;

    // ----- Fixed bound-orbit energy prior on gdot -----
    const double sgsq = sigma_gdot_sq(bk_seed, mu);
    const double gdot_precision = (std::isfinite(sgsq) && sgsq > 0.0) ? (1.0 / sgsq) : 0.0;
    Eigen::Matrix<double, 6, 6> P_prior = Eigen::Matrix<double, 6, 6>::Zero();
    P_prior(5, 5) = gdot_precision;

    // ----- LM workspace -----
    std::vector<size_t> reverse_in_seq, reverse_out_seq;
    std::vector<size_t> forward_in_seq, forward_out_seq;
    std::vector<double> times(detections.size());
    for (size_t i = 0; i < detections.size(); i++)
    {
        times[i] = detections[i].epoch;
    }
    create_sequences(times, initial_guess.epoch,
                     reverse_in_seq, reverse_out_seq,
                     forward_in_seq, forward_out_seq);

    Eigen::SparseMatrix<double> W = get_weight_matrix(detections);

    std::vector<residuals> resid_vec(detections.size());
    std::vector<partials> partials_vec(detections.size());

    // ----- LM loop -----
    // Same initial lambda and accept threshold as the Cartesian fit
    // (orbit_fit at orbit_fit.cpp L553).
    double lambda = (206265.0 * 206265.0) / 1000.0;
    const double rho_accept = 0.1;
    double chi2_prev = HUGE_VAL;
    double chi2_cur = HUGE_VAL;
    Eigen::Matrix<double, 6, 6> C_no_lambda;  // last accepted Hessian sans Marquardt damping
    bool have_accepted_step = false;

    const size_t iter_max = 100;
    const double eps = 1e-12;
    size_t iters;
    for (iters = 0; iters < iter_max; iters++)
    {
        // --- Build current Cartesian state from BK ---
        const Eigen::Matrix<double, 6, 1> cart_now = bk_to_cartesian(bk, fid);
        struct reb_particle p0;
        p0.x = cart_now(0);  p0.y = cart_now(1);  p0.z = cart_now(2);
        p0.vx = cart_now(3); p0.vy = cart_now(4); p0.vz = cart_now(5);
        p0.m = 0.0;          p0.r = 0.0;
        p0.ax = 0.0; p0.ay = 0.0; p0.az = 0.0;
        p0.hash = 0;

        // --- Residuals + Cartesian partials via the existing variational pipeline ---
        compute_residuals(ephem, p0, initial_guess.epoch,
                          detections,
                          resid_vec, partials_vec,
                          forward_in_seq, forward_out_seq,
                          reverse_in_seq, reverse_out_seq);

        // --- Assemble B_cart (2N x 6) and the residual vector ---
        const int N = (int)detections.size();
        Eigen::MatrixXd B_cart(2 * N, 6);
        Eigen::VectorXd r_vec(2 * N);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                B_cart(2 * i, j) = partials_vec[i].x_partials[j];
                B_cart(2 * i + 1, j) = partials_vec[i].y_partials[j];
            }
            r_vec(2 * i) = resid_vec[i].x_resid;
            r_vec(2 * i + 1) = resid_vec[i].y_resid;
        }

        // --- Chain rule: B_bk = B_cart * dcart_dbk(current BK state) ---
        const Eigen::Matrix<double, 6, 6> J = dcart_dbk(bk, fid);
        const Eigen::MatrixXd B_bk = B_cart * J;

        // --- Normal equations in BK basis with Marquardt damping + fixed prior ---
        const Eigen::MatrixXd Bt = B_bk.transpose();
        const Eigen::MatrixXd BtW = Bt * W;
        Eigen::Matrix<double, 6, 6> C_data = BtW * B_bk;  // pure data Hessian
        Eigen::Matrix<double, 6, 6> C = C_data
                                       + lambda * Eigen::Matrix<double, 6, 6>::Identity()
                                       + P_prior;

        // bk as a 6-vector for the prior-gradient term.  The prior mean is zero
        // (only gdot is constrained), so grad_prior = P_prior * bk_vec.
        Eigen::Matrix<double, 6, 1> bk_vec;
        bk_vec << bk.alpha, bk.beta, bk.gamma, bk.adot, bk.bdot, bk.gdot;
        Eigen::Matrix<double, 6, 1> grad = BtW * r_vec + P_prior * bk_vec;

        // chi-square including the prior contribution
        const double chi2_data = (r_vec.transpose() * W * r_vec)(0);
        const double chi2_prior = bk_vec.transpose() * P_prior * bk_vec;
        chi2_cur = chi2_data + chi2_prior;

        // --- Solve and Marquardt accept/reject ---
        Eigen::Matrix<double, 6, 1> dX = C.colPivHouseholderQr().solve(-grad);
        const double rho_num = chi2_prev - chi2_cur;
        const double rho_den = (dX.transpose() * (lambda * dX - grad)).norm();
        const double rho = rho_den > 0.0 ? rho_num / rho_den : -1.0;

        if (rho > rho_accept)
        {
            lambda *= 0.5;
            bk.alpha += dX(0);
            bk.beta  += dX(1);
            bk.gamma += dX(2);
            bk.adot  += dX(3);
            bk.bdot  += dX(4);
            bk.gdot  += dX(5);
            chi2_prev = chi2_cur;
            C_no_lambda = C_data + P_prior;  // Hessian for the final covariance
            have_accepted_step = true;
        }
        else
        {
            lambda *= 2.0;
        }

        // --- Convergence (same predicate as the Cartesian fit) ---
        const size_t ndof = detections.size() * 2 - 6;
        const double thresh = 10.0;
        Eigen::MatrixXd dX_mat = dX;
        if (converged(dX_mat, eps, chi2_cur, ndof, thresh))
        {
            result.flag = 0;
            result.csq = chi2_cur;
            break;
        }
    }

    result.niter = (int)iters;

    // ----- Covariance in BK, then transform to Cartesian -----
    if (have_accepted_step)
    {
        const Eigen::Matrix<double, 6, 6> cov_bk = C_no_lambda.inverse();
        const Eigen::Matrix<double, 6, 6> J = dcart_dbk(bk, fid);
        const Eigen::Matrix<double, 6, 6> cov_cart = J * cov_bk * J.transpose();
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                result.cov[i * 6 + j] = cov_cart(i, j);
    }

    const size_t ndof = detections.size() * 2 - 6;
    const double thresh = 10.0;
    if ((result.csq / (double)ndof) > thresh)
    {
        result.flag = 2;  // "converged" but chi2/dof is too large
    }

    // ----- Cartesian state of the converged BK fit -----
    const Eigen::Matrix<double, 6, 1> cart_final = bk_to_cartesian(bk, fid);
    for (int i = 0; i < 6; i++)
    {
        result.state[i] = cart_final(i);
    }

    return result;
}

#ifdef Py_PYTHON_H
static void bk_fit_bindings(pybind11::module &m)
{
    namespace py = pybind11;
    m.def("run_bk_native_fit", &run_bk_native_fit,
          py::arg("ephem"),
          py::arg("initial_guess"),
          py::arg("detections"),
          py::arg("mu"),
          "Universal-BK Levenberg-Marquardt orbit fit.  Reuses layup's "
          "Cartesian variational machinery, with chain-rule + bound-orbit "
          "energy prior applied in BK basis.");
}
#endif /* Py_PYTHON_H */
