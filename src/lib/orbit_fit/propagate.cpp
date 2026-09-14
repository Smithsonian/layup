// Propagate a fitted state, and its covariance, to a new reference epoch
// (issue #578).
//
// `convert` changes the PARAMETERIZATION of an orbit at a fixed epoch; nothing
// changed the epoch itself. The integration machinery was already here --
// predict.cpp drives assist_integrate_or_interpolate -- but every entry point
// returned sky quantities, so a state at a different epoch had to be obtained
// by writing the integration again outside the package.
//
// This is a purely dynamical operation. It deliberately applies NO light-time
// correction: `predict` does, because it computes an observable, and
// conflating the two would be a subtle and expensive error.
//
// The covariance is carried by the state-transition matrix, C' = Phi C Phi^T,
// with Phi obtained from six variational particles seeded along the state
// axes -- the same mechanism the fitter uses for its partials.
//
// Included from orbit_fit.cpp at the bottom of `namespace orbit_fit`, so that
// FitResult, add_variational_particles and apply_ias15_adaptive_mode are all
// in scope. It is not a standalone translation unit.

    // Returns the propagated fit and the 6x6 state-transition matrix (row-major,
    // Phi[i][j] = d state_i(t) / d state_j(t0)).
    //
    // On failure the returned FitResult carries flag != 0 and the input epoch,
    // so a caller that ignores the flag gets an unchanged orbit rather than a
    // plausible wrong one.
    std::pair<FitResult, std::array<double, 36>>
    propagate_state(struct assist_ephem *ephem, FitResult fit, double target_epoch)
    {
        FitResult out = fit;
        std::array<double, 36> stm{};

        // Nothing to do, and doing it anyway would burn an integration.
        if (target_epoch == fit.epoch)
        {
            for (int i = 0; i < 6; i++) stm[i * 6 + i] = 1.0;
            return {out, stm};
        }

        struct reb_simulation *r = reb_simulation_create();
        r->t = fit.epoch - ephem->jd_ref;

        struct reb_particle p0 = {0};
        p0.x = fit.state[0]; p0.y = fit.state[1]; p0.z = fit.state[2];
        p0.vx = fit.state[3]; p0.vy = fit.state[4]; p0.vz = fit.state[5];
        reb_simulation_add(r, p0);

        // Six variational particles seeded with the unit state axes: their
        // states at the target epoch ARE the columns of Phi. `np = 0` is passed
        // explicitly -- REBOUND's default of -1 would leave ASSIST filling no
        // variational acceleration at all, and Phi would come back as the
        // identity with no error raised.
        int var = 0;
        add_variational_particles(r, 0, &var, 0);

        struct assist_extras *ax = assist_attach(r, ephem);
        apply_ias15_adaptive_mode(r);  // after attach: ASSIST forces mode 1

        assist_integrate_or_interpolate(ax, target_epoch - ephem->jd_ref);

        if (r->status == REB_STATUS_GENERIC_ERROR)
        {
            // Outside ephemeris coverage, or the integration failed. Report the
            // input unchanged with a non-zero flag rather than a wrong state.
            out.flag = 1;
            assist_free(ax);
            reb_simulation_free(r);
            return {out, stm};
        }

        const struct reb_particle pf = r->particles[0];
        out.state[0] = pf.x; out.state[1] = pf.y; out.state[2] = pf.z;
        out.state[3] = pf.vx; out.state[4] = pf.vy; out.state[5] = pf.vz;
        out.epoch = target_epoch;

        for (int j = 0; j < 6; j++)
        {
            const struct reb_particle v = r->particles[var + j];
            stm[0 * 6 + j] = v.x;  stm[1 * 6 + j] = v.y;  stm[2 * 6 + j] = v.z;
            stm[3 * 6 + j] = v.vx; stm[4 * 6 + j] = v.vy; stm[5 * 6 + j] = v.vz;
        }

        // C' = Phi C Phi^T. A zeroed input covariance stays zeroed, which is
        // what an unconverged fit carries (#547) and is the right answer for it.
        Eigen::Matrix<double, 6, 6> Phi, C;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
            {
                Phi(i, j) = stm[i * 6 + j];
                C(i, j) = fit.cov[i * 6 + j];
            }
        const Eigen::Matrix<double, 6, 6> Cn = Phi * C * Phi.transpose();
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                out.cov[i * 6 + j] = Cn(i, j);

        assist_free(ax);
        reb_simulation_free(r);
        return {out, stm};
    }

    static void propagate_bindings(py::module &m)
    {
        m.def("propagate_state", &orbit_fit::propagate_state,
              R"pbdoc(
                Propagate a fitted state and its covariance to a new reference
                epoch (issue #578).

                Returns ``(fit, stm)``: the FitResult re-referenced to
                ``target_epoch``, and the 6x6 state-transition matrix as 36
                row-major values, ``stm[i*6+j] = d state_i(t) / d state_j(t0)``.

                The covariance is carried as ``C' = Phi C Phi^T``. No light-time
                correction is applied -- this is a dynamical operation, not an
                observable. On failure, including a target outside the
                ephemeris coverage, the returned fit carries ``flag != 0`` and
                the ORIGINAL epoch and state.
              )pbdoc",
              py::arg("ephem"), py::arg("fit"), py::arg("target_epoch"));
    }
