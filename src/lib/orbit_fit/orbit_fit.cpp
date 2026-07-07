// Core orbit-fit routines: residuals, partials, and the
// Levenberg-Marquardt driver (plus Gauss-seeded segment fitting).

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
#include "bk_basis.cpp"
#include "../gauss/gauss.cpp"

extern "C" {
#include "rebound.h"
}

namespace orbit_fit
{
    // Optional floor on the IAS15 adaptive step size, in days. When > 0,
    // every freshly-created REBOUND sim has its ri_ias15.min_dt set to
    // this value so the integrator stops shrinking its step below the
    // floor during close encounters. The trade-off is accuracy:
    // truncating IAS15's adaptive control during a real close
    // encounter introduces error in the encounter geometry, but for
    // the LM-picker use case this is acceptable — a wildly-wrong
    // Gauss root sent through a close approach just needs to report
    // "big residual," not a precise integration.
    //
    // Default 0 means "no floor" (historical behavior). Setting
    // 1e-3 days (≈86 s) bounds the worst-case LM iteration to
    // O(arc_days / 1e-3) integrator steps, which kills the hours-long
    // grinds on phantom inner-SS roots and costs <a few arcsec on
    // real well-behaved orbits.
    static double g_ias15_min_dt_days = 0.0;

    inline void set_ias15_min_dt(double v) { g_ias15_min_dt_days = v; }
    inline double get_ias15_min_dt(void)   { return g_ias15_min_dt_days; }

    static inline void apply_ias15_min_dt(struct reb_simulation *r) {
        if (g_ias15_min_dt_days > 0.0)
            r->ri_ias15.min_dt = g_ias15_min_dt_days;
    }

    // Optional override of IAS15's adaptive step-size controller.
    // assist_attach() forces ri_ias15.adaptive_mode = 1 (the legacy
    // controller); setting this global to a non-negative value
    // overrides it on every freshly-attached sim. adaptive_mode = 2 is
    // the newer (Pham, Rein & Spiegel 2024) controller, which Hanno
    // Rein suggested (PR 324 review) may handle close-Earth encounters
    // more gracefully than the min-dt floor. Because assist_attach sets
    // the legacy value, apply_ias15_adaptive_mode() must run *after*
    // assist_attach, not before like apply_ias15_min_dt.
    //
    // Default -1 means "leave ASSIST's choice untouched" (no behavior
    // change for existing callers).
    static int g_ias15_adaptive_mode = -1;

    inline void set_ias15_adaptive_mode(int m) { g_ias15_adaptive_mode = m; }
    inline int  get_ias15_adaptive_mode(void)  { return g_ias15_adaptive_mode; }

    static inline void apply_ias15_adaptive_mode(struct reb_simulation *r) {
        if (g_ias15_adaptive_mode >= 0)
            r->ri_ias15.adaptive_mode = g_ias15_adaptive_mode;
    }
}

#include "predict.cpp"

using std::cout;
namespace py = pybind11;

namespace orbit_fit
{
    // Minimum reciprocal condition number of the normal-matrix CORRELATION matrix
    // for a non-grav (A2) fit to be considered well constrained (issue #351).
    // Below this the A2 column is effectively collinear with the state (typical on
    // short arcs) and the fit is reported as weakly constrained (flag = 6).
    static constexpr double WEAK_NONGRAV_RCOND = 1e-8;

    // Arcseconds per radian (180*3600/pi). Converts astrometric/rate
    // uncertainties from arcseconds to radians and scales residuals for display.
    static constexpr double ARCSEC_PER_RAD = 206265.0;


    // Geometry shared by all three observable residual paths, computed once by
    // compute_single_residuals after the light-time integration: the unit
    // line-of-sight rho_hat, the topocentric distance, and the per-parameter
    // variational position partials (dxk/dyk/dzk) and their range projection
    // (ddist[i] = rho_hat . d r_obj / d param_i). Sized for the max parameter
    // count (6 state + up to 3 non-grav A1/A2/A3); npar selects how many are active.
    struct ResidualGeometry
    {
        double rho_x, rho_y, rho_z; // rho_hat (unit line of sight)
        double rho, invd;           // topocentric distance and 1/rho
        double dxk[9], dyk[9], dzk[9];
        double ddist[9];
    };

    // ---- Radar (delay / Doppler) residuals + partials ----
    // Monostatic two-leg light time. The down leg (integrate_light_time in the
    // caller) put the asteroid at the bounce time using the station at the
    // receive epoch. The up leg was transmitted ~one round-trip earlier, when the
    // station had moved (Earth orbital motion ~30 km/s + rotation): ignoring this
    // biases real radar by thousands of us in delay and ~100 Hz in Doppler, far
    // above the ~1 us / sub-Hz measurement precision (quantified against (99942)
    // Apophis; see ISSUE_146_RADAR_DESIGN.md and
    // ISSUE_146_radar_realdata_crosscheck.py). We add the up leg by Taylor-
    // extrapolating the station state (pos+vel) to the transmit time t_obs - tau
    // using the observer acceleration supplied from Python. Shapiro (relativistic)
    // delay (~2 us here) is the next refinement.
    void compute_radar_residuals(struct reb_simulation *r, const Observation &this_det,
                                 int var, int npar, const ResidualGeometry &g,
                                 residuals &resid, partials &parts)
    {
        const RadarObservation &rd = std::get<RadarObservation>(this_det.observation_type);

        double xe = this_det.observer_position[0];
        double ye = this_det.observer_position[1];
        double ze = this_det.observer_position[2];

        double vax = r->particles[0].vx, vay = r->particles[0].vy, vaz = r->particles[0].vz;
        double vox = this_det.observer_velocity[0];
        double voy = this_det.observer_velocity[1];
        double voz = this_det.observer_velocity[2];
        double vrx = vax - vox, vry = vay - voy, vrz = vaz - voz; // v_rel

        double q = g.rho_x * vrx + g.rho_y * vry + g.rho_z * vrz;     // one-way range rate
        double q_ast = g.rho_x * vax + g.rho_y * vay + g.rho_z * vaz; // asteroid-only (light time)
        double ltdenom = 1.0 + q_ast / SPEED_OF_LIGHT;

        // Bounce point (asteroid at the down-leg retarded time) and observer
        // acceleration for the up-leg station extrapolation.
        double rbx = r->particles[0].x, rby = r->particles[0].y, rbz = r->particles[0].z;
        double aox = this_det.observer_acceleration[0];
        double aoy = this_det.observer_acceleration[1];
        double aoz = this_det.observer_acceleration[2];

        // Up-leg light time: iterate tau_u with the station at the transmit
        // time t_obs - (tau_d + tau_u), Taylor-extrapolated from the receive
        // epoch. rho is the converged down-leg distance from the caller.
        double tau_d = g.rho / SPEED_OF_LIGHT;
        double tau_u = tau_d;
        double rhu_x = g.rho_x, rhu_y = g.rho_y, rhu_z = g.rho_z, rho_u = g.rho;
        for (int it = 0; it < 3; it++)
        {
            double tau = tau_d + tau_u;
            double rtx_x = xe - vox * tau - 0.5 * aox * tau * tau;
            double rtx_y = ye - voy * tau - 0.5 * aoy * tau * tau;
            double rtx_z = ze - voz * tau - 0.5 * aoz * tau * tau;
            rhu_x = rbx - rtx_x;
            rhu_y = rby - rtx_y;
            rhu_z = rbz - rtx_z;
            rho_u = sqrt(rhu_x * rhu_x + rhu_y * rhu_y + rhu_z * rhu_z);
            tau_u = rho_u / SPEED_OF_LIGHT;
        }
        rhu_x /= rho_u; // up-leg unit vector (bounce -> transmit station)
        rhu_y /= rho_u;
        rhu_z /= rho_u;
        double tau = tau_d + tau_u;
        double vtx_x = vox - aox * tau; // station velocity at transmit
        double vtx_y = voy - aoy * tau;
        double vtx_z = voz - aoz * tau;

        double model_delay = tau_d + tau_u; // round-trip light time (days)
        // Round-trip range rate: down leg uses the station velocity at receive,
        // up leg at transmit.
        double model_doppler =
            (g.rho_x * (vax - vox) + g.rho_y * (vay - voy) + g.rho_z * (vaz - voz)) +
            (rhu_x * (vax - vtx_x) + rhu_y * (vay - vtx_y) + rhu_z * (vaz - vtx_z));
        resid.delay_resid = rd.delay - model_delay;
        resid.doppler_resid = rd.doppler - model_doppler;

        // Partials use the single-leg-times-two Jacobian: the up and down legs are
        // parallel to ~|Earth displacement|/rho ~ 2e-4 rad, so the two-leg residual
        // and this Jacobian agree to that level. Gauss-Newton converges on the
        // (data-driven, two-leg) residual regardless; the Jacobian only sets the
        // step direction (cf. the streak treatment). Loops over npar so non-grav
        // (A1/A2/A3) variational partials, when active, are produced for radar too.
        for (int i = 0; i < npar; i++)
        {
            size_t vn = var + i;
            // Exact single-leg range partial (light-time corrected):
            //   d rho / d param = ddist / (1 + (rho_hat . v_ast)/c).
            double range_partial = g.ddist[i] / ltdenom;
            // d(model_delay)/d param ~ (2/c) d rho / d param.
            parts.delay_partials.push_back(-2.0 * range_partial / SPEED_OF_LIGHT);

            // d(model_doppler)/d param = 2 d(rho_hat . v_rel)/d param.
            // Total position partial incl. the light-time shift of the emission
            // time (t_ret = t_obs - rho/c => d t_ret = -range_partial/c).
            double drk_x = g.dxk[i] - vax * range_partial / SPEED_OF_LIGHT;
            double drk_y = g.dyk[i] - vay * range_partial / SPEED_OF_LIGHT;
            double drk_z = g.dzk[i] - vaz * range_partial / SPEED_OF_LIGHT;
            // Velocity variational partials (the a_ast * d t_ret term is ~1e-4
            // smaller and omitted, matching the streak treatment).
            double dvk_x = r->particles[vn].vx;
            double dvk_y = r->particles[vn].vy;
            double dvk_z = r->particles[vn].vz;

            // d(rho_hat)/d param = (I - rho_hat rho_hat^T)/rho . drk, so
            //   dq = rho_hat . dvk + (v_rel . drk - q (rho_hat . drk))/rho.
            double rdotdrk = g.rho_x * drk_x + g.rho_y * drk_y + g.rho_z * drk_z;
            double dq = (g.rho_x * dvk_x + g.rho_y * dvk_y + g.rho_z * dvk_z) +
                        ((vrx * drk_x + vry * drk_y + vrz * drk_z) - q * rdotdrk) * g.invd;
            parts.doppler_partials.push_back(-2.0 * dq);
        }

        int nrows = (rd.has_delay ? 1 : 0) + (rd.has_doppler ? 1 : 0);
        resid.n_resid = parts.n_resid = nrows;
        resid.obs_kind = parts.obs_kind = ObsKind::Radar;
        resid.has_delay = parts.has_delay = rd.has_delay;
        resid.has_doppler = parts.has_doppler = rd.has_doppler;
    }

    // ---- Optical astrometry residuals + partials ----
    void compute_optical_residuals(struct reb_simulation *r, const Observation &this_det,
                                   int var, int npar, const ResidualGeometry &g,
                                   residuals &resid, partials &parts)
    {
        Eigen::Vector3d Av = this_det.a_vec;
        Eigen::Vector3d Dv = this_det.d_vec;

        double Ax = Av.x(), Ay = Av.y(), Az = Av.z();
        double Dx = Dv.x(), Dy = Dv.y(), Dz = Dv.z();

        resid.x_resid = -(g.rho_x * Ax + g.rho_y * Ay + g.rho_z * Az);
        resid.y_resid = -(g.rho_x * Dx + g.rho_y * Dy + g.rho_z * Dz);

        // dxk/dyk/dzk, ddist and invd come from the shared geometry; size for the
        // max param count (6 state + up to 3 non-grav) and loop npar.
        double drho_x[9], drho_y[9], drho_z[9];
        double dx_resid[9];
        double dy_resid[9];
        for (int i = 0; i < npar; i++)
        {
            drho_x[i] = g.dxk[i] * g.invd - g.rho_x * g.invd * g.ddist[i] - r->particles[0].vx * g.ddist[i] * g.invd / SPEED_OF_LIGHT;
            drho_y[i] = g.dyk[i] * g.invd - g.rho_y * g.invd * g.ddist[i] - r->particles[0].vy * g.ddist[i] * g.invd / SPEED_OF_LIGHT;
            drho_z[i] = g.dzk[i] * g.invd - g.rho_z * g.invd * g.ddist[i] - r->particles[0].vz * g.ddist[i] * g.invd / SPEED_OF_LIGHT;

            dx_resid[i] = -(drho_x[i] * Ax + drho_y[i] * Ay + drho_z[i] * Az);
            dy_resid[i] = -(drho_x[i] * Dx + drho_y[i] * Dy + drho_z[i] * Dz);
        }

        for (int i = 0; i < npar; i++)
        {
            parts.x_partials.push_back(dx_resid[i]);
        }
        for (int i = 0; i < npar; i++)
        {
            parts.y_partials.push_back(dy_resid[i]);
        }
    }

    // ---- Streak (sky-motion-rate) residuals + partials ----
    // Only StreakObservations contribute the two extra rate rows (on top of the
    // optical ra/dec rows). The forward model matches Sorcha's RARateCosDec/
    // DecRate (great-circle, cos(Dec) included), including the (1 - d(delta_t)/dt)
    // light-time-rate factor; the partials use the geometric rate Jacobian (the
    // omitted LT factor perturbs them by ~1e-4, negligible for LM). See orbitfit.py
    // for the observed-rate unit/convention contract.
    void compute_streak_residuals(struct reb_simulation *r, const Observation &this_det,
                                  int var, int npar, const ResidualGeometry &g,
                                  residuals &resid, partials &parts)
    {
        const StreakObservation &sk = std::get<StreakObservation>(this_det.observation_type);

        Eigen::Vector3d Av = this_det.a_vec;
        Eigen::Vector3d Dv = this_det.d_vec;
        double Ax = Av.x(), Ay = Av.y(), Az = Av.z();
        double Dx = Dv.x(), Dy = Dv.y(), Dz = Dv.z();

        // Emission-time asteroid velocity (integrate_light_time left the sim at
        // the retarded time) and the observer velocity.
        double vax = r->particles[0].vx, vay = r->particles[0].vy, vaz = r->particles[0].vz;
        double vox = this_det.observer_velocity[0];
        double voy = this_det.observer_velocity[1];
        double voz = this_det.observer_velocity[2];
        double vrx = vax - vox, vry = vay - voy, vrz = vaz - voz; // v_rel

        double q = g.rho_x * vrx + g.rho_y * vry + g.rho_z * vrz; // rho_hat . v_rel = range rate
        double ddeltatdt = q / SPEED_OF_LIGHT;

        // d(rho_hat)/dt with Sorcha's light-time-rate factor.
        double drx = vax * (1.0 - ddeltatdt) - vox;
        double dry = vay * (1.0 - ddeltatdt) - voy;
        double drz = vaz * (1.0 - ddeltatdt) - voz;
        double wx = (drx - q * g.rho_x) * g.invd;
        double wy = (dry - q * g.rho_y) * g.invd;
        double wz = (drz - q * g.rho_z) * g.invd;

        double model_ra_rate = Ax * wx + Ay * wy + Az * wz;
        double model_dec_rate = Dx * wx + Dy * wy + Dz * wz;
        resid.ra_rate_resid = sk.ra_rate - model_ra_rate;
        resid.dec_rate_resid = sk.dec_rate - model_dec_rate;

        // Geometry Jacobians (observed basis A/D held fixed):
        //   p = X.rho_hat, s = X.v_rel, q = rho_hat.v_rel
        //   dR/dr = -(q X + p v_rel + (s - 3 p q) rho_hat)/dist^2
        //   dR/dv = (X - p rho_hat)/dist
        double pA = Ax * g.rho_x + Ay * g.rho_y + Az * g.rho_z;
        double sA = Ax * vrx + Ay * vry + Az * vrz;
        double pD = Dx * g.rho_x + Dy * g.rho_y + Dz * g.rho_z;
        double sD = Dx * vrx + Dy * vry + Dz * vrz;
        double invd2 = g.invd * g.invd;

        double dRdrA[3] = {-(q * Ax + pA * vrx + (sA - 3 * pA * q) * g.rho_x) * invd2,
                           -(q * Ay + pA * vry + (sA - 3 * pA * q) * g.rho_y) * invd2,
                           -(q * Az + pA * vrz + (sA - 3 * pA * q) * g.rho_z) * invd2};
        double dRdvA[3] = {(Ax - pA * g.rho_x) * g.invd, (Ay - pA * g.rho_y) * g.invd, (Az - pA * g.rho_z) * g.invd};
        double dRdrD[3] = {-(q * Dx + pD * vrx + (sD - 3 * pD * q) * g.rho_x) * invd2,
                           -(q * Dy + pD * vry + (sD - 3 * pD * q) * g.rho_y) * invd2,
                           -(q * Dz + pD * vrz + (sD - 3 * pD * q) * g.rho_z) * invd2};
        double dRdvD[3] = {(Dx - pD * g.rho_x) * g.invd, (Dy - pD * g.rho_y) * g.invd, (Dz - pD * g.rho_z) * g.invd};

        for (int i = 0; i < npar; i++)
        {
            size_t vn = var + i;
            // d r_obj/d param_i: variational position + light-time shift of the
            // emission time (v_ast * d t_ret, t_ret = t_obs - rho/c).
            double ltf = g.ddist[i] / SPEED_OF_LIGHT;
            double drk_x = g.dxk[i] - vax * ltf;
            double drk_y = g.dyk[i] - vay * ltf;
            double drk_z = g.dzk[i] - vaz * ltf;
            // d v_obj/d param_i: velocity variational partials (the a_ast * d t_ret
            // light-time term is ~1e-4 smaller and omitted).
            double dvk_x = r->particles[vn].vx;
            double dvk_y = r->particles[vn].vy;
            double dvk_z = r->particles[vn].vz;

            double dmodel_ra = dRdrA[0] * drk_x + dRdrA[1] * drk_y + dRdrA[2] * drk_z +
                               dRdvA[0] * dvk_x + dRdvA[1] * dvk_y + dRdvA[2] * dvk_z;
            double dmodel_dec = dRdrD[0] * drk_x + dRdrD[1] * drk_y + dRdrD[2] * drk_z +
                                dRdvD[0] * dvk_x + dRdvD[1] * dvk_y + dRdvD[2] * dvk_z;

            // residual = observed - model, so partial = -d(model)/d param.
            parts.ra_rate_partials.push_back(-dmodel_ra);
            parts.dec_rate_partials.push_back(-dmodel_dec);
        }

        resid.n_resid = 4;
        parts.n_resid = 4;
    }

    void compute_single_residuals(struct assist_ephem *ephem,
                                  struct assist_extras *ax,
                                  int var,
                                  Observation this_det,
                                  residuals &resid,
                                  partials &parts,
                                  int npar = 6)
    {
        // npar is the number of fitted parameters: 6 (state) or 6+N when fitting
        // non-gravitational parameters. The extra variational particles for those
        // params are contiguous after the 6 state ones (var+6, var+7, ...; see
        // add_variational_particles / issue #351), so every partial loop below
        // simply runs to npar and reads var+i uniformly -- ASSIST drives the param
        // particles via particle_params while the state ones use their seeds.

        // Takes a detection/observation and
        // a simulation as arguments.
        // Returns the model observation, the
        // residuals, and the partial derivatives.

        struct reb_simulation *r = ax->sim;

        double jd_tdb = this_det.epoch;

        double xe = this_det.observer_position[0];
        double ye = this_det.observer_position[1];
        double ze = this_det.observer_position[2];

        // 5. compare the model result to the observation.
        //   This means dotting the model unit vector with the
        //   A and D vectors of the observation

        reb_vec3d r_obs = {xe, ye, ze};
        double t_obs = jd_tdb - ephem->jd_ref;

        int j = 0;
        // Iterate the (up-)leg light time: leaves the sim at the retarded
        // emission/bounce time with the asteroid position AND velocity.
        integrate_light_time(ax, j, t_obs, r_obs, 0.0, 4, SPEED_OF_LIGHT);

        // Topocentric range vector and distance; then normalise to the unit
        // line-of-sight rho_hat (rho keeps the distance).
        double rho_x = r->particles[j].x - xe;
        double rho_y = r->particles[j].y - ye;
        double rho_z = r->particles[j].z - ze;
        double rho = sqrt(rho_x * rho_x + rho_y * rho_y + rho_z * rho_z);

        rho_x /= rho;
        rho_y /= rho;
        rho_z /= rho;
        double invd = 1. / rho;

        // Shared geometry for all three observable paths (rho_hat, the
        // topocentric distance, and the per-parameter variational partials).
        ResidualGeometry g;
        g.rho_x = rho_x;
        g.rho_y = rho_y;
        g.rho_z = rho_z;
        g.rho = rho;
        g.invd = invd;
        for (int i = 0; i < npar; i++)
        {
            size_t vn = var + i;
            g.dxk[i] = r->particles[vn].x;
            g.dyk[i] = r->particles[vn].y;
            g.dzk[i] = r->particles[vn].z;
            g.ddist[i] = g.rho_x * g.dxk[i] + g.rho_y * g.dyk[i] + g.rho_z * g.dzk[i];
        }

        // Dispatch by observation type. Radar is standalone; optical always runs
        // for non-radar, and a streak adds its two rate rows on top.
        if (std::holds_alternative<RadarObservation>(this_det.observation_type))
        {
            compute_radar_residuals(r, this_det, var, npar, g, resid, parts);
            return;
        }
        compute_optical_residuals(r, this_det, var, npar, g, resid, parts);
        if (std::holds_alternative<StreakObservation>(this_det.observation_type))
        {
            compute_streak_residuals(r, this_det, var, npar, g, resid, parts);
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
                                    std::vector<size_t> &out_seq,
                                    int nongrav_mask = 0, const double *a123 = nullptr,
                                    const double *gofr = nullptr)
    {

        // Pass in this simulation stuff to keep it flexible
        struct reb_simulation *r = reb_simulation_create();
        apply_ias15_min_dt(r);
        struct assist_extras *ax = assist_attach(r, ephem);
        apply_ias15_adaptive_mode(r);  // after attach: ASSIST forces mode 1

        // 0. Set initial time, relative to ephem->jd_ref
        r->t = epoch - ephem->jd_ref;

        // 1. Add the main particle to the REBOUND simulation.
        reb_simulation_add(r, p0);

        // 2. incorporate the variational particles. When fitting non-grav params we
        //    add one extra (zero-seed) variational particle per active parameter
        //    for d(state)/dA_i. active[] holds the param indices (0=A1,1=A2,2=A3).
        std::vector<int> active;
        for (int i = 0; i < 3; i++)
            if (nongrav_mask & (1 << i))
                active.push_back(i);
        int nactive = (int)active.size();
        int npar = 6 + nactive;
        int var;
        add_variational_particles(r, 0, &var, nactive);

        // 2b. Non-gravitational setup (issue #351). Enable the Marsden non-grav
        //     force with the standard 1/r^2 g(r), set the asteroid's [A1,A2,A3], and
        //     point each parameter variational particle's direction (dA_i = 1) via
        //     particle_params. ASSIST then integrates d(state)/dA_i into that var
        //     particle. particle_params is laid out as 3*(N_real + N_var): the first
        //     triple is the real particle's [A1,A2,A3]; the triple for variational
        //     particle v is its [dA1,dA2,dA3]. NB: ASSIST skips the whole non-grav
        //     block (incl. its variational source) when A1=A2=A3=0, so the caller
        //     seeds the active params with a tiny nonzero value to keep their
        //     columns non-degenerate. The pp buffer must outlive the integrations.
        std::vector<double> pp;
        if (nactive > 0)
        {
            ax->forces = (enum ASSIST_FORCES)(ax->forces | ASSIST_FORCE_NON_GRAVITATIONAL);
            // g(r) = alpha*(r/r0)^-nm*(1+(r/r0)^nn)^-nk. Default (gofr==nullptr) is the
            // asteroidal inverse-square law (r/r0)^-2; a comet fit passes its Marsden
            // law as [alpha, nm, nn, nk, r0] (e.g. water-ice sublimation).
            if (gofr)
            {
                ax->alpha = gofr[0]; ax->nm = gofr[1]; ax->nn = gofr[2]; ax->nk = gofr[3]; ax->r0 = gofr[4];
            }
            else
            {
                ax->alpha = 1.0; ax->nm = 2.0; ax->nk = 0.0; ax->nn = 1.0; ax->r0 = 1.0;
            }
            int nreal = 1;
            int nvar = npar; // 6 state + nactive param particles
            pp.assign(3 * (nreal + nvar), 0.0);
            if (a123)
                for (int i = 0; i < 3; i++)
                    pp[i] = a123[i]; // real particle [A1,A2,A3]
            for (int k = 0; k < nactive; k++)
                // k-th param var particle (var particle index 6+k) tracks active[k]
                pp[3 * (nreal + 6 + k) + active[k]] = 1.0;
            ax->particle_params = pp.data();
        }

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
                                     parts,
                                     npar);
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
                           std::vector<size_t> &reverse_out_seq,
                           int nongrav_mask = 0, const double *a123 = nullptr,
                           const double *gofr = nullptr)
    {

        compute_residuals_sequence(ephem, p0, epoch,
                                   detections,
                                   resid_vec,
                                   partials_vec,
                                   forward_in_seq,
                                   forward_out_seq,
                                   nongrav_mask, a123, gofr);

        compute_residuals_sequence(ephem, p0, epoch,
                                   detections,
                                   resid_vec,
                                   partials_vec,
                                   reverse_in_seq,
                                   reverse_out_seq,
                                   nongrav_mask, a123, gofr);
    }

    // Forward declaration: create_sequences is defined later in this translation
    // unit (after compute_dX), but residuals_at_state below needs it.
    void create_sequences(std::vector<double> &times, double epoch,
                          std::vector<size_t> &reverse_in_seq, std::vector<size_t> &reverse_out_seq,
                          std::vector<size_t> &forward_in_seq, std::vector<size_t> &forward_out_seq);

    // Per-observation residuals + mapped covariance at a fixed state, computed
    // with the fit's own efficient forward+backward single-pass integration.
    // This is a fast replacement for predict_sequence in the outlier-rejection
    // loop: predict_sequence does N fresh integrations from the epoch (~10x the
    // cost of a whole fit), while this reuses create_sequences + compute_residuals
    // for one forward + one backward pass over the sorted arc.
    //
    // Returns, per detection (input order): [x_resid, y_resid, oc00, oc01, oc11,
    // n_resid], where (x_resid, y_resid) is the raw tangent-plane residual
    // (radians) and oc** is the 2x2 mapped state covariance B*cov*B^T (symmetric,
    // so oc10 == oc01). Astrometry/optical rows only; radar/streak rows fall back
    // to zero covariance (their partials layout differs).
    std::vector<std::array<double, 6>>
    residuals_at_state(struct assist_ephem *ephem,
                       FitResult fit,
                       std::vector<Observation> &detections,
                       Eigen::MatrixXd &cov)
    {
        struct reb_particle p0;
        p0.x = fit.state[0];
        p0.y = fit.state[1];
        p0.z = fit.state[2];
        p0.vx = fit.state[3];
        p0.vy = fit.state[4];
        p0.vz = fit.state[5];
        double epoch = fit.epoch;

        size_t N = detections.size();
        std::vector<residuals> resid_vec(N);
        std::vector<partials> partials_vec(N);

        std::vector<double> times(N);
        for (size_t i = 0; i < N; ++i)
            times[i] = detections[i].epoch;

        std::vector<size_t> forward_in_seq, forward_out_seq, reverse_in_seq, reverse_out_seq;
        create_sequences(times, epoch,
                         reverse_in_seq, reverse_out_seq,
                         forward_in_seq, forward_out_seq);

        compute_residuals(ephem, p0, epoch,
                          detections,
                          resid_vec, partials_vec,
                          forward_in_seq, forward_out_seq,
                          reverse_in_seq, reverse_out_seq);

        std::vector<std::array<double, 6>> out(N);
        for (size_t i = 0; i < N; ++i)
        {
            double oc00 = 0.0, oc01 = 0.0, oc11 = 0.0;
            // Optical astrometry: 2 rows, B is 2x6 over the state parameters.
            if (partials_vec[i].x_partials.size() >= 6 && partials_vec[i].y_partials.size() >= 6)
            {
                Eigen::Matrix<double, 2, 6> B;
                for (int j = 0; j < 6; ++j)
                {
                    B(0, j) = partials_vec[i].x_partials[j];
                    B(1, j) = partials_vec[i].y_partials[j];
                }
                Eigen::Matrix2d oc = B * cov * B.transpose();
                oc00 = oc(0, 0);
                oc01 = oc(0, 1);
                oc11 = oc(1, 1);
            }
            out[i] = {resid_vec[i].x_resid, resid_vec[i].y_resid,
                      oc00, oc01, oc11, static_cast<double>(resid_vec[i].n_resid)};
        }
        return out;
    }

    // Consider having this accept Eigen matrices as
    // input
    void compute_dX(std::vector<residuals> &resid_vec,
                    std::vector<partials> &partials_vec,
                    Eigen::SparseMatrix<double> W,
                    Eigen::MatrixXd &dX,
                    Eigen::MatrixXd &C,
                    Eigen::MatrixXd &chi2,
                    Eigen::MatrixXd &grad,
                    double lambda,
                    int npar = 6,
                    const Eigen::MatrixXd *prior_info = nullptr,
                    const Eigen::VectorXd *prior_dx = nullptr)
    {

        const int mlength = (int)partials_vec.size();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
        Eigen::MatrixXd eye = MatrixXd::Identity(npar, npar);

        // Each observation contributes n_resid rows (2 for astrometry, 4 for a
        // streak: ra/dec position + ra/dec rate). Lay them out with a per-obs
        // prefix-sum offset; the weight matrix uses the identical ordering.
        std::vector<int> row_off(mlength);
        int total_rows = 0;
        for (int i = 0; i < mlength; i++)
        {
            row_off[i] = total_rows;
            total_rows += partials_vec[i].n_resid;
        }

        B.resize(total_rows, npar);
        Eigen::MatrixXd resid_v(total_rows, 1);
        for (size_t i = 0; i < partials_vec.size(); i++)
        {
            const int r0 = row_off[i];

            // Radar observations contribute a delay row and/or a Doppler row;
            // they have no astrometry rows, so handle them separately. Loop npar
            // so non-grav columns are filled when active.
            if (partials_vec[i].obs_kind == ObsKind::Radar)
            {
                int row = r0;
                if (partials_vec[i].has_delay)
                {
                    for (int j = 0; j < npar; j++)
                        B(row, j) = partials_vec[i].delay_partials[j];
                    resid_v(row) = resid_vec[i].delay_resid;
                    row++;
                }
                if (partials_vec[i].has_doppler)
                {
                    for (int j = 0; j < npar; j++)
                        B(row, j) = partials_vec[i].doppler_partials[j];
                    resid_v(row) = resid_vec[i].doppler_resid;
                    row++;
                }
                continue;
            }

            for (int j = 0; j < npar; j++)
            {
                B(r0, j) = partials_vec[i].x_partials[j];
                B(r0 + 1, j) = partials_vec[i].y_partials[j];
            }
            resid_v(r0) = resid_vec[i].x_resid;
            resid_v(r0 + 1) = resid_vec[i].y_resid;

            if (partials_vec[i].n_resid == 4)
            {
                for (int j = 0; j < npar; j++)
                {
                    B(r0 + 2, j) = partials_vec[i].ra_rate_partials[j];
                    B(r0 + 3, j) = partials_vec[i].dec_rate_partials[j];
                }
                resid_v(r0 + 2) = resid_vec[i].ra_rate_resid;
                resid_v(r0 + 3) = resid_vec[i].dec_rate_resid;
            }
        }

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Bt = B.transpose();
        C = Bt * W * B + lambda * eye; // This is where the extra term for LM is.

        grad = Bt * W * resid_v;

        // Gaussian prior / information-filter term (issue #419). For a sequential
        // update the prior N(x0, P0) summarising the previously-fit observations
        // contributes its information matrix Lambda0 = P0^-1 to the normal matrix
        // and Lambda0*(x - x0) to the gradient (x0 = prior mean, x = current
        // iterate). B and resid_v here are assembled over the NEW observations
        // only, so C = BtWB + Lambda0 is exactly the batch normal matrix of the
        // full (old summarised + new) problem, and its inverse (below, after the
        // LM damping is removed) is the posterior covariance. When prior_info is
        // null this reduces to the ordinary least-squares step.
        if (prior_info != nullptr)
        {
            C += *prior_info;
            grad += (*prior_info) * (*prior_dx);
        }

        dX = C.colPivHouseholderQr().solve(-grad);
        // An alternative looks like this.  It's probably less stable.
        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G = C.inverse();
        // dX = -G * Bt * W * resid_v;

        chi2 = resid_v.transpose() * W * resid_v;
        // reset for inverse covariance
        C -= lambda * eye;
    }


    int converged(Eigen::MatrixXd dX, double eps, double chi2)
    {
        // A NaN chi-square or step means the fit has diverged, not converged.
        // Without this guard the loop below reports convergence on a NaN step,
        // because abs(NaN) > eps is false for every element, so the function
        // falls through to "return 1" -- making orbit_fit report flag = 0
        // (success) for a NaN solution.
        // (The reduced-chi-square quality check lives after the LM loop, where
        // chi2_final/ndof > thresh sets flag = 2; see issue #367.)
        if (std::isnan(chi2))
        {
            return 0;
        }
        for (size_t i = 0; i < dX.size(); i++)
        {
            if (std::isnan(dX(i)) || abs(dX(i)) > eps)
            {
                return 0;
            }
        }
        return 1;
    }

    typedef Eigen::Triplet<double> T;

    // Number of residual rows a single observation contributes: 2 (astrometry),
    // 4 (streak: ra/dec position + ra/dec rate), or 1-2 (radar: delay and/or
    // Doppler). Centralised so the layout, weight matrix, and dof count agree.
    size_t observation_residual_rows(const Observation &d)
    {
        if (std::holds_alternative<RadarObservation>(d.observation_type))
        {
            const auto &rd = std::get<RadarObservation>(d.observation_type);
            return (rd.has_delay ? 1 : 0) + (rd.has_doppler ? 1 : 0);
        }
        return std::holds_alternative<StreakObservation>(d.observation_type) ? 4 : 2;
    }

    // Total number of residual rows across all observations. Used for both the
    // normal-equation layout and the degrees-of-freedom count.
    size_t total_residual_rows(const std::vector<Observation> &detections)
    {
        size_t n = 0;
        for (const auto &d : detections)
            n += observation_residual_rows(d);
        return n;
    }

    Eigen::SparseMatrix<double> get_weight_matrix(std::vector<Observation> &detections)
    {
        const int mlength = (int)detections.size();

        // Match compute_dX's row layout: 2 (astrometry), 4 (streak), or 1-2 (radar).
        std::vector<int> row_off(mlength);
        int total_rows = 0;
        for (int i = 0; i < mlength; i++)
        {
            row_off[i] = total_rows;
            total_rows += (int)observation_residual_rows(detections[i]);
        }

        std::vector<T> tripletList;
        tripletList.reserve(total_rows);
        Eigen::SparseMatrix<double> W(total_rows, total_rows);

        for (size_t i = 0; i < mlength; i++)
        {
            const int r0 = row_off[i];

            // Radar: diagonal weights for the delay and/or Doppler rows, in the
            // same order compute_dX lays them out (delay first, then Doppler).
            if (std::holds_alternative<RadarObservation>(detections[i].observation_type))
            {
                const auto &rd = std::get<RadarObservation>(detections[i].observation_type);
                int row = r0;
                if (rd.has_delay)
                {
                    double d_unc = detections[i].delay_unc ? *detections[i].delay_unc : 1.0;
                    tripletList.push_back(T(row, row, 1.0 / (d_unc * d_unc)));
                    row++;
                }
                if (rd.has_doppler)
                {
                    double v_unc = detections[i].doppler_unc ? *detections[i].doppler_unc : 1.0;
                    tripletList.push_back(T(row, row, 1.0 / (v_unc * v_unc)));
                    row++;
                }
                continue;
            }

            double x_unc = *detections[i].ra_unc;
            tripletList.push_back(T(r0, r0, 1.0 / (x_unc * x_unc)));
            double y_unc = *detections[i].dec_unc;
            tripletList.push_back(T(r0 + 1, r0 + 1, 1.0 / (y_unc * y_unc)));

            if (std::holds_alternative<StreakObservation>(detections[i].observation_type))
            {
                double rar_unc = detections[i].ra_rate_unc ? *detections[i].ra_rate_unc : (24.0 / ARCSEC_PER_RAD);
                double der_unc = detections[i].dec_rate_unc ? *detections[i].dec_rate_unc : (24.0 / ARCSEC_PER_RAD);
                tripletList.push_back(T(r0 + 2, r0 + 2, 1.0 / (rar_unc * rar_unc)));
                tripletList.push_back(T(r0 + 3, r0 + 3, 1.0 / (der_unc * der_unc)));
            }
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
                  size_t iter_max,
                  int nongrav_mask = 0,    // bitmask of non-grav params to fit (1=A1,2=A2,4=A3)
                  double *a123io = nullptr, // [A1,A2,A3] seed in / fitted out (3 doubles)
                  const Eigen::MatrixXd *prior_info = nullptr, // #419 sequential update:
                                                              // prior information matrix
                                                              // Lambda0 = P0^-1 (npar x npar),
                                                              // or null for ordinary LSQ
                  const double *gofr = nullptr) // [alpha,nm,nn,nk,r0] Marsden g(r); null -> r^-2
    { // runtime

        // Number of fitted parameters: 6 (state) + one per active non-grav param.
        // active[] holds the param indices (0=A1,1=A2,2=A3) in column order. ASSIST
        // skips the non-grav block when A1=A2=A3=0, which would zero the param
        // columns; seed each active param with a tiny nonzero value so its column
        // is non-degenerate (the partial is independent of the param's magnitude,
        // and 1e-15 au/day^2 is dynamically negligible).
        std::vector<int> active;
        for (int i = 0; i < 3; i++)
            if (nongrav_mask & (1 << i))
                active.push_back(i);
        int nactive = (int)active.size();
        int npar = 6 + nactive;
        double a123[3] = {0.0, 0.0, 0.0};
        if (a123io)
            for (int i = 0; i < 3; i++)
                a123[i] = a123io[i];
        for (int k = 0; k < nactive; k++)
            if (a123[active[k]] == 0.0)
                a123[active[k]] = 1e-15;

        // #419 sequential update: snapshot the prior mean x0 (= the seed state)
        // so each iteration can form the offset (x - x0) that the prior gradient
        // term Lambda0*(x - x0) needs. Only used when prior_info is supplied.
        Eigen::VectorXd x0;
        if (prior_info != nullptr)
        {
            x0.resize(npar);
            x0(0) = p0.x;
            x0(1) = p0.y;
            x0(2) = p0.z;
            x0(3) = p0.vx;
            x0(4) = p0.vy;
            x0(5) = p0.vz;
            for (int k = 0; k < nactive; k++)
                x0(6 + k) = a123[active[k]];
        }

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

        // Initialise so that a fit which runs to iter_max without converging (or
        // iter_max == 0) reports a meaningful chi-square rather than whatever the
        // caller left in this out-parameter. chi2_final is refreshed every
        // iteration below.
        chi2_final = HUGE_VAL;

        double rho_accept = 0.1;

        // Do an initial step
        double lambda = (ARCSEC_PER_RAD * ARCSEC_PER_RAD) / 1000;
        for (iters = 0; iters < iter_max; iters++)
        {

            compute_residuals(ephem, p0, epoch,
                              detections,
                              resid_vec,
                              partials_vec,
                              forward_in_seq,
                              forward_out_seq,
                              reverse_in_seq,
                              reverse_out_seq,
                              nongrav_mask, a123, gofr);

            // #419 sequential update: offset of the current iterate from the
            // prior mean, for the prior gradient term Lambda0*(x - x0). Empty and
            // unused when there is no prior.
            Eigen::VectorXd prior_dx;
            if (prior_info != nullptr)
            {
                prior_dx.resize(npar);
                prior_dx(0) = p0.x - x0(0);
                prior_dx(1) = p0.y - x0(1);
                prior_dx(2) = p0.z - x0(2);
                prior_dx(3) = p0.vx - x0(3);
                prior_dx(4) = p0.vy - x0(4);
                prior_dx(5) = p0.vz - x0(5);
                for (int k = 0; k < nactive; k++)
                    prior_dx(6 + k) = a123[active[k]] - x0(6 + k);
            }

            Eigen::MatrixXd grad;
            compute_dX(resid_vec, partials_vec, W, dX, C, chi2, grad, lambda, npar,
                       prior_info, prior_info != nullptr ? &prior_dx : nullptr);

            double chi2_d = chi2(0, 0);
            // Track the most recent chi-square so the reported value is honest
            // even when the loop exits by exhausting iter_max (e.g. the LM is
            // stuck rejecting steps against a gross outlier, dX -> 0). The
            // reported value is always the data-only chi-square (over the new
            // observations); the prior's contribution is folded into the LM gain
            // ratio below, not into chi2_final.
            chi2_final = chi2_d;

            // A NaN chi-square means the fit has diverged (e.g. from a
            // degenerate IOD seed on a too-short tracklet). Stop immediately,
            // leaving flag != 0, so the result is reported as a failure rather
            // than grinding through every remaining iteration on NaNs.
            if (std::isnan(chi2_d))
            {
                chi2_final = chi2_d;
                break;
            }

            // Full regularised objective for the LM gain ratio: data chi-square
            // over the new obs plus the prior quadratic (x - x0)^T Lambda0 (x - x0).
            // With a strong prior and few new obs (the common steady-state case)
            // the new-obs chi-square barely moves, so scoring steps on the data
            // term alone would stall the damping; the prior term keeps it honest.
            double obj = chi2_d;
            if (prior_info != nullptr)
                obj += prior_dx.dot((*prior_info) * prior_dx);

            double rho = (chi2_prev - obj) / (dX.transpose() * (lambda * dX - grad)).norm();

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
                for (int k = 0; k < nactive; k++)
                    a123[active[k]] += dX(6 + k);
                chi2_prev = obj;
            }
            else
            {

                // Reject the step
                // leave the state
                // increase lambda
                // repeat, unless too many iterations

                lambda *= 2.0;
            }

            int cflag = converged(dX, eps, chi2_d);
            if (cflag)
            {
                flag = 0;
                chi2_final = chi2_d;
                break;
            }
        }

        // Degrees of freedom for the reduced-chi-square quality check. In an
        // ordinary fit the npar parameters are estimated from the observations, so
        // dof = rows - npar. In a #419 sequential update those npar parameters are
        // constrained by the prior information (Lambda0), not consumed from the new
        // observations, so the data-only chi-square (chi2_final) is assessed over
        // the new-obs rows themselves. Skip the check when there are no rows to
        // avoid a divide-by-zero (a prior-only update, which the driver avoids).
        size_t nrows = total_residual_rows(detections);
        size_t ndof = (prior_info != nullptr) ? nrows : (nrows - npar);
        double thresh = 10.0;
        if (ndof > 0 && (chi2_final / ndof) > thresh)
        {
            flag = 2;
        }

        cov = C.inverse();

        // Weak-constraint guard for the non-grav fit (issue #351). On a short arc
        // a non-grav column becomes nearly collinear with the state directions, so
        // the joint solve yields garbage params and a contaminated state. Detect
        // this via the conditioning of the CORRELATION matrix of the normal matrix
        // C -- the raw rcond(C) is useless here because C mixes AU and AU/day^2
        // scales, but the correlation matrix (C normalized by its diagonal) is
        // unit-invariant and measures genuine collinearity. If it is near-singular,
        // mark the fit weakly constrained (flag = 6) so the driver falls back to
        // the 6-parameter solution and reports the non-grav params as NaN.
        if (nactive > 0 && flag == 0)
        {
            Eigen::VectorXd diag = C.diagonal();
            if ((diag.array() > 0.0).all())
            {
                Eigen::VectorXd s = diag.array().rsqrt(); // 1/sqrt(C_ii)
                Eigen::MatrixXd R = s.asDiagonal() * C * s.asDiagonal();
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(R);
                const Eigen::VectorXd &sv = svd.singularValues();
                double rcond = sv(sv.size() - 1) / sv(0);
                if (!(rcond >= WEAK_NONGRAV_RCOND)) // also catches NaN
                    flag = 6;
            }
            else
            {
                flag = 6; // non-positive variance -> degenerate
            }
        }

        // Report the fitted non-grav params back to the caller. cov is the joint
        // npar x npar covariance; the caller reads its non-grav diagonal entries
        // for the parameter uncertainties.
        if (nactive > 0 && a123io)
            for (int i = 0; i < 3; i++)
                a123io[i] = a123[i];

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

    FitResult run_from_vector_with_initial_guess(struct assist_ephem *ephem,
                                                  FitResult initial_guess,
                                                  std::vector<Observation> &detections,
                                                  size_t iter_max = 100,
                                                  int nongrav_mask = 0,
                                                  std::vector<double> gofr = {})
    {
        int success = 1;
        size_t iters;
        double chi2_final;
        size_t dof;

        // Non-grav g(r) law [alpha,nm,nn,nk,r0]; empty -> the default inverse-square.
        if (!gofr.empty() && gofr.size() != 5)
            throw std::invalid_argument("gofr must have 5 elements [alpha, nm, nn, nk, r0]");
        const double *gofr_ptr = gofr.empty() ? nullptr : gofr.data();

        std::vector<residuals> resid_vec(detections.size());
        std::vector<partials> partials_vec(detections.size());

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

        // Seed [A1,A2,A3] from the initial guess for the params being fit (#351).
        double a123[3] = {initial_guess.a1, initial_guess.a2, initial_guess.a3};

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
            iter_max,
            nongrav_mask,
            a123,
            nullptr, // prior_info (ordinary LSQ)
            gofr_ptr);

        int nactive = 0;
        std::vector<int> active;
        for (int i = 0; i < 3; i++)
            if (nongrav_mask & (1 << i))
            {
                active.push_back(i);
                nactive++;
            }
        int npar = 6 + nactive;
        dof = total_residual_rows(detections) - npar;

        FitResult result;
	
        result.method = "orbit_fit";
        result.flag = flag;

        result.epoch = epoch;
        result.csq = chi2_final;
        result.ndof = dof;
        result.niter = iters;
        result.state[0] = p1.x;
        result.state[1] = p1.y;
        result.state[2] = p1.z;
        result.state[3] = p1.vx;
        result.state[4] = p1.vy;
        result.state[5] = p1.vz;

	if (flag == 0) {
            // Populate our covariance matrix (the 6x6 state block; for a joint
            // state+A2 fit this is the marginal state covariance, the top-left
            // block of the 7x7 inverse).
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    // flatten the covariance matrix
                    result.cov[(i * 6) + j] = cov(i, j);
                }
            }
        }

        // Non-grav results (issue #351): fitted values + 1-sigma uncertainties
        // from the corresponding diagonal entries of the joint covariance. Active
        // params are column 6+k for the k-th active param; inactive stay at zero.
        result.nongrav_mask = nongrav_mask;
        result.a1 = a123[0];
        result.a2 = a123[1];
        result.a3 = a123[2];
        double *unc[3] = {&result.a1_unc, &result.a2_unc, &result.a3_unc};
        for (int k = 0; k < nactive; k++)
            *unc[active[k]] =
                (flag == 0 && cov.rows() >= npar) ? std::sqrt(cov(6 + k, 6 + k)) : NAN;

	return result;

    }

    // #419 sequential / information-filter update. Refine a prior orbit fit using
    // ONLY new observations, folding the previously-fit observations in through
    // their Gaussian summary (the prior state + covariance). To the extent the
    // prior is well approximated by a Gaussian at its mean, the result equals a
    // full batch refit over old+new observations -- the assumption the Python
    // driver guards with a nonlinearity check and a full-refit fallback. The fit
    // epoch is pinned to the prior's epoch, so the prior covariance needs no
    // state-transition propagation.
    //
    // The prior is passed as a covariance (FitResult.cov); the information matrix
    // Lambda0 = P0^-1 is formed here via a Cholesky (LLT) solve. Passing the prior
    // as a covariance rather than a raw information matrix keeps the boundary
    // square-root-friendly: a future SRIF variant can carry the Cholesky factor of
    // Lambda0 through this same signature without changing callers.
    FitResult run_sequential_update(struct assist_ephem *ephem,
                                    FitResult prior,
                                    std::vector<Observation> &new_detections,
                                    size_t iter_max = 100)
    {
        size_t iters = 0;
        double chi2_final = HUGE_VAL;
        double eps = 1e-12;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov;

        std::vector<residuals> resid_vec(new_detections.size());
        std::vector<partials> partials_vec(new_detections.size());

        double epoch = prior.epoch; // pin the update epoch to the prior

        FitResult result;
        result.method = "sequential_update";
        result.epoch = epoch;
        result.nongrav_mask = 0;
        for (int i = 0; i < 6; i++)
            result.state[i] = prior.state[i]; // fallback state if the update fails

        // Prior information matrix Lambda0 = P0^-1 (6x6 state block). P0 is a fit
        // covariance, hence symmetric positive-definite; the LLT (Cholesky) solve
        // is its stable inverse and the natural SRIF upgrade point.
        Eigen::Matrix<double, 6, 6> P0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                P0(i, j) = prior.cov[i * 6 + j];

        Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(P0);
        if (llt.info() != Eigen::Success)
        {
            // Prior covariance not positive-definite (a degenerate prior); the
            // information update is ill-posed. Flag failure so the driver falls
            // back to a full refit over all observations.
            result.flag = 7;
            return result;
        }
        Eigen::MatrixXd Lambda0 = llt.solve(Eigen::Matrix<double, 6, 6>::Identity());

        reb_particle p1;
        p1.x = prior.state[0];
        p1.y = prior.state[1];
        p1.z = prior.state[2];
        p1.vx = prior.state[3];
        p1.vy = prior.state[4];
        p1.vz = prior.state[5];

        int flag = orbit_fit(
            ephem, p1, epoch, new_detections,
            resid_vec, partials_vec, iters, chi2_final, cov,
            eps, iter_max, /*nongrav_mask=*/0, /*a123io=*/nullptr, &Lambda0);

        result.flag = flag;
        result.csq = chi2_final;
        // #419 dof convention: the prior supplies the npar parameter constraints,
        // so the data-only chi-square is assessed over the new-obs rows.
        result.ndof = total_residual_rows(new_detections);
        result.niter = iters;
        result.state[0] = p1.x;
        result.state[1] = p1.y;
        result.state[2] = p1.z;
        result.state[3] = p1.vx;
        result.state[4] = p1.vy;
        result.state[5] = p1.vz;
        if (flag == 0)
        {
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                    result.cov[i * 6 + j] = cov(i, j); // posterior (Lambda0 + BtWB)^-1
        }
        return result;
    }

// Universal-BK fit (LM driver in BK basis).  Inlined here, inside
// `namespace orbit_fit`, so all of layup's existing Cartesian-fit
// helpers (compute_residuals, create_sequences, get_weight_matrix,
// converged) and the Observation/FitResult types are in scope without
// forward declarations.  Math primitives come from bk_basis.cpp,
// included at the top of orbit_fit.cpp.
#include "bk_fit.cpp"

// Universal-BK 5-parameter linear IOD.  Same inlining rationale as
// bk_fit.cpp above -- inlined inside `namespace orbit_fit` so that
// Observation, FitResult, SPEED_OF_LIGHT, and the bk_basis primitives
// are all in scope.
#include "bk_iod.cpp"

#ifdef Py_PYTHON_H
    static void orbit_fit_bindings(py::module &m)
    {
        py::class_<assist_ephem>(m, "assist_ephem");
        m.def("orbit_fit", &orbit_fit::orbit_fit, R"pbdoc(Core orbit fit function.)pbdoc");
        m.def("get_ephem", &orbit_fit::get_ephem, R"pbdoc(get ephemeris)pbdoc");
        m.def("run_from_vector_with_initial_guess",
              &orbit_fit::run_from_vector_with_initial_guess,
              py::arg("ephem"), py::arg("initial_guess"),
              py::arg("detections"), py::arg("iter_max") = 100,
              py::arg("nongrav_mask") = 0,
              py::arg("gofr") = std::vector<double>{},
              R"pbdoc(
                Takes an assist_ephem object, a vector of observations, an
                initial guess, and (optionally) a cap on LM iterations
                (`iter_max`, default 100), and runs orbit fit. Reducing
                iter_max gives a cheap screening pass for use in
                multi-candidate IOD picker loops.
            )pbdoc");
        m.def("run_sequential_update",
              &orbit_fit::run_sequential_update,
              py::arg("ephem"), py::arg("prior"),
              py::arg("new_detections"), py::arg("iter_max") = 100,
              R"pbdoc(
                Sequential / information-filter orbit update (issue #419). Takes a
                prior FitResult (its state, 6x6 covariance and epoch summarise the
                previously-fit observations) and ONLY the new observations, and
                returns the updated fit. The prior covariance enters as the
                information matrix Lambda0 = P0^-1 added to the normal equations, so
                the result equals a full batch refit over old+new observations when
                the prior is locally Gaussian. The epoch is pinned to the prior's.
                Fails (flag != 0) if the prior covariance is not positive-definite.
            )pbdoc");
        m.def("residuals_at_state", &orbit_fit::residuals_at_state,
              py::arg("ephem"), py::arg("fit"), py::arg("detections"), py::arg("cov"),
              R"pbdoc(
                Per-observation residuals + mapped covariance at a fixed state,
                using the fit's efficient forward+backward single-pass integration
                (a fast replacement for predict_sequence in the outlier-rejection
                loop). Returns, per detection: [x_resid, y_resid (radians),
                cov00, cov01, cov11 (the 2x2 B*cov*B^T), n_resid].
            )pbdoc");
        m.def("set_ias15_min_dt", &orbit_fit::set_ias15_min_dt,
              py::arg("days"),
              R"pbdoc(
                Set a floor on IAS15's adaptive step size in days. Default
                0 = no floor. A typical safe value is 1e-3 (~86 s): bounds
                worst-case LM-iteration wall time on phantom IOD roots
                whose trajectories pass close to Earth, at the cost of a
                few arcsec of accuracy in real close encounters.
            )pbdoc");
        m.def("get_ias15_min_dt", &orbit_fit::get_ias15_min_dt,
              R"pbdoc(Current IAS15 min-dt floor in days (0 = off).)pbdoc");
        m.def("set_ias15_adaptive_mode", &orbit_fit::set_ias15_adaptive_mode,
              py::arg("mode"),
              R"pbdoc(
                Override IAS15's adaptive step-size controller. -1 (default)
                leaves ASSIST's choice, which is the legacy mode 1. Set 2 to
                use the newer (Pham, Rein & Spiegel 2024) controller. Applied
                after assist_attach, which forces mode 1.
            )pbdoc");
        m.def("get_ias15_adaptive_mode", &orbit_fit::get_ias15_adaptive_mode,
              R"pbdoc(Current IAS15 adaptive_mode override (-1 = ASSIST default).)pbdoc");
    }
#endif /* Py_PYTHON_H */

} // namespace: orbit_fit
