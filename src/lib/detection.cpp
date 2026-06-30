#include <variant>
#include <optional>
#include <array>
#include <Eigen/Dense>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// --- Observation Variant Types ---
// Each variant computes a unit direction vector (rho_hat) from the provided ra and dec.

/*
    // This is the old code for the tangent plane vectors.
    // It is not used in the new code, but is kept here for reference.
    double Ax =  -theta_y;
    double Ay =   theta_x;
    double Az =   0.0;
    double A = sqrt(Ax*Ax + Ay*Ay + Az*Az);
    Ax /= A; Ay /= A; Az /= A;
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
*/

namespace orbit_fit
{
    Eigen::Vector3d rho_hat_from_ra_dec(double ra, double dec)
    {
        Eigen::Vector3d rho_hat;
        rho_hat.x() = std::cos(dec) * std::cos(ra);
        rho_hat.y() = std::cos(dec) * std::sin(ra);
        rho_hat.z() = std::sin(dec);
        return rho_hat;
    }

    // Unit tangent vector in the direction of increasing RA: (-sin α, cos α, 0).
    // The celestial pole is the z-axis. Singular only at ρ̂_z = ±1.
    Eigen::Vector3d a_vec_from_rho_hat(const Eigen::Vector3d &rho_hat)
    {
        double cd = std::sqrt(rho_hat.x() * rho_hat.x() + rho_hat.y() * rho_hat.y());
        Eigen::Vector3d a_vec;
        a_vec.x() = -rho_hat.y() / cd;
        a_vec.y() = rho_hat.x() / cd;
        a_vec.z() = 0.0;
        return a_vec;
    }

    // Unit tangent vector in the direction of increasing Dec.
    Eigen::Vector3d d_vec_from_rho_hat(const Eigen::Vector3d &rho_hat)
    {
        double cd = std::sqrt(rho_hat.x() * rho_hat.x() + rho_hat.y() * rho_hat.y());
        Eigen::Vector3d d_vec;
        d_vec.x() = -rho_hat.z() * rho_hat.x() / cd;
        d_vec.y() = -rho_hat.z() * rho_hat.y() / cd;
        d_vec.z() = cd;
        return d_vec;
    }
    struct AstrometryObservation
    {
        AstrometryObservation() = default;
        // Constructor: takes ra and dec (in radians) and computes rho_hat.
    };

    struct StreakObservation
    {
        double ra_rate;
        double dec_rate;

        StreakObservation() = default;
        // Constructor: takes ra, dec and the corresponding rates, and computes rho_hat.
        StreakObservation(double ra_rate, double dec_rate) : ra_rate(ra_rate), dec_rate(dec_rate) {}
    };

    // Radar observation: round-trip delay (light time) and/or Doppler
    // (round-trip range rate). has_delay/has_doppler flag which components are
    // present; either or both may be supplied. The user/JPL units
    // (microseconds, Hz) are converted to these internal units at Python ingest
    // (see orbitfit.py): delay in days, doppler in au/day. No observed sky
    // direction is needed -- the radar residual derives the line of sight from
    // the modeled asteroid position.
    struct RadarObservation
    {
        double delay = 0.0;   // round-trip light time (days)
        double doppler = 0.0; // round-trip range rate (au/day)
        bool has_delay = false;
        bool has_doppler = false;

        RadarObservation() = default;
        RadarObservation(double delay, double doppler, bool has_delay, bool has_doppler)
            : delay(delay), doppler(doppler), has_delay(has_delay), has_doppler(has_doppler) {}
    };

    using ObservationType = std::variant<AstrometryObservation, StreakObservation, RadarObservation>;

    // Default astrometric 1-sigma uncertainty: 1 arcsecond expressed in radians
    // (206265 arcsec per radian). Used when a caller supplies no per-observation
    // RA/Dec uncertainties.
    constexpr double DEFAULT_ASTROMETRY_UNC_RAD = 1.0 / 206265.0;

    // Default streak (sky-motion-rate) 1-sigma uncertainty: 24 arcseconds in
    // radians (206265 arcsec per radian), used when a caller supplies no
    // per-observation RA-rate/Dec-rate uncertainties.
    constexpr double DEFAULT_RATE_UNC_RAD = 24.0 / 206265.0;

    // --- Main Observation Structure ---
    // Now, Observation has private constructors and public static factory methods.
    struct Observation
    {
        std::string objID;
        double epoch; // utc jd
        ObservationType observation_type;
        std::array<double, 3> observer_position;
        std::array<double, 3> observer_velocity;
        // Barycentric observer acceleration (au/day^2). Only used by the radar
        // two-leg light-time model, which Taylor-extrapolates the station state
        // back to the signal transmit time (~one round-trip earlier). Defaults to
        // zero, so it has no effect on optical/streak observations.
        std::array<double, 3> observer_acceleration{{0.0, 0.0, 0.0}};

        // Computed unit direction vector
        Eigen::Vector3d rho_hat;
        // tangent plane vectors
        Eigen::Vector3d a_vec;
        Eigen::Vector3d d_vec;

        std::optional<Eigen::MatrixXd> inverse_covariance;
        std::optional<double> mag;
        std::optional<double> mag_err;
        std::optional<double> epoch_err;
        std::optional<double> ra_unc;
        std::optional<double> dec_unc;
        // Sky-motion-rate uncertainties (radians/day), used to weight the streak
        // rate residuals. Only meaningful for StreakObservations.
        std::optional<double> ra_rate_unc;
        std::optional<double> dec_rate_unc;
        // Radar delay/Doppler 1-sigma uncertainties, in the same internal units
        // as RadarObservation (delay: days; doppler: au/day). Only meaningful
        // for RadarObservations.
        std::optional<double> delay_unc;
        std::optional<double> doppler_unc;

    private:
        // Private constructor used by the factory methods.
        Observation(double epoch_val,
                    const std::array<double, 3> &obs_position,
                    const std::array<double, 3> &obs_velocity)
            : epoch(epoch_val),
              observer_position(obs_position),
              observer_velocity(obs_velocity)
        {
        }

    public:
        // Astrometry observation with the default RA/Dec uncertainties; delegates
        // to the explicit-uncertainty constructor below.
        Observation(
            double ep,
            std::array<double, 3> obs_position,
            std::array<double, 3> obs_velocity,
            std::array<double, 3> rho,
            std::array<double, 3> avec,
            std::array<double, 3> dvec)
            : Observation(ep, obs_position, obs_velocity, rho, avec, dvec,
                          DEFAULT_ASTROMETRY_UNC_RAD, DEFAULT_ASTROMETRY_UNC_RAD)
        {
        }

        Observation(
            double ep,
            std::array<double, 3> obs_position,
            std::array<double, 3> obs_velocity,
            std::array<double, 3> rho,
            std::array<double, 3> avec,
            std::array<double, 3> dvec,
            double ra_uncy,
            double dec_uncy)
        {
            epoch = ep;
            observer_position = obs_position;
            observer_velocity = obs_velocity;
            observation_type = AstrometryObservation();

            rho_hat.x() = rho[0];
            rho_hat.y() = rho[1];
            rho_hat.z() = rho[2];

            a_vec.x() = avec[0];
            a_vec.y() = avec[1];
            a_vec.z() = avec[2];

            d_vec.x() = dvec[0];
            d_vec.y() = dvec[1];
            d_vec.z() = dvec[2];

            ra_unc = ra_uncy;
            dec_unc = dec_uncy;
        }

        Observation() {}

        // Factory method for an Astrometry observation.
        static Observation from_astrometry(double ra, double dec, double epoch_val,
                                           const std::array<double, 3> &obs_position,
                                           const std::array<double, 3> &obs_velocity)
        {
            Observation obs(epoch_val, obs_position, obs_velocity);
            obs.observation_type = AstrometryObservation();
            obs.ra_unc = DEFAULT_ASTROMETRY_UNC_RAD;
            obs.dec_unc = DEFAULT_ASTROMETRY_UNC_RAD;
            obs.rho_hat = rho_hat_from_ra_dec(ra, dec);
            obs.a_vec = a_vec_from_rho_hat(obs.rho_hat);
            obs.d_vec = d_vec_from_rho_hat(obs.rho_hat);
            return obs;
        }

        // Factory method for an Astrometry observation with an object ID.
        static Observation from_astrometry_with_id(std::string objID,
                                                   double ra, double dec, double epoch_val,
                                                   const std::array<double, 3> &obs_position,
                                                   const std::array<double, 3> &obs_velocity)
        {
            Observation obs = from_astrometry(ra, dec, epoch_val, obs_position, obs_velocity);
            obs.objID = objID;
            return obs;
        }

        // Factory method for a Streak observation.
        // ra_rate/dec_rate and their uncertainties are in radians/day (great-circle;
        // ra_rate already carries the cos(Dec) factor -- see orbitfit.py ingest).
        static Observation from_streak(double ra, double dec, double ra_rate, double dec_rate,
                                       double epoch_val,
                                       const std::array<double, 3> &obs_position,
                                       const std::array<double, 3> &obs_velocity,
                                       double ra_rate_uncy = DEFAULT_RATE_UNC_RAD,
                                       double dec_rate_uncy = DEFAULT_RATE_UNC_RAD)
        {
            Observation obs(epoch_val, obs_position, obs_velocity);
            obs.observation_type = StreakObservation(ra_rate, dec_rate);
            obs.rho_hat = rho_hat_from_ra_dec(ra, dec);
            obs.a_vec = a_vec_from_rho_hat(obs.rho_hat);
            obs.d_vec = d_vec_from_rho_hat(obs.rho_hat);
            obs.ra_unc = 1.0 / 206265;
            obs.dec_unc = 1.0 / 206265;
            obs.ra_rate_unc = ra_rate_uncy;
            obs.dec_rate_unc = dec_rate_uncy;
            return obs;
        }

        static Observation from_streak_with_id(std::string objID,
                                               double ra, double dec, double ra_rate, double dec_rate,
                                               double epoch_val,
                                               const std::array<double, 3> &obs_position,
                                               const std::array<double, 3> &obs_velocity,
                                               double ra_rate_uncy = DEFAULT_RATE_UNC_RAD,
                                               double dec_rate_uncy = DEFAULT_RATE_UNC_RAD)
        {
            Observation obs = from_streak(ra, dec, ra_rate, dec_rate, epoch_val, obs_position, obs_velocity,
                                          ra_rate_uncy, dec_rate_uncy);
            obs.objID = objID;
            return obs;
        }

        // Factory method for a Radar observation.
        // delay = round-trip light time (days); doppler = round-trip range rate
        // (au/day); uncertainties in those same units. has_delay/has_doppler
        // select which components contribute residual rows (1 or 2 total). No
        // rho_hat/a_vec/d_vec are set: the radar residual derives the line of
        // sight from the modeled asteroid position (see orbit_fit.cpp).
        static Observation from_radar(double delay, double doppler,
                                      bool has_delay, bool has_doppler,
                                      double epoch_val,
                                      const std::array<double, 3> &obs_position,
                                      const std::array<double, 3> &obs_velocity,
                                      double delay_uncy = 1.0,
                                      double doppler_uncy = 1.0,
                                      const std::array<double, 3> &obs_acceleration = {{0.0, 0.0, 0.0}})
        {
            Observation obs(epoch_val, obs_position, obs_velocity);
            obs.observation_type = RadarObservation(delay, doppler, has_delay, has_doppler);
            obs.delay_unc = delay_uncy;
            obs.doppler_unc = doppler_uncy;
            obs.observer_acceleration = obs_acceleration;
            return obs;
        }

        static Observation from_radar_with_id(std::string objID,
                                              double delay, double doppler,
                                              bool has_delay, bool has_doppler,
                                              double epoch_val,
                                              const std::array<double, 3> &obs_position,
                                              const std::array<double, 3> &obs_velocity,
                                              double delay_uncy = 1.0,
                                              double doppler_uncy = 1.0,
                                              const std::array<double, 3> &obs_acceleration = {{0.0, 0.0, 0.0}})
        {
            Observation obs = from_radar(delay, doppler, has_delay, has_doppler, epoch_val,
                                         obs_position, obs_velocity, delay_uncy, doppler_uncy,
                                         obs_acceleration);
            obs.objID = objID;
            return obs;
        }
    };

    static void detection_bindings(py::module &m)
    {
        py::class_<AstrometryObservation>(m, "AstrometryObservation")
            .def(py::init<>());

        // Bind StreakObservation type.
        py::class_<StreakObservation>(m, "StreakObservation")
            .def(py::init<double, double>(),
                 py::arg("ra_rate"), py::arg("dec_rate"))
            .def_readonly("ra_rate", &StreakObservation::ra_rate)
            .def_readonly("dec_rate", &StreakObservation::dec_rate);

        // Bind RadarObservation type.
        py::class_<RadarObservation>(m, "RadarObservation")
            .def(py::init<double, double, bool, bool>(),
                 py::arg("delay"), py::arg("doppler"),
                 py::arg("has_delay"), py::arg("has_doppler"))
            .def_readonly("delay", &RadarObservation::delay)
            .def_readonly("doppler", &RadarObservation::doppler)
            .def_readonly("has_delay", &RadarObservation::has_delay)
            .def_readonly("has_doppler", &RadarObservation::has_doppler);

        // Bind the main Observation type with multiple overloaded constructors.
        py::class_<Observation>(m, "Observation")
            // Constructor for an Astrometry observation.
            // bind the ::from_astrometry factory method
            .def(py::init<>())
            .def(py::init<double, std::array<double, 3>, std::array<double, 3>, std::array<double, 3>, std::array<double, 3>, std::array<double, 3>>())
            .def(py::init<double, std::array<double, 3>, std::array<double, 3>, std::array<double, 3>,
                          std::array<double, 3>, std::array<double, 3>, double, double>())
            .def_static("from_astrometry", &Observation::from_astrometry,
                        py::arg("ra"), py::arg("dec"), py::arg("epoch"),
                        py::arg("observer_position"), py::arg("observer_velocity"),
                        "Construct an Astrometry observation")
            .def_static("from_astrometry_with_id", &Observation::from_astrometry_with_id,
                        py::arg("objID"),
                        py::arg("ra"), py::arg("dec"), py::arg("epoch"),
                        py::arg("observer_position"), py::arg("observer_velocity"),
                        "Construct an Astrometry observation")
            // Constructor for a Streak observation.
            // bind the ::from_streak factory method
            .def_static("from_streak", &Observation::from_streak,
                        py::arg("ra"), py::arg("dec"), py::arg("ra_rate"), py::arg("dec_rate"),
                        py::arg("epoch"), py::arg("observer_position"), py::arg("observer_velocity"),
                        py::arg("ra_rate_unc") = DEFAULT_RATE_UNC_RAD, py::arg("dec_rate_unc") = DEFAULT_RATE_UNC_RAD,
                        "Construct a Streak observation")
            .def_static("from_streak_with_id", &Observation::from_streak_with_id,
                        py::arg("objID"),
                        py::arg("ra"), py::arg("dec"), py::arg("ra_rate"), py::arg("dec_rate"),
                        py::arg("epoch"), py::arg("observer_position"), py::arg("observer_velocity"),
                        py::arg("ra_rate_unc") = DEFAULT_RATE_UNC_RAD, py::arg("dec_rate_unc") = DEFAULT_RATE_UNC_RAD,
                        "Construct a Streak observation")
            // Constructor for a Radar (delay/Doppler) observation.
            .def_static("from_radar", &Observation::from_radar,
                        py::arg("delay"), py::arg("doppler"),
                        py::arg("has_delay"), py::arg("has_doppler"),
                        py::arg("epoch"), py::arg("observer_position"), py::arg("observer_velocity"),
                        py::arg("delay_unc") = 1.0, py::arg("doppler_unc") = 1.0,
                        py::arg("observer_acceleration") = std::array<double, 3>{{0.0, 0.0, 0.0}},
                        "Construct a Radar observation (delay in days, doppler in au/day)")
            .def_static("from_radar_with_id", &Observation::from_radar_with_id,
                        py::arg("objID"),
                        py::arg("delay"), py::arg("doppler"),
                        py::arg("has_delay"), py::arg("has_doppler"),
                        py::arg("epoch"), py::arg("observer_position"), py::arg("observer_velocity"),
                        py::arg("delay_unc") = 1.0, py::arg("doppler_unc") = 1.0,
                        py::arg("observer_acceleration") = std::array<double, 3>{{0.0, 0.0, 0.0}},
                        "Construct a Radar observation (delay in days, doppler in au/day)")
            .def_readwrite("epoch", &Observation::epoch, "Observation epoch (as a double)")
            .def_readwrite("observation_type", &Observation::observation_type, "Variant holding the observation data")
            .def_readwrite("observer_position", &Observation::observer_position, "Observer position as a 3D vector")
            .def_readwrite("observer_velocity", &Observation::observer_velocity, "Observer velocity as a 3D vector")
            .def_readwrite("observer_acceleration", &Observation::observer_acceleration, "Observer acceleration (au/day^2; radar two-leg model)")
            .def_readwrite("rho_hat", &Observation::rho_hat, "Unit direction vector")
            .def_readwrite("a_vec", &Observation::a_vec, "Tangent plane vector A")
            .def_readwrite("d_vec", &Observation::d_vec, "Tangent plane vector D")
            .def_readwrite("inverse_covariance", &Observation::inverse_covariance, "Optional inverse covariance matrix")
            .def_readwrite("ra_unc", &Observation::ra_unc, "RA uncertainty")
            .def_readwrite("dec_unc", &Observation::dec_unc, "Dec uncertainty")
            .def_readwrite("ra_rate_unc", &Observation::ra_rate_unc, "RA-rate uncertainty (rad/day)")
            .def_readwrite("dec_rate_unc", &Observation::dec_rate_unc, "Dec-rate uncertainty (rad/day)")
            .def_readwrite("delay_unc", &Observation::delay_unc, "Radar delay uncertainty (days)")
            .def_readwrite("doppler_unc", &Observation::doppler_unc, "Radar Doppler uncertainty (au/day)")
            .def_readwrite("mag", &Observation::mag, "Optional magnitude")
            .def_readwrite("mag_err", &Observation::mag_err, "Optional magnitude error");
    }

} // namespace orbit_fit
