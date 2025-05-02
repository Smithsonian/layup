#include <variant>
#include <optional>
#include <array>
#include <Eigen/Dense>
#include <cmath>

#include <pybind11/pybind11.h>
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

    Eigen::Vector3d a_vec_from_rho_hat(const Eigen::Vector3d &rho_hat)
    {
        Eigen::Vector3d a_vec;
        a_vec.x() = rho_hat.z();
        a_vec.y() = 0.0;
        a_vec.z() = -rho_hat.x();
        return a_vec;
    }

    Eigen::Vector3d d_vec_from_rho_hat(const Eigen::Vector3d &rho_hat)
    {
        Eigen::Vector3d d_vec;
        d_vec.x() = -rho_hat.x() * rho_hat.y();
        d_vec.y() = rho_hat.x() * rho_hat.x() + rho_hat.z() * rho_hat.z();
        d_vec.z() = -rho_hat.z() * rho_hat.y();
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
        StreakObservation(double ra_rate, double dec_rate) : ra_rate(ra_rate), dec_rate(dec_rate){}
    };

    using ObservationType = std::variant<AstrometryObservation, StreakObservation>;

    // --- Main Observation Structure ---
    // Now, Observation has private constructors and public static factory methods.
    struct Observation
    {
        double epoch; // utc jd
        ObservationType observation_type;
        std::array<double, 3> observer_position;
        std::array<double, 3> observer_velocity;

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
        Observation(
            double ep,
            std::array<double, 3> obs_position,
            std::array<double, 3> obs_velocity,
            std::array<double, 3> rho,
            std::array<double, 3> avec,
            std::array<double, 3> dvec)
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

            ra_unc = 1.0 / 206265;
            dec_unc = 1.0 / 206265;
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

        Observation(){}

        // Factory method for an Astrometry observation.
        static Observation from_astrometry(double ra, double dec, double epoch_val,
                                           const std::array<double, 3> &obs_position,
                                           const std::array<double, 3> &obs_velocity)
        {
            Observation obs(epoch_val, obs_position, obs_velocity);
            obs.observation_type = AstrometryObservation();
            obs.ra_unc = 1.0/206265;
            obs.dec_unc = 1.0/206265;
            obs.rho_hat = rho_hat_from_ra_dec(ra, dec);
            obs.a_vec = a_vec_from_rho_hat(obs.rho_hat);
            obs.d_vec = d_vec_from_rho_hat(obs.rho_hat);
            return obs;
        }

        // Factory method for a Streak observation.
        static Observation from_streak(double ra, double dec, double ra_rate, double dec_rate,
                                       double epoch_val,
                                       const std::array<double, 3> &obs_position,
                                       const std::array<double, 3> &obs_velocity)
        {
            Observation obs(epoch_val, obs_position, obs_velocity);
            obs.observation_type = StreakObservation(ra_rate, dec_rate);
            obs.rho_hat = rho_hat_from_ra_dec(ra, dec);
            obs.a_vec = a_vec_from_rho_hat(obs.rho_hat);
            obs.d_vec = d_vec_from_rho_hat(obs.rho_hat);
            obs.ra_unc = 1.0/206265;
            obs.dec_unc = 1.0/206265;
            return obs;
        }

        // Factory method for a Radar observation.
        // not implemented yet
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
            // Constructor for a Streak observation.
            // bind the ::from_streak factory method
            .def_static("from_streak", &Observation::from_streak,
                        py::arg("ra"), py::arg("dec"), py::arg("ra_rate"), py::arg("dec_rate"),
                        py::arg("epoch"), py::arg("observer_position"), py::arg("observer_velocity"),
                        "Construct a Streak observation")
            .def_readwrite("epoch", &Observation::epoch, "Observation epoch (as a double)")
            .def_readwrite("observation_type", &Observation::observation_type, "Variant holding the observation data")
            .def_readwrite("observer_position", &Observation::observer_position, "Observer position as a 3D vector")
            .def_readwrite("observer_velocity", &Observation::observer_velocity, "Observer velocity as a 3D vector")
            .def_readwrite("rho_hat", &Observation::rho_hat, "Unit direction vector")
            .def_readwrite("a_vec", &Observation::a_vec, "Tangent plane vector A")
            .def_readwrite("d_vec", &Observation::d_vec, "Tangent plane vector D")
            .def_readwrite("inverse_covariance", &Observation::inverse_covariance, "Optional inverse covariance matrix")
            .def_readwrite("ra_unc", &Observation::ra_unc, "RA uncertainty")
            .def_readwrite("dec_unc", &Observation::dec_unc, "Dec uncertainty")
            .def_readwrite("mag", &Observation::mag, "Optional magnitude")
            .def_readwrite("mag_err", &Observation::mag_err, "Optional magnitude error");
    }

} // namespace orbit_fit
