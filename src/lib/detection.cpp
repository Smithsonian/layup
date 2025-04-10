#include <variant>
#include <optional>
#include <array>
#include <Eigen/Dense>
#include <cmath>

#include <pybind11/pybind11.h>
namespace py = pybind11;

// --- Observation Variant Types ---
// Each variant computes a unit direction vector (rho_hat) from the provided ra and dec.

namespace orbit_fit {

struct AstrometryObservation {
    Eigen::Vector3d rho_hat;  // Computed unit direction vector

    AstrometryObservation() = default;
    // Constructor: takes ra and dec (in radians) and computes rho_hat.
    AstrometryObservation(double ra, double dec) {
        rho_hat.x() = std::cos(dec) * std::cos(ra);
        rho_hat.y() = std::cos(dec) * std::sin(ra);
        rho_hat.z() = std::sin(dec);
    }
};

struct StreakObservation {
    double ra_rate;
    double dec_rate;
    Eigen::Vector3d rho_hat;

    StreakObservation() = default;
    // Constructor: takes ra, dec and the corresponding rates, and computes rho_hat.
    StreakObservation(double ra, double dec, double ra_rate, double dec_rate)
      : ra_rate(ra_rate), dec_rate(dec_rate)
    {
        rho_hat.x() = std::cos(dec) * std::cos(ra);
        rho_hat.y() = std::cos(dec) * std::sin(ra);
        rho_hat.z() = std::sin(dec);
    }
};

using ObservationType = std::variant<AstrometryObservation, StreakObservation>;

// --- Main Observation Structure ---
// Now, Observation has private constructors and public static factory methods.
struct Observation {
    double epoch; // utc jd
    ObservationType observation_type;
    std::array<double, 3> observer_position;
    std::array<double, 3> observer_velocity;
    std::optional<Eigen::MatrixXd> inverse_covariance;
    std::optional<double> mag;
    std::optional<double> mag_err;
    std::optional<double> epoch_err;
    std::optional<double> ra_unc;
    std::optional<double> dec_unc;

private:
    // Private constructor used by the factory methods.
    Observation(double epoch_val,
                const std::array<double, 3>& obs_position,
                const std::array<double, 3>& obs_velocity)
      : epoch(epoch_val),
        observer_position(obs_position),
        observer_velocity(obs_velocity)
    {}

public:
    // Factory method for an Astrometry observation.
    static Observation from_astrometry(double ra, double dec, double epoch_val,
                                       const std::array<double, 3>& obs_position,
                                       const std::array<double, 3>& obs_velocity)
    {
        Observation obs(epoch_val, obs_position, obs_velocity);
        obs.observation_type = AstrometryObservation(ra, dec);
        obs.ra_unc = 1.0;
        obs.dec_unc = 1.0;
        return obs;
    }

    // Factory method for a Streak observation.
    static Observation from_streak(double ra, double dec, double ra_rate, double dec_rate,
                                   double epoch_val,
                                   const std::array<double, 3>& obs_position,
                                   const std::array<double, 3>& obs_velocity)
    {
        Observation obs(epoch_val, obs_position, obs_velocity);
        obs.observation_type = StreakObservation(ra, dec, ra_rate, dec_rate);
        return obs;
    }

    // Factory method for a Radar observation.
    // not implemented yet

};


static void detection_bindings(py::module& m) {
    py::class_<AstrometryObservation>(m, "AstrometryObservation")
        .def(py::init<double, double>(),
             py::arg("ra"), py::arg("dec"))
        .def_readonly("rho_hat", &AstrometryObservation::rho_hat,
                      "Computed unit direction vector (rho_hat)");


    // Bind StreakObservation type.
    py::class_<StreakObservation>(m, "StreakObservation")
        .def(py::init<double, double, double, double>(),
             py::arg("ra"), py::arg("dec"), py::arg("ra_rate"), py::arg("dec_rate"))
        .def_readonly("ra_rate", &StreakObservation::ra_rate)
        .def_readonly("dec_rate", &StreakObservation::dec_rate)
        .def_readonly("rho_hat", &StreakObservation::rho_hat,
                      "Computed unit direction vector (rho_hat)");

    // Bind the main Observation type with multiple overloaded constructors.
    py::class_<Observation>(m, "Observation")
        // Constructor for an Astrometry observation.
        // bind the ::from_astrometry factory method
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
        .def_readonly("epoch", &Observation::epoch, "Observation epoch (as a double)")
        .def_readonly("observation_type", &Observation::observation_type, "Variant holding the observation data")
        .def_readonly("observer_position", &Observation::observer_position, "Observer position as a 3D vector")
        .def_readonly("observer_velocity", &Observation::observer_velocity, "Observer velocity as a 3D vector")
        .def_readonly("inverse_covariance", &Observation::inverse_covariance, "Optional inverse covariance matrix")
        .def_readonly("mag", &Observation::mag, "Optional magnitude")
        .def_readonly("mag_err", &Observation::mag_err, "Optional magnitude error");
}

} //namespace orbit_fit
