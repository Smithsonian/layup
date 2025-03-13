#include <variant>
#include <optional>
#include <Eigen/Dense>
#include <cmath>

// --- Observation Variant Types ---
// Instead of storing ra/dec, each variant computes a unit direction vector (rho_hat)
// from the provided ra and dec.

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
    // Note the extra tag is not needed here because it’s only for disambiguation at the Observation level.
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
// The Observation class now holds an epoch as a double and observer_position/velocity as 3D vectors.
// We provide overloaded constructors, distinguished by extra tag parameters.
struct Observation {
    double epoch; // e.g., seconds since epoch
    ObservationType observation_type;
    Eigen::Vector3d observer_position;
    Eigen::Vector3d observer_velocity;
    std::optional<Eigen::MatrixXd> inverse_covariance;
    std::optional<double> mag;
    std::optional<double> mag_err;

    // Constructor for an Astrometry observation.
    Observation(double ra, double dec, double epoch_val,
                const Eigen::Vector3d &obs_position, const Eigen::Vector3d &obs_velocity)
      : epoch(epoch_val),
        observer_position(obs_position),
        observer_velocity(obs_velocity)
    {
       observation_type = AstrometryObservation(ra, dec);
    }

    // Constructor for a Streak observation (disambiguated with a StreakTag).
    Observation(double ra, double dec, double ra_rate, double dec_rate,
                double epoch_val, const Eigen::Vector3d &obs_position, const Eigen::Vector3d &obs_velocity)
      : epoch(epoch_val),
        observer_position(obs_position),
        observer_velocity(obs_velocity)
    {
       observation_type = StreakObservation(ra, dec, ra_rate, dec_rate);
    }
};