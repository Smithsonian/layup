#include <variant>
#include <optional>
#include <array>
#include <Eigen/Dense>
#include <cmath>

// --- Observation Variant Types ---
// Each variant computes a unit direction vector (rho_hat) from the provided ra and dec.

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



// #include <variant>
// #include <optional>
// #include <Eigen/Dense>
// #include <cmath>

// // --- Observation Variant Types ---
// // Instead of storing ra/dec, each variant computes a unit direction vector (rho_hat)
// // from the provided ra and dec.

// struct AstrometryObservation {
//     Eigen::Vector3d rho_hat;  // Computed unit direction vector

//     AstrometryObservation() = default;
//     // Constructor: takes ra and dec (in radians) and computes rho_hat.
//     AstrometryObservation(double ra, double dec) {
//         rho_hat.x() = std::cos(dec) * std::cos(ra);
//         rho_hat.y() = std::cos(dec) * std::sin(ra);
//         rho_hat.z() = std::sin(dec);
//     }
// };

// struct StreakObservation {
//     double ra_rate;
//     double dec_rate;
//     Eigen::Vector3d rho_hat;

//     StreakObservation() = default;
//     // Note the extra tag is not needed here because it’s only for disambiguation at the Observation level.
//     StreakObservation(double ra, double dec, double ra_rate, double dec_rate)
//       : ra_rate(ra_rate), dec_rate(dec_rate)
//     {
//         rho_hat.x() = std::cos(dec) * std::cos(ra);
//         rho_hat.y() = std::cos(dec) * std::sin(ra);
//         rho_hat.z() = std::sin(dec);
//     }
// };

// using ObservationType = std::variant<AstrometryObservation, StreakObservation>;

// // --- Main Observation Structure ---
// // The Observation class now holds an epoch as a double and observer_position/velocity as 3D vectors.
// // We provide overloaded constructors
// struct Observation {
//     double epoch; // e.g., seconds since epoch
//     ObservationType observation_type;
//     std::array<double, 3> observer_position;
//     std::array<double, 3> observer_velocity;
//     std::optional<Eigen::MatrixXd> inverse_covariance;
//     std::optional<double> mag;
//     std::optional<double> mag_err;

//     // Constructor for an Astrometry observation.

//     // .from_astrometry
//     Observation(double ra, double dec, double epoch_val,
//                 const std::array<double, 3> &obs_position, const std::array<double, 3> &obs_velocity)
//       : epoch(epoch_val),
//         observer_position(obs_position),
//         observer_velocity(obs_velocity)
//     {
//        observation_type = AstrometryObservation(ra, dec);
//     }

//     // Constructor for a Streak observation (disambiguated with a StreakTag).

//     // .from_streak
//     Observation(double ra, double dec, double ra_rate, double dec_rate,
//                 double epoch_val, const std::array<double, 3> &obs_position, const std::array<double, 3> &obs_velocity)
//       : epoch(epoch_val),
//         observer_position(obs_position),
//         observer_velocity(obs_velocity)
//     {
//        observation_type = StreakObservation(ra, dec, ra_rate, dec_rate);
//     }
// };