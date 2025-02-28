#include <optional>
#include <variant>
#include <string>

struct Astrometry {
    double ra;
    double dec;
};

struct Streak {
    double ra;
    double dec;
    double ra_rate;
    double dec_rate;
};

struct Radar {
    double ra;
    double dec;
    double range;
    double range_rate;
};

struct Complete {
    double ra;
    double dec;
    double ra_rate;
    double dec_rate;
    double range;
    double range_rate;
};

// Define the Observation type as a variant of the above types.
using ObservationVariant = std::variant<Astrometry, Streak, Radar, Complete>;