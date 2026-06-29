
struct detection
{

    detection() : jd_tdb(), theta_x(), theta_y(), theta_z(),
                  ra_unc(), dec_unc(), mag(), mag_unc()
    {
    }

    std::string objID;   // user-supplied object ID.
    std::string obsCode; // observatory code

    double jd_tdb;

    double theta_x; // unit vector x
    double theta_y; // unit vector y
    double theta_z; // unit vector z

    double Ax; // unit vector along RA
    double Ay;
    double Az;

    double Dx; // unit vector along Dec
    double Dy;
    double Dz;

    double xe; // observatory position x
    double ye; // observatory position y
    double ze; // observatory position z

    double ra_unc;  // astrometric uncertainty (radians)
    double dec_unc; // astrometric uncertainty (radians)

    double mag;     // magnitude
    double mag_unc; // magnitude uncertainty
};

// Discriminates the residual layout an observation contributes. OPTICAL covers
// both astrometry (2 rows) and streak (4 rows); RADAR covers delay/Doppler
// (1 or 2 rows). The LM assembly (compute_dX) only sees the partials struct, so
// the kind/flags travel with it rather than the Observation.
enum ObsKind
{
    OPTICAL = 0,
    RADAR = 1
};

struct residuals
{

    residuals() : x_resid(), y_resid(), ra_rate_resid(), dec_rate_resid(),
                  delay_resid(), doppler_resid(), n_resid(2), obs_kind(OPTICAL),
                  has_delay(false), has_doppler(false)
    {
    }
    double x_resid;
    double y_resid;
    // Streak (sky-motion-rate) residuals. Only populated, and only counted in
    // the LM normal equations, when n_resid == 4 (a StreakObservation).
    double ra_rate_resid;
    double dec_rate_resid;
    // Radar residuals (RADAR only): delay in days, doppler in au/day (round-trip).
    double delay_resid;
    double doppler_resid;
    int n_resid; // number of residual rows this observation contributes
    int obs_kind;     // ObsKind: OPTICAL or RADAR
    bool has_delay;   // RADAR only: a delay row is present
    bool has_doppler; // RADAR only: a doppler row is present
};

struct partials
{

    partials() : x_partials(), y_partials(), ra_rate_partials(), dec_rate_partials(),
                 delay_partials(), doppler_partials(), n_resid(2), obs_kind(OPTICAL),
                 has_delay(false), has_doppler(false)
    {
    }

    std::vector<double> x_partials;
    std::vector<double> y_partials;
    // Partials of the rate residuals w.r.t. the 6 state parameters (streaks only).
    std::vector<double> ra_rate_partials;
    std::vector<double> dec_rate_partials;
    // Partials of the radar residuals w.r.t. the 6 state parameters (RADAR only).
    std::vector<double> delay_partials;
    std::vector<double> doppler_partials;
    int n_resid;      // rows this observation contributes; mirrors residuals::n_resid
    int obs_kind;     // ObsKind: OPTICAL or RADAR
    bool has_delay;   // RADAR only: a delay row is present
    bool has_doppler; // RADAR only: a doppler row is present
};
