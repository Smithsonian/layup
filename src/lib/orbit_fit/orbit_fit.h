
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

struct residuals
{

    residuals() : x_resid(), y_resid(), ra_rate_resid(), dec_rate_resid(), n_resid(2)
    {
    }
    double x_resid;
    double y_resid;
    // Streak (sky-motion-rate) residuals. Only populated, and only counted in
    // the LM normal equations, when n_resid == 4 (a StreakObservation).
    double ra_rate_resid;
    double dec_rate_resid;
    int n_resid; // number of residual rows this observation contributes (2 or 4)
};

struct partials
{

    partials() : x_partials(), y_partials(), ra_rate_partials(), dec_rate_partials(), n_resid(2)
    {
    }

    std::vector<double> x_partials;
    std::vector<double> y_partials;
    // Partials of the rate residuals w.r.t. the 6 state parameters (streaks only).
    std::vector<double> ra_rate_partials;
    std::vector<double> dec_rate_partials;
    int n_resid; // 2 (astrometry) or 4 (streak); mirrors residuals::n_resid
};
