
struct detection {

    detection() = default;

    std::string objID; // user-supplied object ID.
    std::string obsCode; // observatory code
    
    double jd_tdb;

    std::array<double, 3> rho_hat;    

    std::array<double, 3> A;
    std::array<double, 3> D;

    std::array<double, 3> r_e;

    double ra_unc; // astrometric uncertainty (radians)
    double dec_unc; // astrometric uncertainty (radians)

    double mag; // magnitude 
    double mag_unc; // magnitude uncertainty

};

struct residuals {

    residuals() : x_resid(), y_resid() {
    }
    double x_resid;
    double y_resid;
};


struct partials {

    partials() : x_partials(), y_partials() {
    }
    
    std::vector<double> x_partials;
    std::vector<double> y_partials;
};

