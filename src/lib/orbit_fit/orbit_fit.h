
struct detection {

    detection() : jd_tdb(), theta_x(), theta_y(), theta_z(),
		  ra_unc(), dec_unc(), mag(), mag_unc() {
    }

    std::string objID; // user-supplied object ID.
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

