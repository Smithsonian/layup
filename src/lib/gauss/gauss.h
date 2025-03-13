struct gauss_soln {

    gauss_soln() : root(), epoch(), x(), y(), z(), vx(), vy(), vz(){
    }

    double root; 
    double epoch;

    double x; 
    double y;
    double z;

    double vx; 
    double vy;
    double vz;

    gauss_soln(double _root, double _epoch, double _x, double _y, double _z, double _vx, double _vy, double _vz) {
	root = _root;
	epoch = _epoch;
	x = _x;
	y = _y;
	z = _z;
	vx = _vx;
	vy = _vy;
	vz = _vz;
    }    

};

std::optional<std::vector<gauss_soln>> gauss(double MU_BARY, detection &o1_in, detection &o2_in, detection &o3_in, double min_distance, double SPEED_OF_LIGHT);
