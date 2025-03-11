// This code requires the following:
// 1. The rebound library
// 2. The assist library
// 3. The Eigen header-only library
//
// It will probably need either pybind11, eigenpy, or something like
// that to connect it to python

// To Do:
// 1. Set up a make file, or something like that.
// 2. Make sure that the physical constants are consistent
//    with the rest of the code
// 3. Work on getting better data weights.
// 4. Work on outlier rejection.
// 5. Write code to handle doppler and range observations
// 6. Write code to handle shift+stack observations (RA/Dec + rates)

// Compile with something like this
// g++ -std=c++11 -I../..//src/ -I../../../rebound/src/ -I/Users/mholman/eigen-3.4.0 -Wl,-rpath,./ -Wpointer-arith -D_GNU_SOURCE -O3 -fPIC -I/usr/local/include -Wall -g  -Wno-unknown-pragmas -D_APPLE -DSERVER -DGITHASH=e99d8d73f0aa7fb7bf150c680a2d581f43d3a8be orbit_fit.cpp -L. -lassist -lrebound -L/usr/local/lib -o orbit_fit

#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>

extern "C"{
#include "rebound.h"
#include "assist.h"
}
using namespace Eigen;
using std::cout;

double AU_M = 149597870700;
double SPEED_OF_LIGHT = 2.99792458e8 * 86400.0 / AU_M;

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

struct cart_orbit {

    cart_orbit() : epoch(), x(), y(), z(), vx(), vy(), vz() {
    }

    double epoch;

    double x; 
    double y;
    double z;

    double vx; 
    double vy;
    double vz;

    cart_orbit(double _epoch, double _x, double _y, double _z, double _vx, double _vy, double _vz) { 
	epoch = _epoch;
	x = _x;
	y = _y;
	z = _z;
	vx = _vx;
	vy = _vy;
	vz = _vz;
    }    

};

// template for gauss
// pass in three detections
std::optional<std::vector<cart_orbit>> gauss(double MU_BARY, detection &o1_in, detection &o2_in, detection &o3_in, double min_distance) {
    // Create a vector of pointers to observations for sorting by epoch
    std::vector<detection> triplet = { o1_in, o2_in, o3_in };
    std::sort(triplet.begin(), triplet.end(), [](const detection a, const detection b) {
        return a.jd_tdb < b.jd_tdb;
    });

    // Get the observer positions and pointing directions
    Eigen::Vector3d o1 = {triplet[0].xe, triplet[0].ye, triplet[0].ze};
    Eigen::Vector3d o2 = {triplet[1].xe, triplet[1].ye, triplet[1].ze};
    Eigen::Vector3d o3 = {triplet[2].xe, triplet[2].ye, triplet[2].ze};        

    Eigen::Vector3d rho1 = {triplet[0].theta_x, triplet[0].theta_y, triplet[0].theta_z};
    Eigen::Vector3d rho2 = {triplet[1].theta_x, triplet[1].theta_y, triplet[1].theta_z};
    Eigen::Vector3d rho3 = {triplet[2].theta_x, triplet[2].theta_y, triplet[2].theta_z};        

    double t1 = triplet[0].jd_tdb;
    double t2 = triplet[1].jd_tdb;
    double t3 = triplet[2].jd_tdb;

    double tau1 = t1 - t2;
    double tau3 = t3 - t2;
    double tau  = t3 - t1;

    // Compute cross products
    Eigen::Vector3d p1 = rho2.cross(rho3);
    Eigen::Vector3d p2 = rho1.cross(rho3);
    Eigen::Vector3d p3 = rho1.cross(rho2);

    double d0 = rho1.dot(p1);

    // Construct the 3x3 d matrix
    Eigen::Matrix3d d;
    d(0, 0) = o1.dot(p1); d(0, 1) = o1.dot(p2); d(0, 2) = o1.dot(p3);
    d(1, 0) = o2.dot(p1); d(1, 1) = o2.dot(p2); d(1, 2) = o2.dot(p3);
    d(2, 0) = o3.dot(p1); d(2, 1) = o3.dot(p2); d(2, 2) = o3.dot(p3);

    double a = (1.0 / d0) * (-d(0, 1) * (tau3 / tau) + d(1, 1) + d(2, 1) * (tau1 / tau));
    double b = (1.0 / (6.0 * d0)) * (d(0, 1) * ((tau3 * tau3) - (tau * tau)) * (tau3 / tau) +
                                     d(2, 1) * ((tau * tau) - (tau1 * tau1)) * (tau1 / tau));
    double e = o2.dot(rho2);
    double o2sq = o2.dot(o2);

    double aa = -((a * a) + 2.0 * a * e + o2sq);
    double bb = -2.0 * MU_BARY * b * (a + e);
    double cc = -std::pow(MU_BARY, 2) * (b * b);

    // Construct the 8x8 companion matrix for the polynomial
    Eigen::Matrix<double, 8, 8> mat = Eigen::Matrix<double, 8, 8>::Zero();
    mat(0, 1) = 1.0;
    mat(1, 2) = 1.0;
    mat(2, 3) = 1.0;
    mat(3, 4) = 1.0;
    mat(4, 5) = 1.0;
    mat(5, 6) = 1.0;
    mat(6, 7) = 1.0;
    mat(7, 0) = -cc;
    mat(7, 3) = -bb;
    mat(7, 6) = -aa;

    // Compute the eigenvalues of the companion matrix
    Eigen::ComplexEigenSolver<Eigen::Matrix<double, 8, 8>> ces(mat);
    if (ces.info() != Eigen::Success) {
        return std::nullopt;
    }

    // Filter eigenvalues: select those with (nearly) zero imaginary part and real part > min_distance
    std::vector<double> roots;
    for (int i = 0; i < ces.eigenvalues().size(); ++i) {
        std::complex<double> eig = ces.eigenvalues()[i];
        if (std::abs(eig.imag()) < 1e-10 && eig.real() > min_distance) {
            roots.push_back(eig.real());
        }
    }
    if (roots.empty()) {
        return std::nullopt;
    }

    std::vector<cart_orbit> res;
    for (double root : roots) {
        double root3 = std::pow(root, 3);

        // Compute a1
        double num1 = 6.0 * (d(2, 0) * (tau1 / tau3) + d(1, 0) * (tau / tau3)) * root3 +
                      MU_BARY * d(2, 0) * ((tau * tau) - (tau1 * tau1)) * (tau1 / tau3);
        double den1 = 6.0 * root3 + MU_BARY * ((tau * tau) - (tau3 * tau3));
        double a1 = (1.0 / d0) * ((num1 / den1) - d(0, 0));

        // Compute a2 and a3
        double a2 = a + (MU_BARY * b) / root3;
        double num3 = 6.0 * (d(0, 2) * (tau3 / tau1) - d(1, 2) * (tau / tau1)) * root3 +
                      MU_BARY * d(0, 2) * ((tau * tau) - (tau3 * tau3)) * (tau3 / tau1);
        double den3 = 6.0 * root3 + MU_BARY * ((tau * tau) - (tau1 * tau1));
        double a3 = (1.0 / d0) * ((num3 / den3) - d(2, 2));

        // Compute position vectors
        Eigen::Vector3d r1 = o1 + a1 * rho1;
        Eigen::Vector3d r2 = o2 + a2 * rho2;
        Eigen::Vector3d r3 = o3 + a3 * rho3;

        // Calculate f and g functions
        double f1 = 1.0 - 0.5 * (MU_BARY / root3) * (tau1 * tau1);
        double f3 = 1.0 - 0.5 * (MU_BARY / root3) * (tau3 * tau3);
        double g1 = tau1 - (1.0 / 6.0) * (MU_BARY / root3) * (tau1 * tau1 * tau1);
        double g3 = tau3 - (1.0 / 6.0) * (MU_BARY / root3) * (tau3 * tau3 * tau3);

        // Solve for the velocity at t2
        Eigen::Vector3d v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1);

        // Extract components for clarity
        double x  = r2.x();
        double y  = r2.y();
        double z  = r2.z();
        double vx = v2.x();
        double vy = v2.y();
        double vz = v2.z();

        // Apply light-time correction
        double ltt = r2.norm() / SPEED_OF_LIGHT;
        double corrected_t = triplet[1].jd_tdb;
        corrected_t -= ltt;

	cart_orbit rock = cart_orbit(x, y, z, vx, vy, vz, corrected_t);
	res.push_back(rock);

    }

    return res;
}

// Does this need to report possible failures?
int integrate_light_time(struct assist_extras* ax, int np, double t, reb_vec3d r_obs, double lt0, size_t iter, double speed_of_light){

    struct reb_simulation* r = ax->sim;    
    double lt = lt0;
    // Performs the light travel time correction between object and observatory iteratively for the object at a given reference time    
    for (int i=0; i<iter; i++){
        assist_integrate_or_interpolate(ax, t - lt);
        double dx = r->particles[np].x - r_obs.x;
        double dy = r->particles[np].y - r_obs.y;
        double dz = r->particles[np].z - r_obs.z;
	double rho_mag = sqrt(dx*dx + dy*dy + dz*dz);
        lt = rho_mag / speed_of_light;
    }

    return 0;

}

struct reb_particle read_initial_conditions(const char *ic_file_name, double *epoch){

    FILE *ic_file;    
    ic_file = fopen(ic_file_name, "r");

    struct reb_particle p0;

    double epch;
    double x0, y0, z0;
    double vx0, vy0, vz0;
    
    fscanf(ic_file, "%lf %lf %lf %lf %lf %lf %lf\n",
	   &epch, &x0, &y0, &z0, &vx0, &vy0, &vz0);

    *epoch = epch;
    p0.x = x0;
    p0.y = y0;
    p0.z = z0;
    p0.vx = vx0;
    p0.vy = vy0;
    p0.vz = vz0;

    return p0;

}

void read_detections(const char *data_file_name,
		     std::vector<detection>& detections,
		     std::vector<double>& times		     
		     ){

    FILE *detections_file;    
    detections_file = fopen(data_file_name, "r");

    char obsCode[10];
    char objID[20];
    char mag_str[6];
    char filt[6];
    double theta_x, theta_y, theta_z;
    double xe, ye, ze;
    double jd_tdb;
    double ast_unc;

    // Define an observation structure or class
    // Write a function to read observations
    
    while(fscanf(detections_file, "%s %s %s %s %lf %lf %lf %lf %lf %lf %lf %lf\n",
		 objID, obsCode, mag_str, filt, &jd_tdb, &theta_x, &theta_y, &theta_z, &xe, &ye, &ze, &ast_unc) !=EOF){
	detection this_det = detection();

	this_det.jd_tdb = jd_tdb;	
	this_det.theta_x = theta_x;
	this_det.theta_y = theta_y;
	this_det.theta_z = theta_z;
	
	this_det.xe = xe;
	this_det.ye = ye;
	this_det.ze = ze;
	
	this_det.obsCode = obsCode;

	// Read these from the file
	this_det.ra_unc = (ast_unc/206265);
	this_det.dec_unc = (ast_unc/206265);	

	// compute A and D vectors, the local tangent plane trick from Danby.
		  
	double Ax =  theta_z;
	double Ay =  0.0;
	double Az = -theta_x;

	double A = sqrt(Ax*Ax + Ay*Ay + Az*Az);
	Ax /= A; Ay /= A; Az /= A;

	this_det.Ax = Ax;
	this_det.Ay = Ay;
	this_det.Az = Az;	

	double Dx = -theta_x*theta_y;
	double Dy =  theta_x*theta_x + theta_z*theta_z;
	double Dz = -theta_z*theta_y;
		  
	double D = sqrt(Dx*Dx + Dy*Dy + Dz*Dz);
	Dx /= D; Dy /= D; Dz /= D;

	this_det.Dx = Dx;
	this_det.Dy = Dy;
	this_det.Dz = Dz;	

	detections.push_back(this_det);
	times.push_back(jd_tdb);

    }

}

void add_variational_particles(struct reb_simulation* r, size_t np, int *var){

    int varx = reb_simulation_add_variation_1st_order(r, np);
    r->particles[varx].x = 1.;

    int vary = reb_simulation_add_variation_1st_order(r, np);
    r->particles[vary].y = 1.;

    int varz = reb_simulation_add_variation_1st_order(r, np);
    r->particles[varz].z = 1.;

    int varvx = reb_simulation_add_variation_1st_order(r, np);
    r->particles[varvx].vx = 1.;

    int varvy = reb_simulation_add_variation_1st_order(r, np);
    r->particles[varvy].vy = 1.;

    int varvz = reb_simulation_add_variation_1st_order(r, np);
    r->particles[varvz].vz = 1.;

    *var = varx;
    
}

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

void compute_single_residuals(struct assist_ephem* ephem,
			      struct assist_extras* ax,
			      int var,
			      detection this_det,
			      residuals& resid,
			      partials& parts
			      ){

    // Takes a detection/observation and
    // a simulation as arguments. 
    // Returns the model observation, the
    // residuals, and the partial derivatives.

    struct reb_simulation* r = ax->sim;        
    
    double jd_tdb = this_det.jd_tdb;

    double xe = this_det.xe;
    double ye = this_det.ye;
    double ze = this_det.ze;

    double Ax = this_det.Ax;
    double Ay = this_det.Ay;
    double Az = this_det.Az;

    double Dx = this_det.Dx;
    double Dy = this_det.Dy;
    double Dz = this_det.Dz;

    // 5. compare the model result to the observation.
    //   This means dotting the model unit vector with the
    //   A and D vectors of the observation

    reb_vec3d r_obs = {xe, ye, ze};
    double t_obs = jd_tdb - ephem->jd_ref;

    int j = 0;
    // Make the number of iterations flexible
    // Could make integrate_light_time return
    // rho and its components, since those are
    // already computed for the light time iteration.
    integrate_light_time(ax, j, t_obs, r_obs, 0.0, 4, SPEED_OF_LIGHT);

    double rho_x = r->particles[j].x - xe;
    double rho_y = r->particles[j].y - ye;
    double rho_z = r->particles[j].z - ze;
    double rho = sqrt(rho_x*rho_x + rho_y*rho_y + rho_z*rho_z);

    rho_x /= rho;
    rho_y /= rho;
    rho_z /= rho;

    // Put the residuals into a struct, so that the
    // struct can be easily managed in a vector.
    resid.x_resid = -(rho_x*Ax + rho_y*Ay + rho_z*Az);
    resid.y_resid = -(rho_x*Dx + rho_y*Dy + rho_z*Dz);

    // 6. Calculate the partial deriviatives of the model
    //    observations with respect to the initial conditions

    double dxk[6], dyk[6], dzk[6];
    double drho_x[6], drho_y[6], drho_z[6];	

    for(size_t i = 0; i<6; i++){
	size_t vn = var + i;
	dxk[i] = r->particles[vn].x;
	dyk[i] = r->particles[vn].y;
	dzk[i] = r->particles[vn].z;
    }

    double invd = 1./rho;

    double ddist[6];
    double dx_resid[6];
    double dy_resid[6];	
    for (size_t i=0; i<6; i++) {

	// Derivative of topocentric distance w.r.t. parameters 
	ddist[i] = rho_x*dxk[i] + rho_y*dyk[i] + rho_z*dzk[i];

	drho_x[i] = dxk[i]*invd - rho_x*invd*ddist[i] - r->particles[0].vx*ddist[i]*invd/SPEED_OF_LIGHT;
	drho_y[i] = dyk[i]*invd - rho_y*invd*ddist[i] - r->particles[0].vy*ddist[i]*invd/SPEED_OF_LIGHT;
	drho_z[i] = dzk[i]*invd - rho_z*invd*ddist[i] - r->particles[0].vz*ddist[i]*invd/SPEED_OF_LIGHT;

	dx_resid[i] = -(drho_x[i]*Ax + drho_y[i]*Ay + drho_z[i]*Az);
	dy_resid[i] = -(drho_x[i]*Dx + drho_y[i]*Dy + drho_z[i]*Dz);

    }

    // Likewise, put the partials into a struct so that they can be more easily
    // managed.

    for (size_t i=0; i<6; i++) {
	parts.x_partials.push_back(dx_resid[i]);
    }
    for (size_t i=0; i<6; i++) {
	parts.y_partials.push_back(dy_resid[i]);
    }

}

// This routine requires that resid_vec and partials_vec are
// preallocated because the indices in out_seq will be used for
// random access of locations in those vectors.
// Also, in_seq and out_seq must be of the same length.
void compute_residuals_sequence(struct assist_ephem* ephem,
				struct reb_particle p0, double epoch,
				std::vector<detection>& detections,
				std::vector<residuals>& resid_vec,
				std::vector<partials>& partials_vec,
				std::vector<size_t>& in_seq,
				std::vector<size_t>& out_seq
				){

    // Pass in this simulation stuff to keep it flexible
    struct reb_simulation* r = reb_simulation_create();
    struct assist_extras* ax = assist_attach(r, ephem);    

    // 0. Set initial time, relative to ephem->jd_ref
    r->t = epoch - ephem->jd_ref;

    // 1. Add the main particle to the REBOUND simulation.
    reb_simulation_add(r, p0);

    // 2. incorporate the variational particles
    int var;
    add_variational_particles(r, 0, &var);

    // 3. iterate over a sequence of detections.
    for(size_t i = 0; i<in_seq.size();  ++i){    

	size_t j = in_seq[i];
	size_t k = out_seq[i];

	residuals resids;
	partials parts;
	detection this_det = detections[j];
	compute_single_residuals(ephem, ax, var,
				 this_det,
				 resids,
				 parts);
	resid_vec[k] = resids;
	partials_vec[k] = parts;
	
    }

    assist_free(ax);
    reb_simulation_free(r);

    return;
    
}

void compute_residuals(struct assist_ephem* ephem,
			  struct reb_particle p0, double epoch,
			  std::vector<detection>& detections,
			  std::vector<residuals>& resid_vec,
			  std::vector<partials>& partials_vec,
			  std::vector<size_t>& forward_in_seq,
			  std::vector<size_t>& forward_out_seq,
			  std::vector<size_t>& reverse_in_seq,
			  std::vector<size_t>& reverse_out_seq
		       ){

    compute_residuals_sequence(ephem, p0, epoch,
			       detections,
			       resid_vec,
			       partials_vec,
			       forward_in_seq,
			       forward_out_seq);

    compute_residuals_sequence(ephem, p0, epoch,
			       detections,
			       resid_vec,
			       partials_vec,
			       reverse_in_seq,
			       reverse_out_seq);
    
}

    
// Consider having this accept Eigen matrices as
// input
void compute_dX(std::vector<residuals>& resid_vec,
		std::vector<partials>& partials_vec,
		Eigen::SparseMatrix<double> W,
		Eigen::MatrixXd& dX,
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& chi2){

    // The block of code from here to the calculation of
    // dX should be a separate function

    const int mlength = (int) partials_vec.size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
    B.resize(mlength*2, 6);

    for (size_t i=0; i<partials_vec.size(); i++) {
	for(size_t j=0; j<6; j++){
	    B(2*i  , j) = partials_vec[i].x_partials[j];
	    B(2*i+1, j) = partials_vec[i].y_partials[j];	    
	}
    }

    Eigen::MatrixXd resid_v(mlength*2, 1);
    for (size_t i=0; i<resid_vec.size(); i++) {
	resid_v(2*i)   = resid_vec[i].x_resid;
	resid_v(2*i+1) = resid_vec[i].y_resid;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Bt = B.transpose();
    C = Bt * W * B;
    
    dX = C.colPivHouseholderQr().solve(-Bt * W * resid_v);

    chi2 = resid_v.transpose() * W * resid_v;

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G = C.inverse();
    //dX = -G *  Bt * W * resid_v;    
    
}

void identify_outliers(std::vector<detection>& detections,
		       std::vector<residuals>& resid_vec,
		       std::vector<bool>& reject){
}

void print_residuals(std::vector<detection>& detections,		     
		     std::vector<residuals>& resid_vec,
		     std::vector<partials>& partials_vec){

    for(size_t j=0; j<resid_vec.size(); j++){

	detection this_det = detections[j];

	printf("%lf %s ", this_det.jd_tdb, this_det.obsCode.c_str());

	residuals resids = resid_vec[j];
	printf("%7.3lf %7.3lf ", resids.x_resid*206265, resids.y_resid*206265);

	partials parts = partials_vec[j];	

	for (size_t i=0; i<6; i++) {

	    printf("%7.3le %7.3le ", parts.x_partials[i], parts.y_partials[i]);

	}

	printf("%7.3le %7.3le ", this_det.ra_unc*206265, this_det.dec_unc*206265);
	

	printf("\n");

    }
}

int converged(Eigen::MatrixXd dX, double eps){

    for(size_t i = 0; i<dX.size(); i++){
	if(abs(dX(i))>eps){
	    return 0;
	}
    }
    return 1;
}

typedef Eigen::Triplet<double> T;

Eigen::SparseMatrix<double> get_weight_matrix(std::vector<detection>& detections){
    const int mlength = (int) detections.size();    
    std::vector<T> tripletList;
    tripletList.reserve(2*mlength);
    Eigen::SparseMatrix<double> W(mlength*2,mlength*2);

    for(size_t i=0; i<mlength; i++){
	double x_unc = detections[i].ra_unc;
	double w2_x = 1.0/(x_unc*x_unc);
	tripletList.push_back(T(2*i, 2*i, w2_x));
	double y_unc = detections[i].dec_unc;
	double w2_y = 1.0/(y_unc*y_unc);	
	tripletList.push_back(T(2*i+1, 2*i+1, w2_y));	
    }
    W.setFromTriplets(tripletList.begin(), tripletList.end());

    return W;
}

void create_sequences(std::vector<double>& times,
		      double epoch,
		      std::vector<size_t>& reverse_in_seq,
		      std::vector<size_t>& reverse_out_seq,
		      std::vector<size_t>& forward_in_seq,
		      std::vector<size_t>& forward_out_seq){

    for(size_t i=0; i<times.size(); i++){
	if(times[i]>epoch){
	    forward_in_seq.push_back(i);
	    forward_out_seq.push_back(i);
	}else{	    
	    reverse_in_seq.push_back(i);
	    reverse_out_seq.push_back(i);
	}
    }

    std::reverse(reverse_in_seq.begin(), reverse_in_seq.end());
    std::reverse(reverse_out_seq.begin(), reverse_out_seq.end());    

}

int orbit_fit(struct assist_ephem* ephem,
	      struct reb_particle& p0, double epoch,
	      std::vector<double>& times,	      
	      std::vector<detection>& detections,
	      std::vector<residuals>& resid_vec,
	      std::vector<partials>& partials_vec,
	      size_t& iters,
	      double eps, size_t iter_max){

    std::vector<size_t> reverse_in_seq;
    std::vector<size_t> reverse_out_seq;

    std::vector<size_t> forward_in_seq;
    std::vector<size_t> forward_out_seq;    

    create_sequences(times, epoch,
		     reverse_in_seq, reverse_out_seq,
		     forward_in_seq, forward_out_seq);

    Eigen::SparseMatrix<double> W = get_weight_matrix(detections);
    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> chi2;
    Eigen::MatrixXd dX;

    int flag = 1;
    
    for(iters=0; iters<iter_max; iters++){

	compute_residuals(ephem, p0, epoch,
			  detections,
			  resid_vec,
			  partials_vec,
			  forward_in_seq,
			  forward_out_seq,
			  reverse_in_seq,
			  reverse_out_seq
			  );
	
	compute_dX(resid_vec, partials_vec, W, dX, C, chi2);
	//std::cout   << "chi2:\n" << chi2 << "\n";        
	//std::cout   << "Cinv:\n" << C.inverse() << "\n";
	
	//std::cout << "matrix dX\n" << dX << std::endl;

	if(converged(dX, eps)){
	    flag = 0;
	    break;
	}

	p0.x += dX(0);
	p0.y += dX(1);
	p0.z += dX(2);
	p0.vx += dX(3);
	p0.vy += dX(4);
	p0.vz += dX(5);

    }

    return flag;

}

// Initial orbit and covariance matrix
// Table of times, observations and uncertainties:
// x, y, z unit vectors; dRA, dDec (and possibly covariance)
// Table of observatory positions at the observation times

// Incorporate these types of observations:
// astrometry (unit vector)
// radar: range and doppler
// shift+stack

int main(int argc, char *argv[]) {

    // ephemeris files should be passed in or put in a config
    // file
    char ephemeris_filename[128] = "../../data/linux_p1550p2650.440";
    char small_bodies_filename[128] = "../../data/sb441-n16.bsp";
    struct assist_ephem* ephem = assist_ephem_create(
	    ephemeris_filename, 
	    small_bodies_filename); 
    if (!ephem){
        printf("Cannot create ephemeris structure.\n");
        exit(-1);
    }

    std::vector<detection> detections;
    std::vector<double> times;    
    
    // Read the observations
    char detections_filename[128]; 
    sscanf(argv[1], "%s", detections_filename);
    read_detections(detections_filename, detections, times);
    
    // Read the initial conditions
    char ic_filename[128]; 
    sscanf(argv[2], "%s", ic_filename);
    
    double epoch;
    struct reb_particle p0 = read_initial_conditions(ic_filename, &epoch);

    // Probably want to turn these into more
    // general vector types, to make calling from
    // python easier.
    // declare and preallocate the result vectors    
    std::vector<residuals> resid_vec(detections.size());
    std::vector<partials> partials_vec(detections.size());

    // Make these parameters flexible.
    size_t iter_max = 10;
    double eps = 1e-12;

    size_t iters;

    int flag = orbit_fit(ephem, p0, epoch,
			 times, 
			 detections,
			 resid_vec,
			 partials_vec,
			 iters,
			 eps, iter_max);

    if(flag != 0){
	printf("flag: %d %lu\n", flag, iters);
    }

    // Write the header
    printf("jd_tdb obsCode dRA dDec rdx rdy rdz rdvx rdvy rdvz ddx ddy ddz ddvx ddvy ddvz ra_unc dec_unc\n");
    print_residuals(detections, resid_vec, partials_vec);    

    // Important issues:
    // 1. Obtaining reliable initial orbit determination for the nonlinear fits.
    // 2. Making sure the weight matrix is as good as it can be
    // 3. Identifying and removing outliers

    // Later issues:
    // 1. Deflection of light
    
    assist_ephem_free(ephem);

}

