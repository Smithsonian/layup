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

// Try gauss's method with an unevenly spaced triplet.  DONE

// Put in a check on the residual as part of the convergence
// criteria.

// Use gauss to get a decent start for fitting a segment of the data.
// Then use the results of that as an initial guess for fitting a
// larger segment of data.

// Important issues:
// 1. Obtaining reliable initial orbit determination for the nonlinear fits.
// 2. Making sure the weight matrix is as good as it can be
// 3. Identifying and removing outliers

// Later issues:
// 1. Deflection of light

// Compile with something like this
// g++ -std=c++11 -I../..//src/ -I../../../rebound/src/ -I/Users/mholman/eigen-3.4.0 -Wl,-rpath,./ -Wpointer-arith -D_GNU_SOURCE -O3 -fPIC -I/usr/local/include -Wall -g  -Wno-unknown-pragmas -D_APPLE -DSERVER -DGITHASH=e99d8d73f0aa7fb7bf150c680a2d581f43d3a8be orbit_fit.cpp -L. -lassist -lrebound -L/usr/local/lib -o orbit_fit

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <pybind11/pybind11.h>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <variant>
#include <optional>


#include "orbit_fit.h"
#include "../gauss/gauss.cpp"
#include "orbit_fit_result.cpp"

extern "C"{
#include "rebound.h"
#include "assist.h"
}
using namespace Eigen;
using std::cout;
namespace py = pybind11;

double AU_M = 149597870700;
double SPEED_OF_LIGHT = 2.99792458e8 * 86400.0 / AU_M;

namespace orbit_fit {

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

void print_initial_condition(struct reb_particle p0, double epoch){

    printf("%lf %.16le %.16le %.16le %.16le %.16le %.16le\n",  epoch, p0.x, p0.y, p0.z, p0.vx, p0.vy, p0.vz);

    return;

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

void compute_single_residuals(struct assist_ephem* ephem,
			      struct assist_extras* ax,
			      int var,
			      Observation this_det,
			      residuals& resid,
			      partials& parts
			      ){

    // Takes a detection/observation and
    // a simulation as arguments. 
    // Returns the model observation, the
    // residuals, and the partial derivatives.

    struct reb_simulation* r = ax->sim;        
    
    double jd_tdb = this_det.epoch;

    double xe = this_det.observer_position[0];
    double ye = this_det.observer_position[1];
    double ze = this_det.observer_position[2];

	Eigen::Vector3d Av = std::get<AstrometryObservation>(this_det.observation_type).a_vec;
	Eigen::Vector3d Dv = std::get<AstrometryObservation>(this_det.observation_type).d_vec;

    double Ax = Av.x();
    double Ay = Av.y();
    double Az = Av.z();

    double Dx = Dv.x();
    double Dy = Dv.y();
    double Dz = Dv.z();

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

void predict(struct assist_ephem* ephem,
	     struct reb_particle p0, double epoch,	     
	     Observation this_det,
	     Eigen::MatrixXd& cov,
	     Eigen::MatrixXd& obs_cov
	     ){

    // Takes an ephemeris object, a simulation,
    // a detection/observation (only the time and observatory
    // state matter), and an orbit covariance
    // as arguments. 
    // Returns the model observation, the partials, and
    // the observation covariance.

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

    double jd_tdb = this_det.epoch;

    double xe = this_det.observer_position[0];
    double ye = this_det.observer_position[1];
    double ze = this_det.observer_position[2];

    Eigen::Vector3d Av = std::get<AstrometryObservation>(this_det.observation_type).a_vec;
    Eigen::Vector3d Dv = std::get<AstrometryObservation>(this_det.observation_type).d_vec;

    double Ax = Av.x();
    double Ay = Av.y();
    double Az = Av.z();

    double Dx = Dv.x();
    double Dy = Dv.y();
    double Dz = Dv.z();

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

    // Calculate the partial deriviatives of the model
    // observations with respect to the initial conditions

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

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
    B.resize(2, 6);

    for(size_t j=0; j<6; j++){
	B(0, j) = dx_resid[j];
	B(1, j) = dy_resid[j];
    }

    //Eigen::MatrixXd obs_cov(6, 6);
    obs_cov = B * cov * B.transpose();

    assist_free(ax);
    reb_simulation_free(r);

}

// This routine requires that resid_vec and partials_vec are
// preallocated because the indices in out_seq will be used for
// random access of locations in those vectors.
// Also, in_seq and out_seq must be of the same length.
void compute_residuals_sequence(struct assist_ephem* ephem,
				struct reb_particle p0, double epoch,
				std::vector<Observation>& detections,
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
	Observation this_det = detections[j];
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
			  std::vector<Observation>& detections,
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
		Eigen::MatrixXd& chi2,
		Eigen::Matrix<double, 6, 1>& grad, 
		double lambda){

    const int mlength = (int) partials_vec.size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
    Eigen::MatrixXd eye = MatrixXd::Identity(6, 6);
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
    C = Bt * W * B + lambda*eye;  // This is where the extra term for LM is.

    grad = Bt * W * resid_v;

    dX = C.colPivHouseholderQr().solve(-grad);
    // An alternative looks like this.  It's probably less stable.
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G = C.inverse();
    //dX = -G * Bt * W * resid_v;    

    chi2 = resid_v.transpose() * W * resid_v;
    // reset for inverse covariance
    C -= lambda*eye; 
    
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

int converged(Eigen::MatrixXd dX, double eps, double chi2, size_t ndof, double thresh){

    if((chi2>ndof)>thresh){
	return 2;
    }
    for(size_t i = 0; i<dX.size(); i++){
	if(abs(dX(i))>eps){
	    return 0;
	}
    }
    return 1;
}

typedef Eigen::Triplet<double> T;

Eigen::SparseMatrix<double> get_weight_matrix(std::vector<Observation>& detections){
    const int mlength = (int) detections.size();    
    std::vector<T> tripletList;
    tripletList.reserve(2*mlength);
    Eigen::SparseMatrix<double> W(mlength*2,mlength*2);

    for(size_t i=0; i<mlength; i++){
	double x_unc = *detections[i].ra_unc;
	double w2_x = 1.0/(x_unc*x_unc);
	tripletList.push_back(T(2*i, 2*i, w2_x));
	double y_unc = *detections[i].dec_unc;
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
	      struct reb_particle& p0,
	      double epoch, // Result from gauss
	      std::vector<Observation>& detections, // In Observation
	      std::vector<residuals>& resid_vec, // Compute as part of run
	      std::vector<partials>& partials_vec, // Compute as part of run
	      size_t& iters, // runtime
	      double& chi2_final, // result
	      Eigen::MatrixXd& cov,
	      double eps, //constant
	      size_t iter_max){ // runtime

    std::vector<size_t> reverse_in_seq;
    std::vector<size_t> reverse_out_seq;

    std::vector<size_t> forward_in_seq;
    std::vector<size_t> forward_out_seq;    

    std::vector<double> times(detections.size());

    for(int i = 0; i < detections.size(); i++) {
	times[i] = detections[i].epoch;
    }

    create_sequences(times, epoch,
		     reverse_in_seq, reverse_out_seq,
		     forward_in_seq, forward_out_seq);

    Eigen::SparseMatrix<double> W = get_weight_matrix(detections);
    
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> chi2;
    Eigen::MatrixXd dX;

    int flag = 1;

    double chi2_prev = HUGE_VAL;

    double rho_accept = 0.1;

    // Do an initial step
    double lambda = (206265.0*206265.0)/1000;
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

	Eigen::Matrix<double, 6, 1> grad;
	compute_dX(resid_vec, partials_vec, W, dX, C, chi2, grad, lambda);

	double chi2_d = chi2(0,0);

	double rho = (chi2_prev - chi2_d) / (dX.transpose() * (lambda * dX - grad)).norm();
	
	if(rho>rho_accept){

	    // Accept the step
	    // Reduce lambda
	    // Update the state
	    // Repeat, unless too many iterations

	    lambda *= 0.5; // reduce lambda and update the state
	    p0.x += dX(0);
	    p0.y += dX(1);
	    p0.z += dX(2);
	    p0.vx += dX(3);
	    p0.vy += dX(4);
	    p0.vz += dX(5);
	    chi2_prev = chi2_d;	    
	    
	}else{

	    // Reject the step
	    // leave the state
	    // increase lambda
	    // repeat, unless too many iterations

	    lambda *= 2.0;
	    
	}

	size_t ndof = detections.size()*2 - 6;
	double thresh = 10;
	int cflag = converged(dX, eps, chi2_d, ndof, thresh);
	if(cflag){
	    flag = 0;
	    chi2_final = chi2_d;
	    break;
	}

    }

    size_t ndof = detections.size()*2 - 6;
    double thresh = 10.0;
    if((chi2_final/ndof)>thresh){
	flag = 2;
    }

    cov = C.inverse();

    return flag;

}

// Go through the detections in reverse order, looking for
// a set of three detections such that each adjacent pair is
// separated by more than interval_min and less than interval_max.
std::vector<std::vector<size_t>> IOD_indices(std::vector<Observation>& detections,
					     double interval0_min,
					     double interval0_max,
					     double interval1_min,
					     double interval1_max,
					     size_t max_count,
					     size_t i_start){

    size_t cnt = 0;
    std::vector<std::vector<size_t>> res;
    for(int i=i_start; i>=2; i--){

	size_t idx_i = (size_t) i;
	Observation d2 = detections[idx_i];
	double t2 = d2.epoch;

	for(int j=i-1; j>=1; j--){

	    size_t idx_j = (size_t) j;	    
	    Observation d1 = detections[j];
	    double t1 = d1.epoch;

	    if(fabs(t2-t1)<interval0_min || fabs(t2-t1)>=interval0_max)
		continue;

	    for(int k=j-1; k>=0; k--){

		size_t idx_k = (size_t) k;
		Observation d0 = detections[idx_k];
		double t0 = d0.epoch;

		if(fabs(t1-t0)<interval1_min || fabs(t1-t0)>=interval1_max)
		    continue;

		cnt++;		

		if(cnt>max_count)
		    return res;

		res.push_back({idx_k, idx_j, idx_i});
		//printf("%lu %lu %lu %lf %lf %lf\n", idx_k, idx_j, idx_i, t0, t1, t2);

	    }

	}

    }

    return res;

}

// Initial orbit and covariance matrix
// Table of times, observations and uncertainties:
// x, y, z unit vectors; dRA, dDec (and possibly covariance)
// Table of observatory positions at the observation times

// Incorporate these types of observations:
// astrometry (unit vector)
// radar: range and doppler
// shift+stack

std::optional<struct assist_ephem*> get_ephem(std::string cache_dir) {

    std::string ephem_kernel = cache_dir + "linux_p1550p2650.440";
    std::string small_bodies_kernel = cache_dir + "sb441-n16.bsp";

    char* ephem_kernel_char = new char[ephem_kernel.length()];
    std::strcpy(ephem_kernel_char, ephem_kernel.c_str());
    char* small_bodies_kernel_char = new char[small_bodies_kernel.length()];
    std::strcpy(small_bodies_kernel_char, small_bodies_kernel.c_str());

    struct assist_ephem* ephem = assist_ephem_create(ephem_kernel_char, small_bodies_kernel_char); 
    if (!ephem){
        printf("Cannot create ephemeris structure.\n");	
        return std::nullopt;
    }else{
	return ephem;
    }
}
    
struct OrbfitResult run_from_files(std::string cache_dir, std::vector<Observation>& detections_full) {
    std::string ephem_kernel = cache_dir + "/linux_p1550p2650.440";
    std::string small_bodies_kernel = cache_dir + "/sb441-n16.bsp";

    // Allocate space for the string plus the null-terminator
    char* ephem_kernel_char = new char[ephem_kernel.length() + 1];
    std::strncpy(ephem_kernel_char, ephem_kernel.c_str(), ephem_kernel.length());
    ephem_kernel_char[ephem_kernel.length()] = '\0'; // Ensure null-termination

    char* small_bodies_kernel_char = new char[small_bodies_kernel.length() + 1];
    std::strncpy(small_bodies_kernel_char, small_bodies_kernel.c_str(), small_bodies_kernel.length());
    small_bodies_kernel_char[small_bodies_kernel.length()] = '\0'; // Ensure null-termination

    struct assist_ephem* ephem = assist_ephem_create(ephem_kernel_char, small_bodies_kernel_char); 
    if (!ephem){
        printf("Cannot create ephemeris structure.\n");
        exit(-1);
    }

    std::vector<double> times_full(detections_full.size());
    for(int i = 0; i < times_full.size (); i++) {
	times_full[i] = detections_full[i].epoch;
    }

    // Generate and print a random integer
    size_t start_i = detections_full.size() - 1;

    // This should be a little more clever and flexible
    // First find a triple of detections in the full data set.
    std::vector<std::vector<size_t>> idx = IOD_indices(detections_full, 2.0, 100.0, 2.0, 100.0, 10, start_i);    

    int success=0;
    size_t iters;
    double chi2_final;
    size_t dof;
    struct reb_particle p1;
        
    for(size_t i=0; i<idx.size(); i++){
	
	std::vector<size_t> indices = idx[i];
	size_t id0 = indices[0];
	size_t id1 = indices[1];
	size_t id2 = indices[2];

	printf("gauss indices: %lu %lu %lu\n", id0, id1, id2);
	fflush(stdout);

	Observation d0 = detections_full[id0];
	Observation d1 = detections_full[id1];
	Observation d2 = detections_full[id2];

	// Put this in a better place
	double GMtotal = 0.00029630927487993194;

	std::optional<std::vector<gauss_soln>> res = gauss(GMtotal, d0, d1, d2, 0.0001, SPEED_OF_LIGHT);
	if (!res.has_value()) {
	    printf("gauss failed.  Try another triplet.\n");
	    fflush(stdout);
	    continue;
	}

	// Now get a segment of data that spans the triplet and that uses up to
	// a certain number of points, if they are available in a time window.
	// This should be a parameter
	size_t num = 40;

	size_t curr_num = id2-id1+1;

	size_t start_index;
	size_t end_index;    
	if(curr_num >= num){
	    // We already have enough data and
	    // don't need to move the limits
	    start_index = id0;
	    end_index = id1;	 
	}else if((detections_full.size()-id0)>=num){
	    // There are enough points if we include
	    // only the later points, we only need to
	    // later limit.
	    start_index = id0;
	    end_index = id0+num;
	}else if(detections_full.size()>=num){
	    end_index = detections_full.size();
	    start_index = detections_full.size()-num;
	}else{
	    start_index = 0;
	    end_index = detections_full.size();
	}

	std::vector<Observation>::const_iterator first_d = detections_full.begin() + start_index;
	std::vector<Observation>::const_iterator last_d = detections_full.begin() + end_index;
	std::vector<Observation> detections(first_d, last_d);    

	std::vector<double>::const_iterator first_t = times_full.begin() + start_index;
	std::vector<double>::const_iterator last_t = times_full.begin() + end_index;
	std::vector<double> times(first_t, last_t);


	std::vector<residuals> resid_vec(detections.size());
	std::vector<partials> partials_vec(detections.size());

	// Make these parameters flexible.
	size_t iter_max = 100;
	double eps = 1e-12;

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov;
	int flag;

	p1.x = res.value()[0].x;
	p1.y = res.value()[0].y;
	p1.z = res.value()[0].z;    
	p1.vx = res.value()[0].vx;
	p1.vy = res.value()[0].vy;
	p1.vz = res.value()[0].vz;

	flag = orbit_fit(
		ephem,
		p1,
		res.value()[0].epoch,
		detections,
		resid_vec,
		partials_vec,
		iters,
		chi2_final,
		cov,
		eps,
		iter_max
	);
	
	dof = 2*detections.size()-6;

	if(flag == 0){
	    printf("flag: %d iters: %lu dof: %lu chi2: %lf %lu\n", flag, iters, dof, chi2_final, i);
	    print_initial_condition(p1, res.value()[0].epoch);
	}else{
	    printf("flag: %d iters: %lu, stage 2 failed.  Try another triplet %lu\n", flag, iters, i);
	    continue;
	}

	Eigen::MatrixXd obs_cov(6, 6);
	predict(
		ephem,
		p1,
		res.value()[0].epoch,
		detections_full[0],
		cov,
		obs_cov
	);

	size_t last = detections_full.size()-1;
	predict(
		ephem,
		p1, 
		res.value()[0].epoch,
		detections_full[last],
		cov,
		obs_cov
	);

	num = detections_full.size();

	first_d = detections_full.begin() + detections_full.size()-num;
	last_d = detections_full.end();
	std::vector<Observation> detections2(first_d, last_d);
	
	first_t = times_full.begin() + times_full.size()-num;
	last_t = times_full.end();
	std::vector<double> times2(first_t, last_t);

	std::vector<residuals> resid_vec2(detections2.size());
	std::vector<partials> partials_vec2(detections2.size());

	dof = 2*detections2.size()-6;


	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov2;	
	flag = orbit_fit(
		ephem,
		p1,
		res.value()[0].epoch,
		detections2,
		resid_vec2,
		partials_vec2,
		iters,
		chi2_final,
		cov2,
		eps,
		iter_max
	);

	if(flag == 0){
	    printf("flag: %d iters: %lu dof: %lu chi2: %lf\n", flag, iters, dof, chi2_final);
	    Eigen::MatrixXd obs_cov2(6, 6);
	    predict(
			ephem,
			p1,
			res.value()[0].epoch,
			detections_full[0],
		    cov2,
			obs_cov2
		);

	    last = detections_full.size()-1;
	    predict(
			ephem,
			p1,
			res.value()[0].epoch,
			detections_full[last],
			cov2,
			obs_cov2
		);
	    success = 1;
	    break;
	}else{
	    // At this point there is a good gauss solution and a good
	    // short interval solution, but the full data set is still failing.
	    // That probably means that the data set should be gradually enlarged.
	    printf("flag: %d iters: %lu, stage 3 failed.  Try another triplet %lu\n", flag, iters, i);	    
	    continue;
	}
    }
    if(success==0){
	printf("failed to find good fit\n");
	fflush(stdout);
    }

    struct OrbfitResult result;

    result.csq = chi2_final;
    result.ndof = dof;
    result.state[0] = p1.x;
    result.state[1] = p1.y;
    result.state[2] = p1.z;    
    result.state[3] = p1.vx;
    result.state[4] = p1.vy;
    result.state[5] = p1.vz;    
    
    result.niter = iters;
    
    assist_ephem_free(ephem);

    return result;

}

struct OrbfitResult run_from_vector(struct assist_ephem* ephem, std::vector<Observation>& detections_full) {

    std::vector<double> times_full(detections_full.size());
    for(int i = 0; i < times_full.size (); i++) {
	times_full[i] = detections_full[i].epoch;
    }

    size_t start_i = detections_full.size() - 1;

    // First find a triple of detections in the full data set.
    std::vector<std::vector<size_t>> idx = IOD_indices(detections_full, 2.0, 100.0, 2.0, 100.0, 10, start_i);    

    int success=0;
    size_t iters;
    double chi2_final;
    double result_epoch;
    size_t dof;
    struct reb_particle p1;
        
    for(size_t i=0; i<idx.size(); i++){
	
	std::vector<size_t> indices = idx[i];
	size_t id0 = indices[0];
	size_t id1 = indices[1];
	size_t id2 = indices[2];

	printf("gauss indices: %lu %lu %lu\n", id0, id1, id2);
	fflush(stdout);

	Observation d0 = detections_full[id0];
	Observation d1 = detections_full[id1];
	Observation d2 = detections_full[id2];

	// Put this in a better place
	double GMtotal = 0.00029630927487993194;

	std::optional<std::vector<gauss_soln>> res = gauss(GMtotal, d0, d1, d2, 0.0001, SPEED_OF_LIGHT);
	if (!res.has_value()) {
	    printf("gauss failed.  Try another triplet.\n");
	    fflush(stdout);
	    continue;
	}

	// Now get a segment of data that spans the triplet and that uses up to
	// a certain number of points, if they are available in a time window.
	// This should be a parameter
	size_t num = 40;

	size_t curr_num = id2-id1+1;

	size_t start_index;
	size_t end_index;    
	if(curr_num >= num){
	    // We already have enough data and
	    // don't need to move the limits
	    start_index = id0;
	    end_index = id1;	 
	}else if((detections_full.size()-id0)>=num){
	    // There are enough points if we include
	    // only the later points, we only need to
	    // later limit.
	    start_index = id0;
	    end_index = id0+num;
	}else if(detections_full.size()>=num){
	    end_index = detections_full.size();
	    start_index = detections_full.size()-num;
	}else{
	    start_index = 0;
	    end_index = detections_full.size();
	}

	std::vector<Observation>::const_iterator first_d = detections_full.begin() + start_index;
	std::vector<Observation>::const_iterator last_d = detections_full.begin() + end_index;
	std::vector<Observation> detections(first_d, last_d);    

	std::vector<double>::const_iterator first_t = times_full.begin() + start_index;
	std::vector<double>::const_iterator last_t = times_full.begin() + end_index;
	std::vector<double> times(first_t, last_t);


	std::vector<residuals> resid_vec(detections.size());
	std::vector<partials> partials_vec(detections.size());

	// Make these parameters flexible.
	size_t iter_max = 100;
	double eps = 1e-12;

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov;
	int flag;
	//size_t iters;

	//struct reb_particle p1;

	p1.x = res.value()[0].x;
	p1.y = res.value()[0].y;
	p1.z = res.value()[0].z;    
	p1.vx = res.value()[0].vx;
	p1.vy = res.value()[0].vy;
	p1.vz = res.value()[0].vz;

	flag = orbit_fit(
		ephem,
		p1,
		res.value()[0].epoch,
		detections,
		resid_vec,
		partials_vec,
		iters,
		chi2_final,
		cov,
		eps,
		iter_max
	);
	
	dof = 2*detections.size()-6;

	if(flag == 0){
	    printf("flag: %d iters: %lu dof: %lu chi2: %lf %lu\n", flag, iters, dof, chi2_final, i);
	    print_initial_condition(p1, res.value()[0].epoch);
	}else{
	    printf("flag: %d iters: %lu, stage 2 failed.  Try another triplet %lu\n", flag, iters, i);
	    continue;
	}

	Eigen::MatrixXd obs_cov(6, 6);
	predict(
		ephem,
		p1,
		res.value()[0].epoch,
		detections_full[0],
		cov,
		obs_cov
	);

	size_t last = detections_full.size()-1;
	predict(
		ephem,
		p1, 
		res.value()[0].epoch,
		detections_full[last],
		cov,
		obs_cov
	);

	num = detections_full.size();

	first_d = detections_full.begin() + detections_full.size()-num;
	last_d = detections_full.end();
	std::vector<Observation> detections2(first_d, last_d);
	
	first_t = times_full.begin() + times_full.size()-num;
	last_t = times_full.end();
	std::vector<double> times2(first_t, last_t);

	std::vector<residuals> resid_vec2(detections2.size());
	std::vector<partials> partials_vec2(detections2.size());

	dof = 2*detections2.size()-6;


	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov2;	
	flag = orbit_fit(
		ephem,
		p1,
		res.value()[0].epoch,
		detections2,
		resid_vec2,
		partials_vec2,
		iters,
		chi2_final,
		cov2,
		eps,
		iter_max
	);

	if(flag == 0){
	    printf("flag: %d iters: %lu dof: %lu chi2: %lf\n", flag, iters, dof, chi2_final);
	    Eigen::MatrixXd obs_cov2(6, 6);
	    result_epoch = res.value()[0].epoch;
	    predict(
			ephem,
			p1,
			res.value()[0].epoch,
			detections_full[0],
		    cov2,
			obs_cov2
		);

	    last = detections_full.size()-1;
	    predict(
			ephem,
			p1,
			res.value()[0].epoch,
			detections_full[last],
			cov2,
			obs_cov2
		);
	    success = 1;
	    break;
	}else{
	    // At this point there is a good gauss solution and a good
	    // short interval solution, but the full data set is still failing.
	    // That probably means that the data set should be gradually enlarged.
	    printf("flag: %d iters: %lu, stage 3 failed.  Try another triplet %lu\n", flag, iters, i);	    
	    continue;
	}
    }
    if(success==0){
	printf("fully failed\n");
	fflush(stdout);
    }

    struct OrbfitResult result;

    result.epoch = result_epoch;
    result.csq = chi2_final;
    result.ndof = dof;
    result.niter = iters;    
    result.state[0] = p1.x;
    result.state[1] = p1.y;
    result.state[2] = p1.z;    
    result.state[3] = p1.vx;
    result.state[4] = p1.vy;
    result.state[5] = p1.vz;    
    
    // Important issues:
    // 1. Obtaining reliable initial orbit determination for the nonlinear fits.
    // 2. Making sure the weight matrix is as good as it can be
    // 3. Identifying and removing outliers

    // Later issues:
    // 1. Deflection of light
    
    return result;

}

std::optional<struct gauss_soln> run_from_vector_with_ic(struct assist_ephem* ephem, gauss_soln soln, std::vector<Observation>& detections) {    

    int success=0;
    size_t iters;
    double chi2_final;
    size_t dof;

    std::vector<residuals> resid_vec(detections.size());
    std::vector<partials> partials_vec(detections.size());

    // Make these parameters flexible.
    size_t iter_max = 100;
    double eps = 1e-12;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov;
    int flag;

    reb_particle p1;

    p1.x = soln.x;
    p1.y = soln.y;
    p1.z = soln.z;    
    p1.vx = soln.vx;
    p1.vy = soln.vy;
    p1.vz = soln.vz;
    double epoch = soln.epoch;

    flag = orbit_fit(
		     ephem,
		     p1,
		     epoch, 
		     detections,
		     resid_vec,
		     partials_vec,
		     iters,
		     chi2_final,
		     cov,
		     eps,
		     iter_max
		     );
	
    dof = 2*detections.size()-6;

    if(flag == 0){

	Eigen::MatrixXd obs_cov(2, 2);
	predict(
		ephem,
		p1,
		epoch,
		detections[0],
		cov,
		obs_cov
		);
	std::cout << "obs_cov: \n" << obs_cov << std::endl;

	size_t last = detections.size()-1;
	predict(
		ephem,
		p1,
		epoch,
		detections[last],
		cov,
		obs_cov
		);
	std::cout << "obs_cov: \n" << obs_cov << std::endl;

	struct gauss_soln result;	

	result.iters = iters;
	result.csq = chi2_final;
	result.dof = dof;
	result.flag = flag;
	result.method = "orbit_fit";

	result.epoch = epoch;
	result.x = p1.x;
	result.y = p1.y;
	result.z = p1.z;    
	result.vx = p1.vx;
	result.vy = p1.vy;
	result.vz = p1.vz;

	result.cov = cov;	

	return result;
	
    }else{
        return std::nullopt;
    }
}
    
#ifdef Py_PYTHON_H
static void orbit_fit_bindings(py::module& m) {
    py::class_<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(m, "MatrixXd")
        .def(py::init<>());
    py::class_<assist_ephem>(m, "assist_ephem");
    m.def("orbit_fit", &orbit_fit::orbit_fit, R"pbdoc(Main function)pbdoc");
    m.def("get_ephem", &orbit_fit::get_ephem, R"pbdoc(get ephemeris)pbdoc");    
    m.def("run_from_files", &orbit_fit::run_from_files, R"pbdoc(Runner function)pbdoc");
    m.def("run_from_vector", &orbit_fit::run_from_vector, R"pbdoc(Runner function v2)pbdoc");
    m.def("run_from_vector_with_ic", &orbit_fit::run_from_vector_with_ic, R"pbdoc(Runner function ic version)pbdoc");
    m.def("predict", &orbit_fit::predict, R"pbdoc(predict)pbdoc");            
}
#endif /* Py_PYTHON_H */

} // namespace: orbit_fit

