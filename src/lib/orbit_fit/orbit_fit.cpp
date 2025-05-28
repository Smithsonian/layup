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
#include "predict.cpp"
#include "mattodiff.hpp"
#include "stumpff.hpp"

#define PI 3.14159265358979323846
#define TWOPI 6.283185307179586476925287

using std::cout;
namespace py = pybind11;


namespace orbit_fit
{

    typedef struct
    {
	Dual f;
	Dual fp;
	Dual fpp;
	Dual fppp;
    } DRootResult;

    typedef struct
    {
	double f;
	double fp;
	double fpp;
	double fppp;
    } RootResult;

    RootResult root_function(double s, double mu, double alpha, double r0, double r0dot, double t){
	/*
	  Root function used in the Halley minimizer
	  Computes the zeroth, first, second, and third derivatives
	  of the universal Kepler equation f

	  Parameters
	  ----------
	  s : float
	  Eccentric anomaly
	  mu : float
	  Standard gravitational parameter GM
	  alpha : float
	  Total energy
	  r0 : float
	  Initial position
	  r0dot : float
	  Initial velocity
	  t : float
	  Time

	  Returns
	  -------
	  f : float
	  universal Kepler equation)
	  fp : float
	  (first derivative of f
	  fpp : float
	  second derivative of f
	  fppp : float
	  third derivative of f

	*/

	Stumpff cs = stumpff(alpha * s * s);
	double c0 = cs.c0;
	double c1 = cs.c1;
	double c2 = cs.c2;
	double c3 = cs.c3;
    
	double zeta = mu - alpha * r0;
	double f = r0 * s * c1 + r0 * r0dot * s * s * c2 + mu * s * s * s * c3 - t;
	double fp = r0 * c0 + r0 * r0dot * s * c1 + mu * s * s * c2; // This is equivalent to r
	double fpp = zeta * s * c1 + r0 * r0dot * c0;
	double fppp = zeta * c0 - r0 * r0dot * alpha * s * c1;

	RootResult rr;
	rr.f = f;
	rr.fp = fp;
	rr.fpp = fpp;
	rr.fppp = fppp;
	return rr;
    }

    DRootResult droot_function(Dual s, double mu, Dual alpha, Dual r0, Dual r0dot, Dual t){
	/*
	  Root function used in the Halley minimizer
	  Computes the zeroth, first, second, and third derivatives
	  of the universal Kepler equation f

	  Parameters
	  ----------
	  s : float
	  Eccentric anomaly
	  mu : float
	  Standard gravitational parameter GM
	  alpha : float
	  Total energy
	  r0 : float
	  Initial position
	  r0dot : float
	  Initial velocity
	  t : float
	  Time

	  Returns
	  -------
	  f : float
	  universal Kepler equation)
	  fp : float
	  (first derivative of f
	  fpp : float
	  second derivative of f
	  fppp : float
	  third derivative of f
	*/

	DStumpff cs = dstumpff(alpha * s * s);
	Dual c0 = cs.c0;
	Dual c1 = cs.c1;
	Dual c2 = cs.c2;
	Dual c3 = cs.c3;

	Dual zeta = mu - alpha * r0;
	Dual f = r0 * s * c1 + r0 * r0dot * s * s * c2 + mu * s * s * s * c3 - t;
	Dual fp = r0 * c0 + r0 * r0dot * s * c1 + mu * s * s * c2; // This is equivalent to r
	Dual fpp = zeta * s * c1 + r0 * r0dot * c0;
	Dual fppp = zeta * c0 - r0 * r0dot * alpha * s * c1;

	DRootResult rr;
	rr.f = f;
	rr.fp = fp;
	rr.fpp = fpp;
	rr.fppp = fppp;
	return rr;
    }

    typedef struct
    {
	int flag;
	double x;
	double fp;
    } HalleyResult;

    HalleyResult halley_safe(double x1, double x2, double mu, double alpha, double r0, double r0dot, double t, double xacc=2e-15, size_t maxit=100){
	/*
	  Applies the Halley root finding algorithm on the universal Kepler equation

	  Parameters
	  ----------
	  x1 : float
	  Previous guess used in minimization
	  x2 : float
	  Current guess for minimization
	  mu : float
	  Standard gravitational parameter GM
	  alpha : float
	  Total energy
	  r0 : float
	  Initial position
	  r0dot : float
	  Initial velocity
	  t : float
	  Time
	  xacc : float
	  Accuracy in x before algorithm declares convergence
	  maxit : int
	  Maximum number of iterations

	  Returns
	  ----------
	  : boolean
	  True if minimization converged, False otherwise
	  : float
	  Solution
	  : float
	  First derivative of solution

	*/
	// verify the bracket
	// Use these values later
	RootResult rr_l = root_function(x1, mu, alpha, r0, r0dot, t);
	double fl = rr_l.f;
	double fpl = rr_l.fp;
	double fppl = rr_l.fpp;
    
	RootResult rr_h = root_function(x2, mu, alpha, r0, r0dot, t);
	double fh = rr_h.f;
	double fph = rr_h.fp;
	double fpph = rr_h.fpp;

	double f, fp, fpp, fppp;
    
	if((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)){
	    return {1, x1, fl};
	}
	if(fl == 0){
	    return {0, x1, fpl};
	}
	if(fh == 0){
	    return{0, x2, fph};
	}

	double xl, xh, rts;
	    
	// Orient the search so that f(xl) < 0 and f(xh)>0
	if(fl < 0.0){
	    xl = x1;
	    xh = x2;
	}else{
	    xh = x1;
	    xl = x2;
	}

	// Use the initial values
	if(fabs(fl) < fabs(fh)){
	    rts = xl;
	    f = fl;
	    fp = fpl;
	    fpp = fppl;
	}else{
	    rts = xh;
	    f = fh;
	    fp = fph;
	    fpp = fpph;
	}

	rts = 0.5 * (x1 + x2);        // Initialize the guess for root
	double dxold = fabs(x2 - x1); // the “stepsize before last"
	double dx = dxold;            // and the last step.

	RootResult rr = root_function(rts, mu, alpha, r0, r0dot, t);
	f = rr.f;
	fp = rr.fp;
	fpp = rr.fpp;
	fppp = rr.fppp;

	for(size_t j = 0; j < maxit; j++){ // Loop over allowed iterations.
	    // Check the criteria.
	    if((((rts - xh) * fp - f) * ((rts - xl) * fp - f) > 0.0) ||
	       (fabs(2.0 * f) > fabs(dxold * fp))){
		// Bisect the interval
		dxold = dx;
		dx = 0.5 * (xh - xl);
		rts = xl + dx;
		if(fabs(dx / rts) < xacc){
		    return {0, rts, fp};
		}
	    }else{
		// Take a Hally step
		dxold = dx;
		dx = -f / fp;
		dx = -f /(fp + dx * fpp / 2.0 );
		dx = -f /(fp + dx * fpp / 2.0 + dx * dx * fppp / 6.0);	    
		rts += dx;
		if(fabs(dx / rts) < xacc){
		    return {0, rts, fp};
		}
	    }
	    rr = root_function(rts, mu, alpha, r0, r0dot, t);
	    //printf("here %lf %lf %lf %lf %lf\n", rts, f, fp, fpp, fppp);    
	
	    f = rr.f;
	    fp = rr.fp;
	    fpp = rr.fpp;
	    fppp = rr.fppp;	
	    
	    // Maintain the bracket on the root.
	    if(f < 0.0){
		xl = rts;
		fl = f;
	    }else{
		xh = rts;
		fh = f;
	    }
	}

	return {1, rts, fp};
    
    }

    typedef struct
    {
	int flag;
	Dual x;
	Dual fp;
    } DHalleyResult;

    DHalleyResult halley_safe(Dual x1, Dual x2, double mu, Dual alpha, Dual r0, Dual r0dot, Dual t,  Dual xg=-1000.0, double xacc=2e-15, size_t maxit=100){
	/*
	  Applies the Halley root finding algorithm on the universal Kepler equation

	  Parameters
	  ----------
	  x1 : float
	  Previous guess used in minimization
	  x2 : float
	  Current guess for minimization
	  mu : float
	  Standard gravitational parameter GM
	  alpha : float
	  Total energy
	  r0 : float
	  Initial position
	  r0dot : float
	  Initial velocity
	  t : float
	  Time
	  xacc : float
	  Accuracy in x before algorithm declares convergence
	  maxit : int
	  Maximum number of iterations

	  Returns
	  ----------
	  : boolean
	  True if minimization converged, False otherwise
	  : float
	  Solution
	  : float
	  First derivative of solution

	*/
	// verify the bracket
	// Use these values later
	DRootResult rr_l = droot_function(x1, mu, alpha, r0, r0dot, t);
	Dual fl = rr_l.f;
	Dual fpl = rr_l.fp;
	Dual fppl = rr_l.fpp;
    
	DRootResult rr_h = droot_function(x2, mu, alpha, r0, r0dot, t);
	Dual fh = rr_h.f;
	Dual fph = rr_h.fp;
	Dual fpph = rr_h.fpp;

	Dual f, fp, fpp, fppp;
    
	if((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)){
	    return {1, x1, fl};
	}
	if(fl == 0){
	    return {0, x1, fpl};
	}
	if(fh == 0){
	    return{0, x2, fph};
	}

	Dual xl, xh, rts;

	// Orient the search so that f(xl) < 0 and f(xh)>0
	if(fl < 0.0){
	    xl = x1;
	    xh = x2;
	}else{
	    xh = x1;
	    xl = x2;
	}

	// Use the initial values
	if(fabs(fl.real) < fabs(fh.real)){
	    rts = xl;
	    f = fl;
	    fp = fpl;
	    fpp = fppl;
	}else{
	    rts = xh;
	    f = fh;
	    fp = fph;
	    fpp = fpph;
	}

	rts = 0.5 * (x1 + x2);        // Initialize the guess for root

	if(xg.real != -1000.0){
	    rts = xg;
	}
	Dual dxold = (x2 - x1);     // the “stepsize before last"
	Dual dx = dxold;            // and the last step.

	DRootResult rr = droot_function(rts, mu, alpha, r0, r0dot, t);
	f = rr.f;
	fp = rr.fp;
	fpp = rr.fpp;
	fppp = rr.fppp;

	for(size_t j = 0; j < maxit; j++){ // Loop over allowed iterations.
	    //printf("j: %lu\n", j);	
	    // Check the criteria.
	    if((((rts - xh) * fp - f) * ((rts - xl) * fp - f) > 0.0) ||
	       (fabs((2.0 * f).real) > fabs((dxold * fp).real))){
		// Bisect the interval
		dxold = dx;
		dx = 0.5 * (xh - xl).real;
		rts = xl + dx;
		if(fabs((dx / rts).real) < xacc){
		    return {0, rts, fp};
		}
	    }else{
		// Take a Hally step
		dxold = dx;
		dx = -f / fp;
		dx = -f /(fp + dx * fpp / 2.0 );
		dx = -f /(fp + dx * fpp / 2.0 + dx * dx * fppp / 6.0);
		rts = rts + dx;
		if(fabs((dx / rts).real) < xacc){
		    return {0, rts, fp};
		}
	    }
	    rr = droot_function(rts, mu, alpha, r0, r0dot, t);
	
	    f = rr.f;
	    fp = rr.fp;
	    fpp = rr.fpp;
	    fppp = rr.fppp;	
	    
	    // Maintain the bracket on the root.
	    if(f.real < 0.0){
		xl = rts;
		fl = f;
	    }else{
		xh = rts;
		fh = f;
	    }
	}

	return {1, rts, fp};
    
    }

    typedef struct
    {
	double x, y, z;
	double xd, yd, zd;
    } CartesianState;

    typedef struct
    {
	int flag;
	CartesianState state;
    } CartesianResult;


    CartesianResult universal_cartesian(double mu, double q, double e,
					double incl, double longnode, double argperi, double tp,
					double epochMJD_TDB,
					size_t maxit=100				    
					){
	/*
	  Converts from a series of orbital elements into state vectors
	  using the universal variable formulation

	  The output vector will be oriented in the same system as
	  the positional angles (i, Omega, omega)

	  Note that mu, q, tp and epochMJD_TDB must have compatible units
	  As an example, if q is in au and tp/epoch are in days, mu must
	  be in (au^3)/days^2

	  Parameters
	  ----------
	  mu : float
	  Standard gravitational parameter GM (see note above about units)
	  q : float
	  Perihelion (see note above about units)
	  e : float
	  Eccentricity
	  incl : float
	  Inclination (radians)
	  longnode : float
	  Longitude of ascending node (radians)
	  argperi : float
	  Argument of perihelion (radians)
	  tp : float
	  Time of perihelion passage in TDB scale (see note above about units)
	  epochMJD_TDB : float
	  Epoch (in TDB) when the elements are defined (see note above about units)

	  Returns
	  ----------
	  : float
	  x coordinate
	  : float
	  y coordinate
	  : float
	  z coordinate
	  : float
	  x velocity
	  : float
	  y velocity
	  : float
	  z velocity
	*/

	double t = epochMJD_TDB - tp;
	/*
	// MJH: I removed this part because I haven't
	// implemented the fmod function for dual numbers
	// because it's discontinuous.
	//
	double a, per;      
	if(e < 1){
	// Remove extra full orbits
	a = q / (1.0 - e);
        per = TWOPI / sqrt(mu / (a * a * a));
	t = std::fmod(t, per);
	}
	*/

	// Establish constants for Kepler's equation,
	// starting at pericenter.
	double r0 = q;
	double r0dot = 0.0;
	double v2 = mu * (1.0 + e) / q;
	double alpha = 2.0 * mu / r0 - v2;

	// bracket the root
	double ds = (t - 0.0) / 100.0;
	double s_prev = 0.0;
	RootResult rr = root_function(s_prev, mu, alpha, r0, r0dot, t);

	double f_prev = rr.f;

	double s = s_prev + ds;

	rr = root_function(s, mu, alpha, r0, r0dot, t);
	double f = rr.f;

	while(f * f_prev > 0.0){
	    s_prev = s;
	    f_prev = f;
	    s = s_prev + ds;
	    rr = root_function(s, mu, alpha, r0, r0dot, t);
	    f = rr.f;
	}


	CartesianResult cr;
	int flag;

	HalleyResult hr = halley_safe(s_prev, s, mu, alpha, r0, r0dot, t);
	flag = hr.flag;
	double ss = hr.x;
	double fp = hr.fp;

	if(flag != 0){
	    printf("flag: %d\n", flag);
	}
	size_t count = 0;
	while(flag != 0){
	    rr = root_function(s, mu, alpha, r0, r0dot, t);
	    f = rr.f;
	    fp = rr.fp;
	    s_prev = s;
	    s = s - f / fp;
	    hr = halley_safe(s_prev, s, mu, alpha, r0, r0dot, t);
	    flag = hr.flag;
	    ss = hr.x;
	    fp = hr.fp;
	    count += 1;
	    if(count > maxit){
		cr.flag = 1;
		return cr;
	    }
	}

	Stumpff cs = stumpff(alpha * ss * ss);
	double c0 = cs.c0;
	double c1 = cs.c1;
	double c2 = cs.c2;
	double c3 = cs.c3;

	double r = r0 * c0 + r0 * r0dot * ss * c1 + mu * ss * ss * c2; // This is equivalent to fp.

	double g1 = c1 * ss;
	double g2 = c2 * ss * ss;
	double g3 = c3 * ss * ss * ss;

	f = 1.0 - (mu / r0) * g2;
	double g = t - mu * g3;
	double fdot = -(mu / (r * r0)) * g1;
	double gdot = 1.0 - (mu / r) * g2;

	// define position and velocity at pericenter
	double x0 = q;
	double y0 = 0.0;
	double z0 = 0.0;
	double xd0 = 0.0;
	double yd0 = sqrt(v2);
	double zd0 = 0;

	// compute position and velocity at time t (from pericenter)
	double xt = f * x0 + g * xd0;
	double yt = f * y0 + g * yd0;
	double zt = f * z0 + g * zd0;    
	double xdt = fdot * x0 + gdot * xd0;
	double ydt = fdot * y0 + gdot * yd0;
	double zdt = fdot * z0 + gdot * zd0;    

	// rotate by argument of perihelion in orbit plane
	double cosw = cos(argperi);
	double sinw = sin(argperi);
	double xp = xt * cosw - yt * sinw;
	double yp = xt * sinw + yt * cosw;
	double zp = zt;
	double xdp = xdt * cosw - ydt * sinw;
	double ydp = xdt * sinw + ydt * cosw;
	double zdp = zdt;

	// rotate by inclination about x axis 
	double cosi = cos(incl);
	double sini = sin(incl);
	double x = xp;
	double y = yp * cosi - zp * sini;
	double z = yp * sini + zp * cosi;
	double xd = xdp;
	double yd = ydp * cosi - zdp * sini;
	double zd = ydp * sini + zdp * cosi;

	// rotate by longitude of node about z axis 
	double cosnode = cos(longnode);
	double sinnode = sin(longnode);
	xp = x * cosnode - y * sinnode;
	yp = x * sinnode + y * cosnode;
	zp = z;
	xdp = xd * cosnode - yd * sinnode;
	ydp = xd * sinnode + yd * cosnode;
	zdp = zd;

	cr.flag = 0;
	cr.state = {xp, yp, zp, xdp, ydp, zdp};

	return cr;  
    }

    typedef struct
    {
	Dual x, y, z;
	Dual xd, yd, zd;
    } DCartesianState;

    typedef struct
    {
	int flag;
	DCartesianState state;
	Dual s;
    } DCartesianResult;

    DCartesianResult universal_cartesian(double mu, Dual q, Dual e,
					 Dual incl, Dual longnode, Dual argperi, Dual tp,
					 double epochMJD_TDB,
					 Dual sg=-1000.0,
					 size_t maxit=100				    
					 ){
	/*
	  Converts from a series of orbital elements into state vectors
	  using the universal variable formulation

	  The output vector will be oriented in the same system as
	  the positional angles (i, Omega, omega)

	  Note that mu, q, tp and epochMJD_TDB must have compatible units
	  As an example, if q is in au and tp/epoch are in days, mu must
	  be in (au^3)/days^2

	  Parameters
	  ----------
	  mu : float
	  Standard gravitational parameter GM (see note above about units)
	  q : float
	  Perihelion (see note above about units)
	  e : float
	  Eccentricity
	  incl : float
	  Inclination (radians)
	  longnode : float
	  Longitude of ascending node (radians)
	  argperi : float
	  Argument of perihelion (radians)
	  tp : float
	  Time of perihelion passage in TDB scale (see note above about units)
	  epochMJD_TDB : float
	  Epoch (in TDB) when the elements are defined (see note above about units)

	  Returns
	  ----------
	  : float
	  x coordinate
	  : float
	  y coordinate
	  : float
	  z coordinate
	  : float
	  x velocity
	  : float
	  y velocity
	  : float
	  z velocity
	*/

	Dual t = epochMJD_TDB - tp;

	/*
	// MJH: I removed this part because I haven't
	// implemented the fmod function for dual numbers
	// because it's discontinuous.
	//
	Dual a, per;
	if(e < 1){
	// Remove extra full orbits
	a = q / (1.0 - e);
        per = TWOPI / sqrt(mu / (a * a * a));
	t = std::fmod(t, per);
	}
	*/

	// Establish constants for Kepler's equation,
	// starting at pericenter.
	Dual r0 = q;
	Dual r0dot = 0.0;
	Dual v2 = mu * (1.0 + e) / q;
	Dual alpha = 2.0 * mu / r0 - v2;

	// bracket the root
	Dual ds = (t - 0.0) / 10.0;
	Dual s_prev = 0.0;
	DRootResult rr = droot_function(s_prev, mu, alpha, r0, r0dot, t);

	Dual f_prev = rr.f;

	Dual s = s_prev + ds;

	rr = droot_function(s, mu, alpha, r0, r0dot, t);
	Dual f = rr.f;

	while(f * f_prev > 0.0){
	    s_prev = s;
	    f_prev = f;
	    s = s_prev + ds;
	    //printf("while s: %lf %lf\n", s.real, s.dual);	
	    rr = droot_function(s, mu, alpha, r0, r0dot, t);
	    f = rr.f;
	}

	DCartesianResult cr;
	int flag;

	DHalleyResult hr;
    
	if(sg.real != -1000.0){
	    sg.dual = 0.0;
	    hr = halley_safe(s_prev, s, mu, alpha, r0, r0dot, t, sg);
	}else{
	    hr = halley_safe(s_prev, s, mu, alpha, r0, r0dot, t);
	}
	flag = hr.flag;
	Dual ss = hr.x;
	Dual fp = hr.fp;

	if(flag != 0){
	    printf("flag: %d\n", flag);
	}
	size_t count = 0;
	while(flag != 0){
	    rr = droot_function(s, mu, alpha, r0, r0dot, t);
	    f = rr.f;
	    fp = rr.fp;
	    s_prev = s;
	    s = s - f / fp;
	    hr = halley_safe(s_prev, s, mu, alpha, r0, r0dot, t);
	    flag = hr.flag;
	    ss = hr.x;
	    fp = hr.fp;
	    count += 1;
	    if(count > maxit){
		cr.flag = 1;
		return cr;
	    }
	}

	DStumpff cs = dstumpff(alpha * ss * ss);
	Dual c0 = cs.c0;
	Dual c1 = cs.c1;
	Dual c2 = cs.c2;
	Dual c3 = cs.c3;

	Dual r = r0 * c0 + r0 * r0dot * ss * c1 + mu * ss * ss * c2; // This is equivalent to fp.

	//Dual g0 = c0;
	Dual g1 = c1 * ss;
	Dual g2 = c2 * ss * ss;
	Dual g3 = c3 * ss * ss * ss;

	f = 1.0 - (mu / r0) * g2;
	Dual g = t - mu * g3;
	Dual fdot = -(mu / (r * r0)) * g1;
	Dual gdot = 1.0 - (mu / r) * g2;

	// define position and velocity at pericenter
	Dual x0 = q;
	Dual y0 = 0.0;
	Dual z0 = 0.0;
	Dual xd0 = 0.0;
	Dual yd0 = sqrt(v2);
	Dual zd0 = 0;

	// compute position and velocity at time t (from pericenter)
	Dual xt = f * x0 + g * xd0;
	Dual yt = f * y0 + g * yd0;
	Dual zt = f * z0 + g * zd0;    
	Dual xdt = fdot * x0 + gdot * xd0;
	Dual ydt = fdot * y0 + gdot * yd0;
	Dual zdt = fdot * z0 + gdot * zd0;    

	// rotate by argument of perihelion in orbit plane
	Dual cosw = cos(argperi);
	Dual sinw = sin(argperi);
	Dual xp = xt * cosw - yt * sinw;
	Dual yp = xt * sinw + yt * cosw;
	Dual zp = zt;
	Dual xdp = xdt * cosw - ydt * sinw;
	Dual ydp = xdt * sinw + ydt * cosw;
	Dual zdp = zdt;

	// rotate by inclination about x axis 
	Dual cosi = cos(incl);
	Dual sini = sin(incl);
	Dual x = xp;
	Dual y = yp * cosi - zp * sini;
	Dual z = yp * sini + zp * cosi;
	Dual xd = xdp;
	Dual yd = ydp * cosi - zdp * sini;
	Dual zd = ydp * sini + zdp * cosi;

	// rotate by longitude of node about z axis 
	Dual cosnode = cos(longnode);
	Dual sinnode = sin(longnode);
	xp = x * cosnode - y * sinnode;
	yp = x * sinnode + y * cosnode;
	zp = z;
	xdp = xd * cosnode - yd * sinnode;
	ydp = xd * sinnode + yd * cosnode;
	zdp = zd;

	cr.flag = 0;
	cr.state = {xp, yp, zp, xdp, ydp, zdp};

	cr.s = ss;

	return cr;  
    }

    typedef Eigen::Matrix<double, 6, 6> Matrix6d;    
    
    Matrix6d universal_cartesian_jacobian(double mu,
					  double q, double e, double incl,
					  double longnode, double argperi, double tp,
					  double epoch){
	Dual qd = {q, 0.0};
	Dual ed = {e, 0.0};
	Dual longnoded = {longnode, 0.0};
	Dual argperid = {argperi, 0.0};
	Dual incld = {incl, 0.0};
	Dual tpd = {tp, 0.0};    

	qd.dual = 1.0;
	DCartesianResult dcr = universal_cartesian(mu, qd, ed, incld, longnoded, argperid, tpd, epoch);
	Dual s = dcr.s;
	qd.dual = 0.0;

	Matrix6d jac;
	jac(0, 0) = dcr.state.x.dual;
	jac(0, 1) = dcr.state.y.dual;
	jac(0, 2) = dcr.state.z.dual;
	jac(0, 3) = dcr.state.xd.dual;
	jac(0, 4) = dcr.state.yd.dual;
	jac(0, 5) = dcr.state.zd.dual;	

	ed.dual = 1.0;
	dcr = universal_cartesian(mu, qd, ed, incld, longnoded, argperid, tpd, epoch, s);
	ed.dual = 0.0;

	jac(1, 0) = dcr.state.x.dual;
	jac(1, 1) = dcr.state.y.dual;
	jac(1, 2) = dcr.state.z.dual;
	jac(1, 3) = dcr.state.xd.dual;
	jac(1, 4) = dcr.state.yd.dual;
	jac(1, 5) = dcr.state.zd.dual;		

	incld.dual = 1.0;
	dcr = universal_cartesian(mu, qd, ed, incld, longnoded, argperid, tpd, epoch, s);
	incld.dual = 0.0;

	jac(2, 0) = dcr.state.x.dual;
	jac(2, 1) = dcr.state.y.dual;
	jac(2, 2) = dcr.state.z.dual;
	jac(2, 3) = dcr.state.xd.dual;
	jac(2, 4) = dcr.state.yd.dual;
	jac(2, 5) = dcr.state.zd.dual;	

	longnoded.dual = 1.0;
	dcr = universal_cartesian(mu, qd, ed, incld, longnoded, argperid, tpd, epoch, s);
	longnoded.dual = 0.0;

	jac(3, 0) = dcr.state.x.dual;
	jac(3, 1) = dcr.state.y.dual;
	jac(3, 2) = dcr.state.z.dual;
	jac(3, 3) = dcr.state.xd.dual;
	jac(3, 4) = dcr.state.yd.dual;
	jac(3, 5) = dcr.state.zd.dual;	

	argperid.dual = 1.0;
	dcr = universal_cartesian(mu, qd, ed, incld, longnoded, argperid, tpd, epoch, s);
	argperid.dual = 0.0;

	jac(4, 0) = dcr.state.x.dual;
	jac(4, 1) = dcr.state.y.dual;
	jac(4, 2) = dcr.state.z.dual;
	jac(4, 3) = dcr.state.xd.dual;
	jac(4, 4) = dcr.state.yd.dual;
	jac(4, 5) = dcr.state.zd.dual;	

	tpd.dual = 1.0;
	dcr = universal_cartesian(mu, qd, ed, incld, longnoded, argperid, tpd, epoch, s);
	tpd.dual = 0.0;

	jac(5, 0) = dcr.state.x.dual;
	jac(5, 1) = dcr.state.y.dual;
	jac(5, 2) = dcr.state.z.dual;
	jac(5, 3) = dcr.state.xd.dual;
	jac(5, 4) = dcr.state.yd.dual;
	jac(5, 5) = dcr.state.zd.dual;

	return jac;
    }

    double principal_value(double theta)
    {
	theta -= 2.0*PI*floor(theta/(2.0*PI));
	return(theta);
    }

    Dual principal_value(Dual theta)
    {
	theta = theta - 2.0*PI*floor(theta.real/(2.0*PI));
	return(theta);
    }

    int universal_cometary(double mu, CartesianState state,
			   double& q, double& e, double& incl, double& longnode,
			   double& argperi, double& tp, double epochMJD_TDB=0.0){
	/*
	  Converts from a state vectors into cometary orbital elements
	  using the universal variable formulation

	  The input vector will determine the orientation
	  of the positional angles (i, Omega, omega)


	  Note that mu and the state vectors must have compatible units
	  As an example, if x is in au and vx are in au/days, mu must
	  be in (au^3)/days^2


	  Parameters
	  -----------
	  mu : float
	  Standard gravitational parameter GM (see note above about units)
	  x : float
	  x coordinate
	  y : float
	  y coordinate
	  z : float
	  z coordinate
	  vx : float
	  x velocity
	  vy : float
	  y velocity
	  vz : float
	  z velocity
	  epochMJD_TDB (float):
	  Epoch (in TDB) when the elements are defined (see note above about units)

	  Returns
	  ----------
	  float
	  Perihelion (see note above about units)
	  float
	  Eccentricity
	  float
	  Inclination (radians)
	  float
	  Longitude of ascending node (radians)
	  float
	  Argument of perihelion (radians)
	  float
	  Time of perihelion passage in TDB scale (see note above about units)
	*/

	/* find direction of angular momentum vector */
	double rxv_x = state.y * state.zd - state.z * state.yd;
	double rxv_y = state.z * state.xd - state.x * state.zd;
	double rxv_z = state.x * state.yd - state.y * state.xd;
	double hs = rxv_x * rxv_x + rxv_y * rxv_y + rxv_z * rxv_z;
	double h = sqrt(hs);

	double r = sqrt(state.x * state.x + state.y * state.y + state.z * state.z);
	double vs = state.xd * state.xd + state.yd * state.yd + state.zd * state.zd;
	double rdotv = state.x * state.xd + state.y * state.yd + state.z * state.zd;
	double rdot = rdotv / r;
	double p = hs / mu;

	incl = acos(rxv_z / h);

	if(rxv_x!=0.0 || rxv_y!=0.0) {
	    longnode = atan2(rxv_x, -rxv_y);
	} else {
	    longnode = 0.0;
	}

	double alpha = 2.0 * mu / r - vs;
	q = p / (1 + e);

	double ecostrueanom = p / r - 1.0;
	double esintrueanom = rdot * h / mu;
	e = sqrt(ecostrueanom * ecostrueanom + esintrueanom * esintrueanom);

	double trueanom; 
	if(esintrueanom!=0.0 || ecostrueanom!=0.0) {
	    trueanom = atan2(esintrueanom, ecostrueanom);
	} else {
	    trueanom = 0.0;
	}

	double cosnode = cos(longnode);
	double sinnode = sin(longnode);

	/* u is the argument of latitude */
	double rcosu = state.x * cosnode + state.y * sinnode;
	double rsinu = (state.y * cosnode - state.x * sinnode)/cos(incl);

	double u;
	if(rsinu!=0.0 || rcosu!=0.0) {
	    u = atan2(rsinu, rcosu);
	} else {
	    u = 0.0;
	}

	argperi = u - trueanom;
	argperi = principal_value(argperi);

	longnode = principal_value(longnode);    

	//eccanom = 2.0 * atan(sqrt((1.0 - e)/(1.0 + e)) * tan(trueanom/2.0));
	//meananom = eccanom - e * sin(eccanom);

	// There should a better way to handle this.
	// Branch on e at this point, until there's a better solution
	// Be careful with the e=1 transition.
	if(fabs(e - 1.0) < 1e-15){
	    e = 1.0;
	}

	// Try to do this in terms of gauss f and g functions and
	// Stumpff functions.  If that works, incorporate the result
	// back into the python version.
	//
	// r0 = q, r0dot = 0
    
	// f = r/p * (cosf - 1) + 1
	// f = 1 - (mu/r0) * s*s*c2;
    
	// g = r*rdot*sinf/sqrt(mu*p)
	// g = r0 * s*c1 + r0*r0dot * s*s*c2 = t - t0 - mu * s*s*s*c3
	// g = r0 * s*c1 = t - t0 - mu * s*s*s*c3
    
	// fdot = -sqrt(mu/(p*p*p))*(sinf+ e*sinf)
	// fdot = - (mu/(r*r0)) * s*c1
    
	// gdot = r0/p * (cosf - 1) + 1
	// gdot = 1 - (mu/r) * s*s*c2;

	// t - t0 = r0 * s*c1 + r0*r0dot * s*s*c2 + mu * s*s*s*c3
	// t - t0 = r0 * s*c1 + mu * s*s*s*c3
	// t - t0 = g + mu * s*s*s*c3

	// 1. Use these expressions to get s*s*c2 and s*c1
	// 2. Use s*s*c2 to get c0

	// alpha * s*s*c3(alpa*s*s) = 1/1! - c1(alpha*s*s)
	// -->
	// alpha * s*s*s*c3(alpha*s*s) = s - s*c1(alpha*s*s)
	// s = alpha * s*s*s*c3(alpha*s*s) + s*c1(alpha*s*s)
	// s = 

	int branch;
	if(e < 1){
	    // elliptical
	    branch = -1;
	    double eccanom = 2.0 * atan(sqrt((1.0 - e) / (1.0 + e)) * tan(trueanom / 2.0));
	    double meananom = eccanom - e * sin(eccanom);
	    meananom = principal_value(meananom);
	    double a = mu / alpha;
	    double mm = sqrt(mu / (a * a * a));
	    double per = TWOPI/mm;
	    // Pick the next pericenter passage
	    tp = per + epochMJD_TDB - meananom / mm;
	}else if(e == 1){
	    // parabolic
	    branch = 0;
	    double tf = tan(0.5 * trueanom);
	    double B = 0.5 * (tf * tf * tf + 3 * tf);
	    double mm = sqrt(mu / (p * p * p));
	    tp = epochMJD_TDB - B / (3 * mm);
	}else{
	    // hyperbolic
	    branch = 1;
	    double heccanom = 2.0 * atanh(sqrt((e - 1.0) / (e + 1.0)) * tan(trueanom / 2.0));
	    double N = e * sinh(heccanom) - heccanom;
	    double a = mu / alpha;
	    double mm = sqrt(-mu / (a * a * a));
	    tp = epochMJD_TDB - N / mm;
	}

	return branch;

    }

    int universal_cometary(double mu, DCartesianState state,
			   Dual& q, Dual& e, Dual& incl, Dual& longnode,
			   Dual& argperi, Dual& tp, Dual epochMJD_TDB=0.0){
	/*
	  Converts from a state vectors into cometary orbital elements
	  using the universal variable formulation

	  The input vector will determine the orientation
	  of the positional angles (i, Omega, omega)


	  Note that mu and the state vectors must have compatible units
	  As an example, if x is in au and vx are in au/days, mu must
	  be in (au^3)/days^2


	  Parameters
	  -----------
	  mu : float
	  Standard gravitational parameter GM (see note above about units)
	  x : float
	  x coordinate
	  y : float
	  y coordinate
	  z : float
	  z coordinate
	  vx : float
	  x velocity
	  vy : float
	  y velocity
	  vz : float
	  z velocity
	  epochMJD_TDB (float):
	  Epoch (in TDB) when the elements are defined (see note above about units)

	  Returns
	  ----------
	  float
	  Perihelion (see note above about units)
	  float
	  Eccentricity
	  float
	  Inclination (radians)
	  float
	  Longitude of ascending node (radians)
	  float
	  Argument of perihelion (radians)
	  float
	  Time of perihelion passage in TDB scale (see note above about units)
	*/

	/* find direction of angular momentum vector */
	Dual rxv_x = state.y * state.zd - state.z * state.yd;
	Dual rxv_y = state.z * state.xd - state.x * state.zd;
	Dual rxv_z = state.x * state.yd - state.y * state.xd;
	Dual hs = rxv_x * rxv_x + rxv_y * rxv_y + rxv_z * rxv_z;
	Dual h = sqrt(hs);

	Dual r = sqrt(state.x * state.x + state.y * state.y + state.z * state.z);
	Dual vs = state.xd * state.xd + state.yd * state.yd + state.zd * state.zd;
	Dual rdotv = state.x * state.xd + state.y * state.yd + state.z * state.zd;
	Dual rdot = rdotv / r;
	Dual p = hs / mu;

	incl = acos(rxv_z / h);

	if(rxv_x!=0.0 || rxv_y!=0.0) {
	    longnode = atan2(rxv_x, -rxv_y);
	} else {
	    longnode = 0.0;
	}

	Dual alpha = 2.0 * mu / r - vs;

	Dual ecostrueanom = p / r - 1.0;
	Dual esintrueanom = rdot * h / mu;
	e = sqrt(ecostrueanom * ecostrueanom + esintrueanom * esintrueanom);

	q = p / (1 + e);

	Dual trueanom; 
	if(esintrueanom!=0.0 || ecostrueanom!=0.0) {
	    trueanom = atan2(esintrueanom, ecostrueanom);
	} else {
	    trueanom = 0.0;
	}

	Dual cosnode = cos(longnode);
	Dual sinnode = sin(longnode);

	/* u is the argument of latitude */
	Dual rcosu = state.x * cosnode + state.y * sinnode;
	Dual rsinu = (state.y * cosnode - state.x * sinnode)/cos(incl);

	Dual u;
	if(rsinu!=0.0 || rcosu!=0.0) {
	    u = atan2(rsinu, rcosu);
	} else {
	    u = 0.0;
	}

	argperi = u - trueanom;
	argperi = principal_value(argperi);

	longnode = principal_value(longnode);    

	//eccanom = 2.0 * atan(sqrt((1.0 - e)/(1.0 + e)) * tan(trueanom/2.0));
	//meananom = eccanom - e * sin(eccanom);

	// There should a better way to handle this.
	// Branch on e at this point, until there's a better solution
	// Be careful with the e=1 transition.
	if(fabs(e.real - 1.0) < 1e-15){
	    e.real = 1.0;
	}

	// Try to do this in terms of gauss f and g functions and
	// Stumpff functions.  If that works, incorporate the result
	// back into the python version.
	//
	// r0 = q, r0dot = 0
    
	// f = r/p * (cosf - 1) + 1
	// f = 1 - (mu/r0) * s*s*c2;
    
	// g = r*rdot*sinf/sqrt(mu*p)
	// g = r0 * s*c1 + r0*r0dot * s*s*c2 = t - t0 - mu * s*s*s*c3
	// g = r0 * s*c1 = t - t0 - mu * s*s*s*c3
    
	// fdot = -sqrt(mu/(p*p*p))*(sinf+ e*sinf)
	// fdot = - (mu/(r*r0)) * s*c1
    
	// gdot = r0/p * (cosf - 1) + 1
	// gdot = 1 - (mu/r) * s*s*c2;

	// t - t0 = r0 * s*c1 + r0*r0dot * s*s*c2 + mu * s*s*s*c3
	// t - t0 = r0 * s*c1 + mu * s*s*s*c3
	// t - t0 = g + mu * s*s*s*c3

	// 1. Use these expressions to get s*s*c2 and s*c1
	// 2. Use s*s*c2 to get c0

	// alpha * s*s*c3(alpa*s*s) = 1/1! - c1(alpha*s*s)
	// -->
	// alpha * s*s*s*c3(alpha*s*s) = s - s*c1(alpha*s*s)
	// s = alpha * s*s*s*c3(alpha*s*s) + s*c1(alpha*s*s)
	// s = 

	int branch;
	if(e < 1){
	    // elliptical
	    branch = -1;
	    Dual eccanom = 2.0 * atan(sqrt((1.0 - e) / (1.0 + e)) * tan(trueanom / 2.0));
	    Dual meananom = eccanom - e * sin(eccanom);
	    meananom = principal_value(meananom);
	    Dual a = mu / alpha;
	    Dual mm = sqrt(mu / (a * a * a));
	    Dual per = TWOPI/mm;
	    // Pick the next pericenter passage
	    tp = per + epochMJD_TDB - meananom / mm;
	}else if(e == 1){
	    // parabolic
	    branch = 0;
	    Dual tf = tan(0.5 * trueanom);
	    Dual B = 0.5 * (tf * tf * tf + 3 * tf);
	    Dual mm = sqrt(mu / (p * p * p));
	    tp = epochMJD_TDB - B / (3 * mm);
	}else{
	    // hyperbolic
	    branch = 1;
	    Dual heccanom = 2.0 * atanh(sqrt((e - 1.0) / (e + 1.0)) * tan(trueanom / 2.0));
	    Dual N = e * sinh(heccanom) - heccanom;
	    Dual a = mu / alpha;
	    Dual mm = sqrt(-mu / (a * a * a));
	    tp = epochMJD_TDB - N / mm;
	}

	return branch;

    }

    Matrix6d universal_cometary_jacobian(double mu, DCartesianState dstate,
					 Dual& qd, Dual& ed, Dual& incld, Dual& longnoded,
					 Dual& argperid, Dual& tpd, Dual epoch=0.0){

	Matrix6d cjac;

	dstate.x.dual = 1.0;
	universal_cometary(mu, dstate, qd, ed, incld, longnoded, argperid, tpd, epoch);
	dstate.x.dual = 0.0;

	cjac(0, 0) = qd.dual;
	cjac(0, 1) = ed.dual;
	cjac(0, 2) = incld.dual;
	cjac(0, 3) = longnoded.dual;
	cjac(0, 4) = argperid.dual;
	cjac(0, 5) = tpd.dual;

	dstate.y.dual = 1.0;
	universal_cometary(mu, dstate, qd, ed, incld, longnoded, argperid, tpd, epoch);
	dstate.y.dual = 0.0;

	cjac(1, 0) = qd.dual;
	cjac(1, 1) = ed.dual;
	cjac(1, 2) = incld.dual;
	cjac(1, 3) = longnoded.dual;
	cjac(1, 4) = argperid.dual;
	cjac(1, 5) = tpd.dual;

	dstate.z.dual = 1.0;
	universal_cometary(mu, dstate, qd, ed, incld, longnoded, argperid, tpd, epoch);
	dstate.z.dual = 0.0;

	cjac(2, 0) = qd.dual;
	cjac(2, 1) = ed.dual;
	cjac(2, 2) = incld.dual;
	cjac(2, 3) = longnoded.dual;
	cjac(2, 4) = argperid.dual;
	cjac(2, 5) = tpd.dual;

	dstate.xd.dual = 1.0;
	universal_cometary(mu, dstate, qd, ed, incld, longnoded, argperid, tpd, epoch);
	dstate.xd.dual = 0.0;

	cjac(3, 0) = qd.dual;
	cjac(3, 1) = ed.dual;
	cjac(3, 2) = incld.dual;
	cjac(3, 3) = longnoded.dual;
	cjac(3, 4) = argperid.dual;
	cjac(3, 5) = tpd.dual;

	dstate.yd.dual = 1.0;
	universal_cometary(mu, dstate, qd, ed, incld, longnoded, argperid, tpd, epoch);
	dstate.yd.dual = 0.0;

	cjac(4, 0) = qd.dual;
	cjac(4, 1) = ed.dual;
	cjac(4, 2) = incld.dual;
	cjac(4, 3) = longnoded.dual;
	cjac(4, 4) = argperid.dual;
	cjac(4, 5) = tpd.dual;

	dstate.zd.dual = 1.0;
	universal_cometary(mu, dstate, qd, ed, incld, longnoded, argperid, tpd, epoch);
	dstate.zd.dual = 0.0;

	cjac(5, 0) = qd.dual;
	cjac(5, 1) = ed.dual;
	cjac(5, 2) = incld.dual;
	cjac(5, 3) = longnoded.dual;
	cjac(5, 4) = argperid.dual;
	cjac(5, 5) = tpd.dual;

	return cjac;

    }
    
    struct reb_particle read_initial_conditions(const char *ic_file_name, double *epoch)
    {

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

    void print_initial_condition(struct reb_particle p0, double epoch)
    {

        printf("%lf %.16le %.16le %.16le %.16le %.16le %.16le\n", epoch, p0.x, p0.y, p0.z, p0.vx, p0.vy, p0.vz);

        return;
    }

    void read_detections(const char *data_file_name,
                         std::vector<detection> &detections,
                         std::vector<double> &times)
    {

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

        while (fscanf(detections_file, "%s %s %s %s %lf %lf %lf %lf %lf %lf %lf %lf\n",
                      objID, obsCode, mag_str, filt, &jd_tdb, &theta_x, &theta_y, &theta_z, &xe, &ye, &ze, &ast_unc) != EOF)
        {
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
            this_det.ra_unc = (ast_unc / 206265);
            this_det.dec_unc = (ast_unc / 206265);

            // compute A and D vectors, the local tangent plane trick from Danby.

            double Ax =  -theta_y;
            double Ay =   theta_x;
            double Az =   0.0;

            double A = sqrt(Ax * Ax + Ay * Ay + Az * Az);
            Ax /= A;
            Ay /= A;
            Az /= A;

            this_det.Ax = Ax;
            this_det.Ay = Ay;
            this_det.Az = Az;

            double sd = theta_z;
            double cd = sqrt(1-theta_z*theta_z);
            double ca = theta_x/cd;
            double sa = theta_y/cd;
        
            double Dx = -sd*ca;
            double Dy = -sd*sa;
            double Dz = cd;        

            double D = sqrt(Dx * Dx + Dy * Dy + Dz * Dz);
            Dx /= D;
            Dy /= D;
            Dz /= D;

            this_det.Dx = Dx;
            this_det.Dy = Dy;
            this_det.Dz = Dz;

            detections.push_back(this_det);
            times.push_back(jd_tdb);
        }
    }

    void compute_single_residuals(struct assist_ephem *ephem,
                                  struct assist_extras *ax,
                                  int var,
                                  Observation this_det,
                                  residuals &resid,
                                  partials &parts)
    {

        // Takes a detection/observation and
        // a simulation as arguments.
        // Returns the model observation, the
        // residuals, and the partial derivatives.

        struct reb_simulation *r = ax->sim;

        double jd_tdb = this_det.epoch;

        double xe = this_det.observer_position[0];
        double ye = this_det.observer_position[1];
        double ze = this_det.observer_position[2];

        Eigen::Vector3d Av = this_det.a_vec;
        Eigen::Vector3d Dv = this_det.d_vec;

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
        double rho = sqrt(rho_x * rho_x + rho_y * rho_y + rho_z * rho_z);

        rho_x /= rho;
        rho_y /= rho;
        rho_z /= rho;

        // Put the residuals into a struct, so that the
        // struct can be easily managed in a vector.
        resid.x_resid = -(rho_x * Ax + rho_y * Ay + rho_z * Az);
        resid.y_resid = -(rho_x * Dx + rho_y * Dy + rho_z * Dz);

        // 6. Calculate the partial deriviatives of the model
        //    observations with respect to the initial conditions

        double dxk[6], dyk[6], dzk[6];
        double drho_x[6], drho_y[6], drho_z[6];

        for (size_t i = 0; i < 6; i++)
        {
            size_t vn = var + i;
            dxk[i] = r->particles[vn].x;
            dyk[i] = r->particles[vn].y;
            dzk[i] = r->particles[vn].z;
        }

        double invd = 1. / rho;

        double ddist[6];
        double dx_resid[6];
        double dy_resid[6];
        for (size_t i = 0; i < 6; i++)
        {

            // Derivative of topocentric distance w.r.t. parameters
            ddist[i] = rho_x * dxk[i] + rho_y * dyk[i] + rho_z * dzk[i];

            drho_x[i] = dxk[i] * invd - rho_x * invd * ddist[i] - r->particles[0].vx * ddist[i] * invd / SPEED_OF_LIGHT;
            drho_y[i] = dyk[i] * invd - rho_y * invd * ddist[i] - r->particles[0].vy * ddist[i] * invd / SPEED_OF_LIGHT;
            drho_z[i] = dzk[i] * invd - rho_z * invd * ddist[i] - r->particles[0].vz * ddist[i] * invd / SPEED_OF_LIGHT;

            dx_resid[i] = -(drho_x[i] * Ax + drho_y[i] * Ay + drho_z[i] * Az);
            dy_resid[i] = -(drho_x[i] * Dx + drho_y[i] * Dy + drho_z[i] * Dz);
        }

        // Likewise, put the partials into a struct so that they can be more easily
        // managed.

        for (size_t i = 0; i < 6; i++)
        {
            parts.x_partials.push_back(dx_resid[i]);
        }
        for (size_t i = 0; i < 6; i++)
        {
            parts.y_partials.push_back(dy_resid[i]);
        }
    }

    // This routine requires that resid_vec and partials_vec are
    // preallocated because the indices in out_seq will be used for
    // random access of locations in those vectors.
    // Also, in_seq and out_seq must be of the same length.
    void compute_residuals_sequence(struct assist_ephem *ephem,
                                    struct reb_particle p0, double epoch,
                                    std::vector<Observation> &detections,
                                    std::vector<residuals> &resid_vec,
                                    std::vector<partials> &partials_vec,
                                    std::vector<size_t> &in_seq,
                                    std::vector<size_t> &out_seq)
    {

        // Pass in this simulation stuff to keep it flexible
        struct reb_simulation *r = reb_simulation_create();
        struct assist_extras *ax = assist_attach(r, ephem);

        // 0. Set initial time, relative to ephem->jd_ref
        r->t = epoch - ephem->jd_ref;

        // 1. Add the main particle to the REBOUND simulation.
        reb_simulation_add(r, p0);

        // 2. incorporate the variational particles
        int var;
        add_variational_particles(r, 0, &var);

        // 3. iterate over a sequence of detections.
        for (size_t i = 0; i < in_seq.size(); ++i)
        {

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

    void compute_residuals(struct assist_ephem *ephem,
                           struct reb_particle p0, double epoch,
                           std::vector<Observation> &detections,
                           std::vector<residuals> &resid_vec,
                           std::vector<partials> &partials_vec,
                           std::vector<size_t> &forward_in_seq,
                           std::vector<size_t> &forward_out_seq,
                           std::vector<size_t> &reverse_in_seq,
                           std::vector<size_t> &reverse_out_seq)
    {

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
    void compute_dX(std::vector<residuals> &resid_vec,
                    std::vector<partials> &partials_vec,
                    Eigen::SparseMatrix<double> W,
                    Eigen::MatrixXd &dX,
                    Eigen::MatrixXd &C,
                    Eigen::MatrixXd &chi2,
                    Eigen::Matrix<double, 6, 1> &grad,
                    double lambda)
    {

        const int mlength = (int)partials_vec.size();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
        Eigen::MatrixXd eye = MatrixXd::Identity(6, 6);
        B.resize(mlength * 2, 6);

        for (size_t i = 0; i < partials_vec.size(); i++)
        {
            for (size_t j = 0; j < 6; j++)
            {
                B(2 * i, j) = partials_vec[i].x_partials[j];
                B(2 * i + 1, j) = partials_vec[i].y_partials[j];
            }
        }

        Eigen::MatrixXd resid_v(mlength * 2, 1);
        for (size_t i = 0; i < resid_vec.size(); i++)
        {
            resid_v(2 * i) = resid_vec[i].x_resid;
            resid_v(2 * i + 1) = resid_vec[i].y_resid;
        }

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Bt = B.transpose();
        C = Bt * W * B + lambda * eye; // This is where the extra term for LM is.

        grad = Bt * W * resid_v;

        dX = C.colPivHouseholderQr().solve(-grad);
        // An alternative looks like this.  It's probably less stable.
        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G = C.inverse();
        // dX = -G * Bt * W * resid_v;

        chi2 = resid_v.transpose() * W * resid_v;
        // reset for inverse covariance
        C -= lambda * eye;
    }

    void identify_outliers(std::vector<detection> &detections,
                           std::vector<residuals> &resid_vec,
                           std::vector<bool> &reject)
    {
    }

    void print_residuals(std::vector<detection> &detections,
                         std::vector<residuals> &resid_vec,
                         std::vector<partials> &partials_vec)
    {

        for (size_t j = 0; j < resid_vec.size(); j++)
        {

            detection this_det = detections[j];

            printf("%lf %s ", this_det.jd_tdb, this_det.obsCode.c_str());

            residuals resids = resid_vec[j];
            printf("%7.3lf %7.3lf ", resids.x_resid * 206265, resids.y_resid * 206265);

            partials parts = partials_vec[j];

            for (size_t i = 0; i < 6; i++)
            {

                printf("%7.3le %7.3le ", parts.x_partials[i], parts.y_partials[i]);
            }

            printf("%7.3le %7.3le ", this_det.ra_unc * 206265, this_det.dec_unc * 206265);

            printf("\n");
        }
    }

    int converged(Eigen::MatrixXd dX, double eps, double chi2, size_t ndof, double thresh)
    {

        if ((chi2 > ndof) > thresh)
        {
            return 2;
        }
        for (size_t i = 0; i < dX.size(); i++)
        {
            if (abs(dX(i)) > eps)
            {
                return 0;
            }
        }
        return 1;
    }

    typedef Eigen::Triplet<double> T;

    Eigen::SparseMatrix<double> get_weight_matrix(std::vector<Observation> &detections)
    {
        const int mlength = (int)detections.size();
        std::vector<T> tripletList;
        tripletList.reserve(2 * mlength);
        Eigen::SparseMatrix<double> W(mlength * 2, mlength * 2);

        for (size_t i = 0; i < mlength; i++)
        {
            double x_unc = *detections[i].ra_unc;
            double w2_x = 1.0 / (x_unc * x_unc);
            tripletList.push_back(T(2 * i, 2 * i, w2_x));
            double y_unc = *detections[i].dec_unc;
            double w2_y = 1.0 / (y_unc * y_unc);
            tripletList.push_back(T(2 * i + 1, 2 * i + 1, w2_y));
        }
        W.setFromTriplets(tripletList.begin(), tripletList.end());

        return W;
    }

    void create_sequences(std::vector<double> &times,
                          double epoch,
                          std::vector<size_t> &reverse_in_seq,
                          std::vector<size_t> &reverse_out_seq,
                          std::vector<size_t> &forward_in_seq,
                          std::vector<size_t> &forward_out_seq)
    {

        for (size_t i = 0; i < times.size(); i++)
        {
            if (times[i] > epoch)
            {
                forward_in_seq.push_back(i);
                forward_out_seq.push_back(i);
            }
            else
            {
                reverse_in_seq.push_back(i);
                reverse_out_seq.push_back(i);
            }
        }

        std::reverse(reverse_in_seq.begin(), reverse_in_seq.end());
        std::reverse(reverse_out_seq.begin(), reverse_out_seq.end());
    }

    int orbit_fit(struct assist_ephem *ephem,
                  struct reb_particle &p0,
                  double epoch,                            // Result from gauss
                  std::vector<Observation> &detections, // In Observation
                  std::vector<residuals> &resid_vec,    // Compute as part of run
                  std::vector<partials> &partials_vec,    // Compute as part of run
                  size_t &iters,                        // runtime
                  double &chi2_final,                    // result
                  Eigen::MatrixXd &cov,
                  double eps, // constant
                  size_t iter_max)
    { // runtime

        std::vector<size_t> reverse_in_seq;
        std::vector<size_t> reverse_out_seq;

        std::vector<size_t> forward_in_seq;
        std::vector<size_t> forward_out_seq;

        std::vector<double> times(detections.size());

        for (int i = 0; i < detections.size(); i++)
        {
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
        double lambda = (206265.0 * 206265.0) / 1000;
        for (iters = 0; iters < iter_max; iters++)
        {

            compute_residuals(ephem, p0, epoch,
                              detections,
                              resid_vec,
                              partials_vec,
                              forward_in_seq,
                              forward_out_seq,
                              reverse_in_seq,
                              reverse_out_seq);

            Eigen::Matrix<double, 6, 1> grad;
            compute_dX(resid_vec, partials_vec, W, dX, C, chi2, grad, lambda);

            double chi2_d = chi2(0, 0);

            double rho = (chi2_prev - chi2_d) / (dX.transpose() * (lambda * dX - grad)).norm();

            if (rho > rho_accept)
            {

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
            }
            else
            {

                // Reject the step
                // leave the state
                // increase lambda
                // repeat, unless too many iterations

                lambda *= 2.0;
            }

            size_t ndof = detections.size() * 2 - 6;
            double thresh = 10;
            int cflag = converged(dX, eps, chi2_d, ndof, thresh);
            if (cflag)
            {
                flag = 0;
                chi2_final = chi2_d;
                break;
            }
        }

        size_t ndof = detections.size() * 2 - 6;
        double thresh = 10.0;
        if ((chi2_final / ndof) > thresh)
        {
            flag = 2;
        }

        cov = C.inverse();

        return flag;
    }

    // Go through the detections in reverse order, looking for
    // a set of three detections such that each adjacent pair is
    // separated by more than interval_min and less than interval_max.
    std::vector<std::vector<size_t>> IOD_indices(std::vector<Observation> &detections,
                                                 double interval0_min,
                                                 double interval0_max,
                                                 double interval1_min,
                                                 double interval1_max,
                                                 size_t max_count,
                                                 size_t i_start)
    {

        size_t cnt = 0;
        std::vector<std::vector<size_t>> res;
        for (int i = i_start; i >= 2; i--)
        {

            size_t idx_i = (size_t)i;
            Observation d2 = detections[idx_i];
            double t2 = d2.epoch;

            for (int j = i - 1; j >= 1; j--)
            {

                size_t idx_j = (size_t)j;
                Observation d1 = detections[j];
                double t1 = d1.epoch;

                if (fabs(t2 - t1) < interval0_min || fabs(t2 - t1) >= interval0_max)
                    continue;

                for (int k = j - 1; k >= 0; k--)
                {

                    size_t idx_k = (size_t)k;
                    Observation d0 = detections[idx_k];
                    double t0 = d0.epoch;

                    if (fabs(t1 - t0) < interval1_min || fabs(t1 - t0) >= interval1_max)
                        continue;

                    cnt++;

                    if (cnt > max_count)
                        return res;

                    res.push_back({idx_k, idx_j, idx_i});
                    // printf("%lu %lu %lu %lf %lf %lf\n", idx_k, idx_j, idx_i, t0, t1, t2);
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

    struct assist_ephem * get_ephem(std::string cache_dir)
    {
        std::string ephem_kernel = cache_dir + "/linux_p1550p2650.440";
        std::string small_bodies_kernel = cache_dir + "/sb441-n16.bsp";

        // Allocate space for the string plus the null-terminator
        char *ephem_kernel_char = new char[ephem_kernel.length() + 1];
        std::strncpy(ephem_kernel_char, ephem_kernel.c_str(), ephem_kernel.length());
        ephem_kernel_char[ephem_kernel.length()] = '\0'; // Ensure null-termination

        char *small_bodies_kernel_char = new char[small_bodies_kernel.length() + 1];
        std::strncpy(small_bodies_kernel_char, small_bodies_kernel.c_str(), small_bodies_kernel.length());
        small_bodies_kernel_char[small_bodies_kernel.length()] = '\0'; // Ensure null-termination

        struct assist_ephem *ephem = assist_ephem_create(ephem_kernel_char, small_bodies_kernel_char);
        if (!ephem)
        {
            printf("Cannot create ephemeris structure.\n");
            exit(-1);
        }
        return ephem;
    }

    FitResult run_from_vector_with_initial_guess(struct assist_ephem *ephem, FitResult initial_guess, std::vector<Observation> &detections)
    {
        int success = 1;
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

        p1.x = initial_guess.state[0];
        p1.y = initial_guess.state[1];
        p1.z = initial_guess.state[2];
        p1.vx = initial_guess.state[3];
        p1.vy = initial_guess.state[4];
        p1.vz = initial_guess.state[5];
        double epoch = initial_guess.epoch;

	double GMtotal = 0.0002963092748799319;
	double q, e, incl, longnode, argperi, tp;
	CartesianState state = {p1.x, p1.y, p1.z, p1.vx, p1.vy, p1.vz};
	universal_cometary(GMtotal, state, q, e, incl, longnode, argperi, tp, epoch);
	printf("%lf %lf %lf %lf %lf %lf\n", q, e, incl, longnode, argperi, tp);

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
            iter_max);

        dof = 2 * detections.size() - 6;

        FitResult result;
	
        result.method = "orbit_fit";
        result.flag = flag;

        result.epoch = epoch;
        result.csq = chi2_final;
        result.ndof = dof;
        result.niter = iters;
        result.state[0] = p1.x;
        result.state[1] = p1.y;
        result.state[2] = p1.z;
        result.state[3] = p1.vx;
        result.state[4] = p1.vy;
        result.state[5] = p1.vz;

	if (flag == 0) {
            // Populate our covariance matrix
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    // flatten the covariance matrix
                    result.cov[(i * 6) + j] = cov(i, j);
                }
            }
        }

	return result;

    }

#ifdef Py_PYTHON_H
    static void orbit_fit_bindings(py::module &m)
    {
        py::class_<assist_ephem>(m, "assist_ephem");
        m.def("orbit_fit", &orbit_fit::orbit_fit, R"pbdoc(Core orbit fit function.)pbdoc");
        m.def("get_ephem", &orbit_fit::get_ephem, R"pbdoc(get ephemeris)pbdoc");
        m.def("run_from_vector_with_initial_guess", &orbit_fit::run_from_vector_with_initial_guess, R"pbdoc(
                Takes an assist_ephem object, a vector of observations and an initial guess
                and runs orbit fit.
            )pbdoc");
    }
#endif /* Py_PYTHON_H */

} // namespace: orbit_fit
