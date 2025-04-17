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
// g++ -std=c++17 -I../..//src/ -I../../../rebound/src/ -I/Users/mholman/eigen-3.4.0 -Wl,-rpath,./ -Wpointer-arith -D_GNU_SOURCE -O3 -fPIC -I/usr/local/include -Wall -g  -Wno-unknown-pragmas -D_APPLE -DSERVER -DGITHASH=e99d8d73f0aa7fb7bf150c680a2d581f43d3a8be gauss_main.cpp gauss.cpp -L. -lassist -lrebound -L/usr/local/lib -o gauss

#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>

#include "orbit_fit.h"
#include "gauss.h"
#include <chrono>

extern "C"
{
#include "rebound.h"
#include "assist.h"
}
using namespace Eigen;
using std::cout;

double AU_M = 149597870700;
double SPEED_OF_LIGHT = 2.99792458e8 * 86400.0 / AU_M;

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

        double Ax = theta_z;
        double Ay = 0.0;
        double Az = -theta_x;

        double A = sqrt(Ax * Ax + Ay * Ay + Az * Az);
        Ax /= A;
        Ay /= A;
        Az /= A;

        this_det.Ax = Ax;
        this_det.Ay = Ay;
        this_det.Az = Az;

        double Dx = -theta_x * theta_y;
        double Dy = theta_x * theta_x + theta_z * theta_z;
        double Dz = -theta_z * theta_y;

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

int main(int argc, char *argv[])
{

    // ephemeris files should be passed in or put in a config
    // file
    char ephemeris_filename[128] = "../../data/linux_p1550p2650.440";
    char small_bodies_filename[128] = "../../data/sb441-n16.bsp";
    struct assist_ephem *ephem = assist_ephem_create(
        ephemeris_filename,
        small_bodies_filename);
    if (!ephem)
    {
        printf("Cannot create ephemeris structure.\n");
        exit(-1);
    }

    std::vector<detection> detections;
    std::vector<double> times;

    // Read the observations
    char detections_filename[128];
    sscanf(argv[1], "%s", detections_filename);
    read_detections(detections_filename, detections, times);

    size_t id0, id1, id2;
    sscanf(argv[2], "%lu", &id0);
    sscanf(argv[3], "%lu", &id1);
    sscanf(argv[4], "%lu", &id2);

    // Make this general
    double GMtotal = 0.00029630927487993194;

    detection d0 = detections[id0];
    detection d1 = detections[id1];
    detection d2 = detections[id2];

    auto start = std::chrono::high_resolution_clock::now();
    std::optional<std::vector<gauss_soln>> res = gauss(GMtotal, d0, d1, d2, 0.0001, SPEED_OF_LIGHT);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    if (res.has_value())
    {
        for (size_t i = 0; i < res.value().size(); i++)
        {
            printf("%lu %lf %lf %lf %lf %lf %lf %lf %lf\n", i, res.value()[i].root, res.value()[i].epoch, res.value()[i].x, res.value()[i].y, res.value()[i].z,
                   res.value()[i].vx, res.value()[i].vy, res.value()[i].vz);
        }
    }
    else
    {
        printf("gauss failed\n");
        exit(1);
    }

    assist_ephem_free(ephem);
}
