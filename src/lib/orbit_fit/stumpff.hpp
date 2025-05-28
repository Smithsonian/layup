#ifndef STUMPFF_H
#define STUMPFF_H

#include "mattodiff.hpp"

typedef struct
{
double c0;
double c1;
double c2;
double c3;
} Stumpff;

Stumpff stumpff(double x){
/*
Computes the Stumpff function c_k(x) for k = 0, 1, 2, 3

Parameters
----------
x : float
Argument of the Stumpff function

Returns
---------
Stumpff struct
c_0(x) : float
c_1(x) : float
c_2(x) : float
c_3(x) : float
*/

int n = 0;
double xm = 0.1;

while(fabs(x) > xm){
n += 1;
x /= 4;
}

double c2 = (
1 - x * (1 - x * (1 - x * (1 - x * (1 - x * (1 - x / 182.0) / 132.0) / 90.0) / 56.0) / 30.0) / 12.0
) / 2.0;
double c3 = (
1 - x * (1 - x * (1 - x * (1 - x * (1 - x * (1 - x / 210.0) / 156.0) / 110.0) / 72.0) / 42.0) / 20.0
) / 6.0;

double c1 = 1.0 - x * c3;
double c0 = 1.0 - x * c2;

while(n > 0){
n -= 1;
c3 = (c2 + c0 * c3) / 4.0;
c2 = c1 * c1 / 2.0;
c1 = c0 * c1;
c0 = 2.0 * c0 * c0 - 1.0;
}

Stumpff res;
res.c0 = c0;
res.c1 = c1;
res.c2 = c2;
res.c3 = c3;

return res;
}

typedef struct
{
Dual c0;
Dual c1;
Dual c2;
Dual c3;
} DStumpff;

DStumpff dstumpff(Dual x){
/*
Computes the Stumpff function c_k(x) for k = 0, 1, 2, 3

Parameters
----------
x : Dual
Argument of the Stumpff function

Returns
---------
Stumpff struct
c_0(x) : Dual
c_1(x) : Dual
c_2(x) : Dual
c_3(x) : Dual
*/

int n = 0;
double xm = 0.1;

while(fabs(x.real) > xm){
n += 1;
x = x / 4.0;
}

Dual c2 = (
1 - x * (1 - x * (1 - x * (1 - x * (1 - x * (1 - x / 182.0) / 132.0) / 90.0) / 56.0) / 30.0) / 12.0
) / 2.0;
Dual c3 = (
1 - x * (1 - x * (1 - x * (1 - x * (1 - x * (1 - x / 210.0) / 156.0) / 110.0) / 72.0) / 42.0) / 20.0
) / 6.0;

Dual c1 = 1.0 - x * c3;
Dual c0 = 1.0 - x * c2;

while(n > 0){
n -= 1;
c3 = (c2 + c0 * c3) / 4.0;
c2 = c1 * c1 / 2.0;
c1 = c0 * c1;
c0 = 2.0 * c0 * c0 - 1.0;
}

DStumpff res;
res.c0 = c0;
res.c1 = c1;
res.c2 = c2;
res.c3 = c3;

return res;
}

#endif
