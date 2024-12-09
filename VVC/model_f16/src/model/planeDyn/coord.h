
/* 
 * Coordinate conversion function
 * The US mode coordinate
 * Created by whh on 20210427
 */

#ifndef _COORD_H
#define _COORD_H

#include <math.h>
 
void body2wind(double bc[3], double wc[3], double alpha, double beta, bool deg=false);
void wind2body(double wc[3], double bc[3], double alpha, double beta, bool deg=false);
void stability2body(double sc[3], double bc[3], double alpha, double beta, bool deg=false);
void body2stability(double bc[3], double sc[3], double alpha, double beta, bool deg=false);
void body2ground(double* bc, double* gc, double phi, double theta, double psi, bool deg=false);
void ground2body(double* gc, double* bc, double phi, double theta, double psi, bool deg=false);

#endif