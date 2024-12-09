#ifndef _TOOLSDYN_H
#define _TOOLSDYN_H

#include <math.h>
#include <stdio.h>

#define g 9.81
#define grav 9.81
#define PI 3.141592654
#define R2D 57.29577951

void atmos(double altt, double vtt, double *coeff);
void accels(double *state, double *xdot, double *y);
int fix(double in);
int sign(double in);
void eularToQuater_fromRu(double eangles[], double qangles[]);
void quaterToEular_fromRu(double qangles[], double eangles[]);
void quaterCal_fromRu(double wx, double wy, double wz, double dt, double qangles[]);
void eularToQuater(double eangles[], double qangles[]);
void quaterToEular(double qangles[], double eangles[]);
double TAS2IAS(double TAS, double alt);
double IAS2TAS(double TAS, double alt);
double soundSpeed(double alt);
double airDensity(double alt);
double muCal(double vh, double v);

template<class T>
T setInRange(T target, T upperBound, T lowerBound = 0){
	if(upperBound < lowerBound){
		T temp = upperBound;
		upperBound = lowerBound;
		lowerBound = temp;
	}
	if(target > upperBound) return upperBound;
	if(target < lowerBound) return lowerBound;
	return target;
}

#endif //_TOOLSDYN_H