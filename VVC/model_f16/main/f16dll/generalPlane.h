/**
* generalPlane.h contains declarations necessary for the generation of generalPlane.dll file
* Created by Weijia on 2018/04/20
* Last modified by wrr on 2020/10/28
*/

#ifndef _GENERALPLANE_H
#define _GENERALPLANE_H

#ifdef __unix
#define GENERALPLANE_API
#else

#ifdef F16DLL_EXPORTS
#define GENERALPLANE_API __declspec(dllexport)
#else
#define GENERALPLANE_API __declspec(dllimport)
#endif

#endif

#include "../../src/model/f16/f16Plane.h"

extern "C"{
    GENERALPLANE_API void* init(double h0, double v0, double stepTime, int order, char* libName);
    GENERALPLANE_API void reset(void* planePtr);
    GENERALPLANE_API void stepsim(void* planePtr, double ail, double ele, double rud, double pla, double* stateOutput);
	GENERALPLANE_API void stepsimthr(void* planePtr, double ail, double ele, double rud, double thrust, double* stateOutput);
    GENERALPLANE_API void allSetTo0(void* planePtr);
    GENERALPLANE_API void setlla(void* planePtr, double lon, double lat);
    GENERALPLANE_API void setAttitude(void* planePtr, double phi_out, double theta_out, double psi_out);
	GENERALPLANE_API void setPQR(void* planePtr, double p_out, double q_out, double r_out);
	GENERALPLANE_API void setPhi(void* planePtr, double phi_out);
	GENERALPLANE_API void setTheta(void* planePtr, double theta_out);
	GENERALPLANE_API void setPsi(void* planePtr, double psi_out);
    GENERALPLANE_API double getMaxThrust(void* planePtr, double v, double h);
	GENERALPLANE_API void getFMDCS(void* planePtr, double* fmout);
	GENERALPLANE_API void setStateDCS(void* planePtr, double* stsout);
	GENERALPLANE_API void loadPlaneInfo(void* planePtr, double* stin, double* ctrlin);
	GENERALPLANE_API void getStates(void* planePtr, double* stateOutput);
	GENERALPLANE_API void setPos(void* planePtr, double npos_out, double epos_out);
	GENERALPLANE_API void deletePtr(void* planePtr);
	GENERALPLANE_API void saveStates(void* planePtr);
	GENERALPLANE_API void loadStates(void* planePtr);
	GENERALPLANE_API void setLog(void* planePtr, char* logName);
}

typedef void* (*INIT)(double, double, double, int, char*);
typedef void (*RESET)(void*);
typedef void (*STEPSIM)(void*, double, double, double, double, double*);
typedef void (*STEPSIMTHR)(void*, double, double, double, double, double*);
typedef void (*ALLSETTO0)(void*);
typedef void (*SETLLA)(void*, double, double);
typedef void (*SETATTITUDE)(void*, double, double, double);
typedef void (*SETPHI)(void*, double);
typedef void (*SETTHETA)(void*, double);
typedef void (*SETPSI)(void*, double);
typedef void (*GETMAXTHRUST)(void*, double, double);
typedef void (*GETFMDCS)(void*, double*);
typedef void (*SETSTATEDCS)(void*, double*);
typedef void (*LOADPLANEINFO)(void*, double*, double*);
typedef void (*GETSTATES)(void*, double*);
typedef void (*SETPOS)(void*, double, double);
typedef void (*SETLOG)(void*, char*);

#endif