/**
* generalplane->cpp is a realization of generalplane->dll based on the use of F16model class
* Created by Weijia on 2018/04/20
* Last modified by Wrr on 2020/10/28
*/
#include "generalPlane.h"


GENERALPLANE_API void* init(double h0, double v0, double stepTime, int order, char* libName)
{
	F16Plane *planePtr = new F16Plane(h0, v0, stepTime, order, libName);
	return planePtr;
}

GENERALPLANE_API void setlla(void* planePtr, double lon, double lat)
{
	((F16Plane*)planePtr)->setLLA(lon, lat);
}

GENERALPLANE_API void deletePtr(void* planePtr)
{
	delete ((F16Plane*)planePtr);
}

GENERALPLANE_API void setPos(void* planePtr, double npos_out, double epos_out)
{
	((F16Plane*)planePtr)->setPos(npos_out, epos_out);
}

GENERALPLANE_API void reset(void* planePtr)
{
	((F16Plane*)planePtr)->reset();
}

GENERALPLANE_API void stepsim(void* planePtr, double ail, double ele, double rud, double pla, double* stateOutput)
{
	// ((F16Plane*)planePtr)->setDamage(stateOutput);
	((F16Plane*)planePtr)->step_frac(ail, ele, rud, pla);
	((F16Plane*)planePtr)->getStatesForGame(stateOutput);
}

GENERALPLANE_API void stepsimthr(void* planePtr, double ail, double ele, double rud, double thrust, double* stateOutput)
{
	// ((F16Plane*)planePtr)->setDamage(stateOutput);
	((F16Plane*)planePtr)->step(ail, ele, rud, thrust);
	((F16Plane*)planePtr)->getStatesForGame(stateOutput);
} 

GENERALPLANE_API void allSetTo0(void* planePtr)
{
	((F16Plane*)planePtr)->allSetTo0();
}

GENERALPLANE_API void setAttitude(void* planePtr, double phi_out, double theta_out, double psi_out)
{
	((F16Plane*)planePtr)->setAttitude(phi_out, theta_out, psi_out);
	((F16Plane*)planePtr)->initPlaneInfo();
}

GENERALPLANE_API void setPQR(void* planePtr, double p_out, double q_out, double r_out)
{
	((F16Plane*)planePtr)->setPQR(p_out, q_out, r_out);
	((F16Plane*)planePtr)->initPlaneInfo();
}

GENERALPLANE_API void setPhi(void* planePtr, double phi_out)
{
	((F16Plane*)planePtr)->setPhi(phi_out);
	((F16Plane*)planePtr)->initPlaneInfo();
}

GENERALPLANE_API void setTheta(void* planePtr, double theta_out)
{
	((F16Plane*)planePtr)->setTheta(theta_out);
	((F16Plane*)planePtr)->initPlaneInfo();
}

GENERALPLANE_API void setPsi(void* planePtr, double psi_out)
{
	((F16Plane*)planePtr)->setPsi(psi_out);
	((F16Plane*)planePtr)->initPlaneInfo();
}

GENERALPLANE_API double getMaxThrust(void* planePtr, double v, double h)
{
	return ((F16Plane*)planePtr)->getMaxThrust(h, v);
}

// fmout, RU, Fx, Fy, Fz, Mx, My, Mz, Tx, Ty, Tz
GENERALPLANE_API void getFMDCS(void* planePtr, double* fmout)
{
	((F16Plane*)planePtr)->getForceDCS(fmout);
	((F16Plane*)planePtr)->getMomentDCS(fmout+3);
	((F16Plane*)planePtr)->getThrustDCS(fmout+6);
}

// TAS, alpha, beta, phi, theta, psi, p, q, r, h
GENERALPLANE_API void setStateDCS(void* planePtr, double* stsout)
{
	((F16Plane*)planePtr)->setStateDCS(stsout);
}

GENERALPLANE_API void loadPlaneInfo(void* planePtr, double* stin, double* ctrlin)
{
	PlaneState planeSts;
	Actuator ctrlPos;
	planeSts.vt = stin[0];
    planeSts.alpha = stin[1] / R2D;
    planeSts.beta = stin[2] / R2D;
    planeSts.p = stin[3] / R2D;
    planeSts.q = stin[4] / R2D;
    planeSts.r = stin[5] / R2D;
    planeSts.phi = stin[6] / R2D;
    planeSts.theta = stin[7] / R2D;
    planeSts.psi = stin[8] / R2D;
    planeSts.h = stin[9];
    planeSts.npos = stin[10];
    planeSts.epos = stin[11];
    planeSts.qw = 1;
    planeSts.qx = 0;
    planeSts.qy = 0;
    planeSts.qz = 0;
    
    ctrlPos.ail = ctrlin[0];
    ctrlPos.ele = ctrlin[1];
    ctrlPos.rud = ctrlin[2];
    ctrlPos.thrust = ctrlin[3];
    ctrlPos.lef = 0;
    ctrlPos.jsb = 0;

	((F16Plane*)planePtr)->loadPlaneInfo(&planeSts, &ctrlPos);
	((F16Plane*)planePtr)->setLLA(stin[12], stin[13]);
}

GENERALPLANE_API void getStates(void* planePtr, double* stateOutput)
{
	((F16Plane*)planePtr)->getStatesForGame(stateOutput);
}

GENERALPLANE_API void saveStates(void* planePtr)
{
	((F16Plane*)planePtr)->savePlaneInfoNext();
}

GENERALPLANE_API void loadStates(void* planePtr)
{
	((F16Plane*)planePtr)->loadPlaneInfo();
}

GENERALPLANE_API void setLog(void* planePtr, char* logName)
{
	((F16Plane*)planePtr)->setLog(logName);
}