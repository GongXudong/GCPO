#ifndef _F16PLANE_H
#define _F16PLANE_H

#include "f16AeroCal.h"
#include "../planeDyn/planeDynUS.h"
#include "../../tools/utils/control.h"

class F16Plane
{
public:
    F16Plane(double h, double v, double stepTimeSet, int order, char* libName);
    ~F16Plane();
	Actuator ctrlCmd;     // struct plane control cmd: ail, ele, rud, thrust
    Actuator ctrlPos;     // struct plane control pos: ail, ele, rud, thrust

public:
    F16Aero* f16AeroIns;
    double h0;
    double v0;
    double tAlpha;
    double tEle;
    double tThrust;
	double tThrOpening;
    double time;
    double stepTime;

	double m, Jx, Jy, Jz, Jxz, faip;
    
    FILE *lg;
    bool logOpen;
    bool actuatorModeFlag;
    bool engineModeFlag;
	bool aeroDataDebug;
    
    PlaneState planeSts;     // struct plane state
	PlaneState planeSts0;     // struct plane state

    FM planeFM;       // struct force and moment
	PlaneStateInfo pinfoStepStart;
	PlaneStateInfo pinfoStepEnd;
	PlaneStateInfo pinfoStepEndSave;
    
    MyPlane *planeDyn;
    
    Lag *ailLag;
    Lag *eleLag;
    Lag *rudLag;
	Lag *lefLag;
    Lag *engineLag;

	double Emax;
	double Emin;
	double Amax;
	double Amin;
	double Rmax;
	double Rmin;
	double PLAmax;
	double PLAmin;
	double maxThrust;
	double pla;

	double lon0;     // rad
	double lat0;     // rad
	double earth_R; // m
	double latC;    // rad
	double earth_e2;
    
public:
    double getStepTime();
	void initRange();
    void setOrder(double stepTimeSet, int order);
    void setLog(char* logName);
    void setLog(const char* logName);
    void logOn();
    void logOff();
    void writeSts();
    
    void setAttitude(double phiIn, double thetaIn, double psiIn);
    void setPhi(double phiIn);
	void setTheta(double thetaIn);
    void setPsi(double psiIn);
    void setAlpha(double alphaIn);
	void setBeta(double betaIn);
	void setPQR(double pIn, double qIn, double rIn);
    void setTime(double timeIn);

    double getTAS();
    void setTAS(double v);
    void setIAS(double IAS);
    void setAltitude(double alt);
	void setPos(double npos_out, double epos_out);
    void setLLA(double lon_out, double lat_out);
	void calLLA();
    void getLLA(double &lon_out, double &lat_out);
    
    void setCtrl(double ail, double ele, double rud, double thrust);
    void getCtrlCmd(double *acts);
    void getCtrlPos(double *acts);
    
	void step();
	void step(double ail, double ele, double rud, double thrust);
	void step_frac(double ail, double ele, double rud, double thrOpening);
    void actuatorStep();
    void engineStep();
    void lefStep();

    // Set the pitch and yaw angle to trimmed values
    void ailTo0();
    void eleTo0();
    void stickTo0();
    void PLAto0();
    void allSetTo0();
	void reset();
    
    void trimPlane(double h, double v, bool showTrim=false);
    void getTrimValue(double* tv, bool showTrim=false);
    void getTrimValue(double h, double v, double* tv, bool showTrim=false);
	double getMaxThrust(double h, double v);
	double getMaxThrust();
	void setThrust(double tin);
	void initPlaneInfo();
	void calPlaneInfo(PlaneStateInfo* pinfout);
	void getPinfoStart(PlaneStateInfo *pinfo_out);
	void getPinfoEnd(PlaneStateInfo *pinfo_out);
    void savePlaneInfoNext();
	void getPlaneInfoNextSave(PlaneStateInfo *pinfo_out);
    void loadPlaneInfo();
	void loadPosToLag();
    void loadPlaneInfo(PlaneState* stin, Actuator* ctrlin);
    void getPlaneState(PlaneState* stin);
    void getActState(Actuator* ctrlin);
	void showSelf();
	void showInfo(PlaneStateInfo* pinfout);
	void getForceDCS(double* force);
	void getMomentDCS(double* moment);
	void getThrustDCS(double* thr);
	void setStateDCS(double* stsout);
	void getStatesForGame(double* stateOutput);
    
    void actuatorModeOn();
    void actuatorModeOff();
    bool isActuatorModeOn();
    void engineModeOn();
    void engineModeOff();
    bool isEngineModeOn();

    void setDamage(double dmg[]); //NY modified 20220526
};

#endif