
#include "f16Plane.h"

F16Plane::F16Plane(double h, double v, double stepTimeSet, int order, char* libName)
{
    // const char* libName = "f16/lib/f16_air.lib";
    f16AeroIns = new F16Aero(libName);

    time = 0;
    h0 = h;
    v0 = v;
    logOpen = false;
	aeroDataDebug = true;
    setLLA(120.0, 30.0);
    
    memset((void*)&planeSts, 0, sizeof(PlaneState));
    memset((void*)&ctrlCmd, 0, sizeof(Actuator));
    memset((void*)&ctrlPos, 0, sizeof(Actuator));
    memset((void*)&planeFM, 0, sizeof(FM));

	// actuator model
    stepTime = stepTimeSet;
    ailLag = new Lag(25, stepTime);
    eleLag = new Lag(25, stepTime);
    rudLag = new Lag(25, stepTime);
	lefLag = new Lag(25, stepTime);
    engineLag = new Lag(1, stepTime);
    
    planeSts.h = h;
    planeSts.vt = v;
    
    trimPlane(h0, v0, false);
    planeSts.alpha = tAlpha;
    planeSts.theta = tAlpha;
    eularToQuater(&planeSts.phi, &planeSts.qw);
    
    ctrlCmd.ele = tEle*R2D;
    ctrlPos.ele = tEle*R2D;
	eleLag->setout(ctrlPos.ele);
    ctrlCmd.thrust = tThrust;
    ctrlPos.thrust = tThrust;
	engineLag->setout(ctrlPos.thrust);

	actuatorModeOff();
    engineModeOff();
    
    // init plane dynamic
    planeDyn = new MyPlane(stepTime, order);
    
    f16AeroIns->getInertia(m, Jx, Jy, Jz, Jxz, faip);
    planeDyn->setPlaneParams(m, Jx, Jy, Jz, Jxz, faip);

    memcpy(&planeSts0, &planeSts, sizeof(PlaneState));
    initPlaneInfo();
	initRange();
    setLog("f16trace.csv");
}

F16Plane::~F16Plane()
{
    if(logOpen)
    {
        fclose(lg);
    }
    // unUseLib();
    delete f16AeroIns;
	delete planeDyn;
	delete ailLag;
	delete eleLag;
	delete rudLag;
	delete lefLag;
	delete engineLag;
}

void F16Plane::initPlaneInfo()
{
    f16AeroIns->calAero(&planeSts, &ctrlPos, &planeFM);
	calPlaneInfo(&pinfoStepStart);
    calPlaneInfo(&pinfoStepEnd);
	savePlaneInfoNext();
}

void F16Plane::setThrust(double tin)
{
	tin = setInRange(tin, getMaxThrust(), 0.0);
	ctrlCmd.thrust = tin;
    ctrlPos.thrust = tin;
	engineLag->setout(ctrlPos.thrust);
}

void F16Plane::initRange()
{
    Emax = 25;
    Emin = -25;
    Amax = 21.5;
    Amin = -21.5;
    Rmax = 30;
    Rmin = -30;
    PLAmax = 1;
    PLAmin = 0;
	pla = 0;
	double twratio = 1.26;
	/*
	FILE* f = fopen("f16/lib/max_thrust_to_weight_ratio.dat", "r");
	if(f == NULL)
	{
		printf("max_thrust_to_weight_ratio.dat open error!\n");
		printf("default max_thrust_to_weight_ratio = %f\n", twratio);
	}
	else
	{
		int err = fscanf(f, "%lf", &twratio);
		// printf("set max thrust to weight ratio to %f\n", twratio);
	}
	if(f != NULL)
	{
		fclose(f);
	}
	*/

	// maxThrust = 42220;		// 2017, first wrong thrust, 19000 / 0.45 = 42220N
    // maxThrust = 70211;		// 2020, increase max thrust according to pilot, 0.77 * 9295.44 * 9.81 = 70215N
	// maxThrust = 84550;		// 2021, according to wrong calculation in 2017, 19000 * 0.4536 * 9.81 = 84546.5N
	maxThrust = twratio * m * g;		// 20210811, change thrust-to-weight ratio to 1.26 according to wikipedia, 1.26 * 9295.44 * 9.81 = 114900N
}

void F16Plane::trimPlane(double h, double v, bool showTrim)
{
    f16AeroIns->trim(h, v, &tAlpha, &tEle, &tThrust);
    tThrOpening = tThrust / getMaxThrust(h, v);
    if(showTrim)
    {
		printf("trim result, h: %f, v: %f, ma: %f, IAS: %f\n", h, v, v / soundSpeed(h), TAS2IAS(v, h));
        printf("alpha(deg): %f, ele(deg): %f, thrust: %f\n", tAlpha*R2D, tEle*R2D, tThrust);
    }
}

void F16Plane::setOrder(double stepTimeSet, int order)
{
    stepTime = stepTimeSet;
    planeDyn = new MyPlane(stepTimeSet, order);
}

double F16Plane::getStepTime()
{
    return stepTime;
}

void F16Plane::setLog(char* logName)
{
    fopen_s(&lg, logName, "w");
    fprintf(lg, "time");
    fprintf(lg, ",npos,epos,h,alpha,beta,phi,theta,psi,p,q,r,TAS");
    fprintf(lg, ",ail,ele,rud,thrust,ail_cmd,ele_cmd,rud_cmd,thrust_cmd");
    fprintf(lg, ",vn,ve,vh,mu,chi,nx,ny,nz,nx_ref,ny_ref,nz_ref");
    fprintf(lg, ",Fx,Fy,Fz,Mx,My,Mz,lon,lat,lef,lef_cmd");
    if(aeroDataDebug)
    {
        fprintf(lg, ",Cz,delta_Cz_lef,Czq,delta_Czq_lef");
        fprintf(lg, ",Cm,delta_Cm_lef,Cmq,delta_Cmq_lef");
        fprintf(lg, ",delta_Cm,delta_Cm_ds");
    }
    for(int i=0; i<10; i++)
    {
        fprintf(lg, ",d%d", i);
    }
    fprintf(lg, "\n");
    logOpen = true;
}

void F16Plane::logOn()
{
    logOpen = true;
}

void F16Plane::logOff()
{
    logOpen = false;
}

void F16Plane::setLog(const char* logName)
{
    setLog((char*)logName);
}

void F16Plane::setCtrl(double ail, double ele, double rud, double thrust)
{
    ctrlCmd.ail = setInRange(ail, Amax, Amin);
    ctrlCmd.ele = setInRange(ele, Emax, Emin);
    ctrlCmd.rud = setInRange(rud, Rmax, Rmin);
    ctrlCmd.thrust = setInRange(thrust, maxThrust, 0.0);
}

void F16Plane::getCtrlCmd(double *acts)
{
    acts[0] = ctrlCmd.ail;
    acts[1] = ctrlCmd.ele;
    acts[2] = ctrlCmd.rud;
    acts[3] = ctrlCmd.thrust;
}

void F16Plane::getCtrlPos(double *acts)
{
    acts[0] = ctrlPos.ail;
    acts[1] = ctrlPos.ele;
    acts[2] = ctrlPos.rud;
    acts[3] = ctrlPos.thrust;
}

void F16Plane::setAttitude(double phiIn, double thetaIn, double psiIn)
{
    planeSts.phi = phiIn / R2D;     // degree to rad
    planeSts.theta = thetaIn / R2D;
    planeSts.psi = psiIn / R2D;
    eularToQuater(&planeSts.phi, &planeSts.qw);
}

void F16Plane::setPhi(double phiIn)
{
    planeSts.phi = phiIn / R2D;
    eularToQuater(&planeSts.phi, &planeSts.qw);
}

void F16Plane::setTheta(double thetaIn)
{
    planeSts.theta = thetaIn / R2D;
    eularToQuater(&planeSts.phi, &planeSts.qw);
}

void F16Plane::setPsi(double psiIn)
{
    planeSts.psi = psiIn / R2D;
    eularToQuater(&planeSts.phi, &planeSts.qw);
}

void F16Plane::setAlpha(double alphaIn)
{
    planeSts.alpha = alphaIn / R2D;
}

void F16Plane::setBeta(double betaIn)
{
    planeSts.beta = betaIn / R2D;
}

void F16Plane::setPQR(double pIn, double qIn, double rIn)
{
    planeSts.p = pIn / R2D;
	planeSts.q = qIn / R2D;
	planeSts.r = rIn / R2D;
}

double F16Plane::getTAS()
{
    return planeSts.vt;
}

void F16Plane::setTAS(double TAS)
{
    planeSts.vt = TAS;
}

void F16Plane::setIAS(double IAS)
{
    planeSts.vt = IAS2TAS(IAS, planeSts.h);
}

void F16Plane::setAltitude(double alt)
{
    planeSts.h = alt;
}

void F16Plane::setLLA(double lon_out, double lat_out)
{
    lon0 = lon_out;
    lat0 = lat_out;
    pinfoStepStart.lon = lon0;
    pinfoStepStart.lat = lat0;
    pinfoStepEnd.lon = lon0;
    pinfoStepEnd.lat = lat0;

    double b = 6356752.3142;
    earth_e2 = 0.00669437999013;
    
    latC = atan( tan(lat0 / R2D) * (1 - earth_e2) );
    earth_R = b / sqrt(1 - earth_e2 * cos(latC) * cos(latC / R2D)) + planeSts.h;
	pinfoStepStart.latC = latC;
	pinfoStepStart.earth_R = earth_R;
	pinfoStepEnd.latC = latC;
	pinfoStepEnd.earth_R = earth_R;
}

void F16Plane::setPos(double npos_out, double epos_out)
{
    pinfoStepStart.npos = npos_out;
    pinfoStepStart.epos = epos_out;
    pinfoStepEnd.npos = npos_out;
    pinfoStepEnd.epos = epos_out;
}

void F16Plane::setTime(double timeIn)
{
    time = timeIn;
}

void F16Plane::calLLA()
{
    earth_R += pinfoStepStart.vh * stepTime;
    latC = latC + pinfoStepStart.vn / earth_R * stepTime;
    pinfoStepEnd.lon += pinfoStepStart.ve / (earth_R * cos(latC)) * R2D * stepTime;
    pinfoStepEnd.lat = atan(tan(latC) / (1-earth_e2)) * R2D;
	pinfoStepStart.latC = latC;
	pinfoStepStart.earth_R = earth_R;
	pinfoStepEnd.latC = latC;
	pinfoStepEnd.earth_R = earth_R;
}

void F16Plane::getLLA(double &lon_out, double &lat_out)
{
    lon_out = pinfoStepEnd.lon;
    lat_out = pinfoStepEnd.lat;
}

void F16Plane::step()
{
    actuatorStep();
    engineStep();
    f16AeroIns->calAeroWithDamage(&planeSts, &ctrlPos, &planeFM);
	// f16AeroIns->calAero(&planeSts, &ctrlPos, &planeFM);
    calPlaneInfo(&pinfoStepStart);
    if(logOpen)
    {
        writeSts();
    }
    planeDyn->runQuan(&planeFM, &planeSts);
    calLLA();
    time = time + stepTime;
    calPlaneInfo(&pinfoStepEnd);
}

void F16Plane::step(double ail, double ele, double rud, double thrust)
{
    setCtrl(ail, ele, rud, thrust);
    step();
}

void F16Plane::step_frac(double ail, double ele, double rud, double thrOpening)
{
    thrOpening = setInRange(thrOpening, 1.0, 0.0);
	pla = thrOpening;
    step(ail, ele, rud, thrOpening*getMaxThrust());
}

void F16Plane::actuatorStep()
{
	lefStep();
    if(actuatorModeFlag)
    {
        ctrlPos.ail = (*ailLag)(ctrlCmd.ail);
        ctrlPos.ele = (*eleLag)(ctrlCmd.ele);
        ctrlPos.rud = (*rudLag)(ctrlCmd.rud);
		ctrlPos.lef = (*lefLag)(ctrlCmd.lef);
    }
    else
    {
        ctrlPos.ail = ctrlCmd.ail;
        ctrlPos.ele = ctrlCmd.ele;
        ctrlPos.rud = ctrlCmd.rud;
		ctrlPos.lef = ctrlCmd.lef;
    }
}

// add flap auto work function 20210221
void F16Plane::lefStep()
{
	double alpha_deg = planeSts.alpha * R2D;
	if (alpha_deg > 10)
	{
		ctrlCmd.lef = (alpha_deg - 10) * 2.5;
	}
	else
	{
		ctrlCmd.lef = 0;
	}
	ctrlCmd.lef = setInRange(ctrlCmd.lef, 0.0, 25.0);
}

void F16Plane::engineStep()
{
	ctrlCmd.thrust = setInRange(ctrlCmd.thrust, getMaxThrust(), 0.0);
    if(engineModeFlag)
    {
        ctrlPos.thrust = (*engineLag)(ctrlCmd.thrust);
    }
    else
    {
        ctrlPos.thrust = ctrlCmd.thrust;
    }
}

void F16Plane::writeSts()
{
    fprintf(lg, "%f", time);
    fprintf(lg, ",%f,%f,%f", pinfoStepStart.npos, pinfoStepStart.epos, pinfoStepStart.h);
    fprintf(lg, ",%f,%f", pinfoStepStart.alpha, pinfoStepStart.beta);
    fprintf(lg, ",%f,%f,%f", pinfoStepStart.phi, pinfoStepStart.theta, pinfoStepStart.psi);
    fprintf(lg, ",%f,%f,%f", pinfoStepStart.p, pinfoStepStart.q, pinfoStepStart.r);
    fprintf(lg, ",%f", pinfoStepStart.vt);
    fprintf(lg, ",%f,%f,%f,%f", ctrlPos.ail, ctrlPos.ele, ctrlPos.rud, ctrlPos.thrust);
    fprintf(lg, ",%f,%f,%f,%f", ctrlCmd.ail, ctrlCmd.ele, ctrlCmd.rud, ctrlCmd.thrust);
    fprintf(lg, ",%f,%f,%f", pinfoStepStart.vn, pinfoStepStart.ve, pinfoStepStart.vh);
    fprintf(lg, ",%f,%f", pinfoStepStart.mu, pinfoStepStart.chi);
    fprintf(lg, ",%f,%f,%f", pinfoStepStart.nx, pinfoStepStart.ny, pinfoStepStart.nz);
    fprintf(lg, ",%f,%f,%f", pinfoStepStart.nx_ref, pinfoStepStart.ny_ref, pinfoStepStart.nz_ref);
    fprintf(lg, ",%f,%f,%f", planeFM.Fx, planeFM.Fy, planeFM.Fz);
    fprintf(lg, ",%f,%f,%f", planeFM.Mx, planeFM.My, planeFM.Mz);
    fprintf(lg, ",%f,%f", pinfoStepStart.lon, pinfoStepStart.lat);
	fprintf(lg, ",%f,%f", ctrlPos.lef, ctrlCmd.lef);
	if(aeroDataDebug)
	{
		fprintf(lg, ",%f,%f,%f,%f", f16AeroIns->Cz, f16AeroIns->delta_Cz_lef, f16AeroIns->Czq, f16AeroIns->delta_Czq_lef);
		fprintf(lg, ",%f,%f,%f,%f", f16AeroIns->Cm, f16AeroIns->delta_Cm_lef, f16AeroIns->Cmq, f16AeroIns->delta_Cmq_lef);
		fprintf(lg, ",%f,%f", f16AeroIns->delta_Cm, f16AeroIns->delta_Cm_ds);
	}
	for(int i=0; i<10; i++)
	{
		fprintf(lg, ",%.2f", f16AeroIns->damage[i]);
	}
	fprintf(lg, "\n");
}

void F16Plane::showSelf()
{
    printf("%f", time);
    printf(",%f,%f,%f", pinfoStepStart.npos, pinfoStepStart.epos, pinfoStepStart.h);
    printf(",%f,%f", pinfoStepStart.alpha, pinfoStepStart.beta);
    printf(",%f,%f,%f", pinfoStepStart.phi, pinfoStepStart.theta, pinfoStepStart.psi);
    printf(",%f,%f,%f", pinfoStepStart.p, pinfoStepStart.q, pinfoStepStart.r);
    printf(",%f", pinfoStepStart.vt);
    printf(",%f,%f,%f,%f", ctrlPos.ail, ctrlPos.ele, ctrlPos.rud, ctrlPos.thrust);
    printf(",%f,%f,%f,%f", ctrlCmd.ail, ctrlCmd.ele, ctrlCmd.rud, ctrlCmd.thrust);
    printf(",%f,%f,%f", pinfoStepStart.vn, pinfoStepStart.ve, pinfoStepStart.vh);
    printf(",%f,%f", pinfoStepStart.mu, pinfoStepStart.chi);
    printf(",%f,%f,%f", pinfoStepStart.nx, pinfoStepStart.ny, pinfoStepStart.nz);
    printf(",%f,%f,%f", pinfoStepStart.nx_ref, pinfoStepStart.ny_ref, pinfoStepStart.nz_ref);
    printf(",%f,%f,%f", planeFM.Fx, planeFM.Fy, planeFM.Fz);
    printf(",%f,%f,%f", planeFM.Mx, planeFM.My, planeFM.Mz);
    printf(",%f,%f", pinfoStepStart.lon, pinfoStepStart.lat);
	printf(",%f,%f\n", ctrlPos.lef, ctrlCmd.lef);
}

void F16Plane::showInfo(PlaneStateInfo* pinfout)
{
    int infosize = sizeof(PlaneStateInfo) / sizeof(double);
    double *info = (double*)pinfout;
    printf("%f", info[0]);
    for(int i=0; i<infosize; i++)
    {
        printf(", %f", info[i]);
    }
    printf("\n");
}

// DCS uses Russian Coordinate system
void F16Plane::getForceDCS(double* force)
{
    force[0] = planeFM.Fx;
    force[1] = -planeFM.Fz;
    force[2] = planeFM.Fy;
}

void F16Plane::getMomentDCS(double* moment)
{
    moment[0] = planeFM.Mx;
    moment[1] = -planeFM.Mz;
    moment[2] = planeFM.My;
}

void F16Plane::getThrustDCS(double* thr)
{
    thr[0] = planeFM.T * cos(planeFM.faip);
    thr[1] = planeFM.T * sin(planeFM.faip);
    thr[2] = 0;
}

void F16Plane::setStateDCS(double* stsout)
{
    planeSts.vt = stsout[0];
    planeSts.alpha = stsout[1];
    planeSts.beta = stsout[2];
    planeSts.phi = stsout[3];
    planeSts.theta = stsout[4];
    planeSts.psi = stsout[5];
    planeSts.p = stsout[6];
    planeSts.q = stsout[7];
    planeSts.r = stsout[8];
    planeSts.h = stsout[9];
    eularToQuater(&planeSts.phi, &planeSts.qw);
	initPlaneInfo();
}

void F16Plane::ailTo0()
{
    ctrlCmd.ail = 0;
}

void F16Plane::eleTo0()
{
    ctrlCmd.ele = tEle * R2D;
}

void F16Plane::stickTo0()
{
    ailTo0();
    eleTo0();
}

void F16Plane::PLAto0()
{
    ctrlCmd.thrust = tThrust;
}

void F16Plane::allSetTo0()
{
    stickTo0();
    PLAto0();
}

void F16Plane::reset()
{
    memcpy(&planeSts, &planeSts0, sizeof(PlaneState));
}

void F16Plane::getTrimValue(double* tv, bool showTrim)
{
    tv[0] = h0; // m
    tv[1] = v0; // m/s
    tv[2] = tAlpha * R2D; // theta, alf, deg
    tv[3] = tEle * R2D;   // ele, deg
    tv[4] = tThrust;      // thrust, N
    if(showTrim)
    {
        printf("h(m): %.1f, v(m/s): %.1f, alpha(deg): %.4f, ele(deg): %.4f, thrust(N): %.1f\n", tv[0], tv[1], tv[2], tv[3], tv[4]);
    }
}

void F16Plane::getTrimValue(double h, double v, double* tv, bool showTrim)
{
    tv[0] = h; // m
    tv[1] = v; // m/s
    f16AeroIns->trim(h, v, tv+2, tv+3, tv+4);
    if(showTrim)
    {
        printf("h(m): %.1f, v(m/s): %.1f, alpha(deg): %.4f, ele(deg): %.4f, thrust(N): %.1f\n", tv[0], tv[1], tv[2], tv[3], tv[4]);
    }
}

double F16Plane::getMaxThrust(double h, double v)
{
    double ma = v / soundSpeed(h);
    double prop = 0.5;
    double fracH = prop + (10000 - h) / 10000 * (1 - prop);
    if (fracH < prop) fracH = prop;
    else if(fracH > 1) fracH = 1;

    double currMaxT = f16AeroIns->f16data->_power(ma) * fracH * maxThrust;
    return currMaxT;
}

double F16Plane::getMaxThrust()
{
    return getMaxThrust(planeSts.h, planeSts.vt);
}

void F16Plane::calPlaneInfo(PlaneStateInfo* pinfout)
{
    pinfout->vt = planeSts.vt;
    pinfout->alpha = planeSts.alpha * R2D;
    pinfout->beta = planeSts.beta * R2D;
    pinfout->p = planeSts.p * R2D;
    pinfout->q = planeSts.q * R2D;
    pinfout->r = planeSts.r * R2D;
    pinfout->phi = planeSts.phi * R2D;
    pinfout->theta = planeSts.theta * R2D;
    pinfout->psi = planeSts.psi * R2D;
    pinfout->h = planeSts.h;
    pinfout->npos = planeSts.npos;
    pinfout->epos = planeSts.epos;
    pinfout->qw = planeSts.qw;
    pinfout->qx = planeSts.qx;
    pinfout->qy = planeSts.qy;
    pinfout->qz = planeSts.qz;

    double q0 = planeSts.qw;
    double q1 = planeSts.qx;
    double q2 = planeSts.qy;
    double q3 = planeSts.qz;
    double sa = sin(planeSts.alpha);
    double ca = cos(planeSts.alpha);
    double sb = sin(planeSts.beta);
    double cb = cos(planeSts.beta);
    double vt, vel_u, vel_v, vel_w;
    vt = planeSts.vt;
    vel_u = vt*ca*cb;
    vel_v = vt*sb;
    vel_w = vt*sa*cb;
    pinfout->vn = (q0*q0 + q1*q1 - q2*q2 - q3*q3) * vel_u + 2*(q1*q2 - q0*q3) * vel_v + 2*(q1*q3 + q0*q2) * vel_w;
    pinfout->ve = 2*(q1*q2 + q0*q3) * vel_u + (q0*q0 - q1*q1 + q2*q2 - q3*q3) * vel_v + 2*(q2*q3 - q0*q1) * vel_w;
    pinfout->vh = (-1) * (2*(q1*q3 - q0*q2) * vel_u + 2*(q2*q3 + q0*q1) * vel_v + (q0*q0 - q1*q1 - q2*q2 + q3*q3) * vel_w);
    pinfout->mu = muCal(pinfoStepStart.vh, vt);
    pinfout->chi = atan2(pinfoStepStart.ve, pinfoStepStart.vn) * R2D;

    pinfout->ail = ctrlPos.ail;
    pinfout->ele = ctrlPos.ele;
    pinfout->rud = ctrlPos.rud;
    pinfout->thrust = ctrlPos.thrust;
    pinfout->lef = ctrlPos.lef;
    pinfout->jsb = ctrlPos.jsb;

    pinfout->nx = (planeFM.Fx  + ctrlPos.thrust * cos(planeFM.faip)) / (m*g);
    pinfout->ny = planeFM.Fy / (m*g);
    pinfout->nz = (planeFM.Fz  - ctrlPos.thrust * sin(planeFM.faip)) / (m*g);

    pinfout->nx_ref = pinfout->nx - sin(planeSts.theta);
    pinfout->ny_ref = pinfout->ny + cos(planeSts.theta)*sin(planeSts.phi);
    pinfout->nz_ref = pinfout->nz + cos(planeSts.theta)*cos(planeSts.phi);

    pinfout->lon = pinfoStepEnd.lon;
    pinfout->lat = pinfoStepEnd.lat;
}

void F16Plane::getPinfoStart(PlaneStateInfo *pinfo_out)
{
    memcpy(pinfo_out, &pinfoStepStart, sizeof(PlaneStateInfo));
}

void F16Plane::getPinfoEnd(PlaneStateInfo *pinfo_out)
{
    memcpy(pinfo_out, &pinfoStepEnd, sizeof(PlaneStateInfo));
}

void F16Plane::savePlaneInfoNext()
{
    memcpy(&pinfoStepEndSave, &pinfoStepEnd, sizeof(PlaneStateInfo));
}

void F16Plane::getPlaneInfoNextSave(PlaneStateInfo *pinfo_out)
{
    memcpy(pinfo_out, &pinfoStepEndSave, sizeof(PlaneStateInfo));
}

void F16Plane::loadPlaneInfo()
{
    planeSts.vt = pinfoStepEndSave.vt;
    planeSts.alpha = pinfoStepEndSave.alpha / R2D;
    planeSts.beta = pinfoStepEndSave.beta / R2D;
    planeSts.p = pinfoStepEndSave.p / R2D;
    planeSts.q = pinfoStepEndSave.q / R2D;
    planeSts.r = pinfoStepEndSave.r / R2D;
    planeSts.phi = pinfoStepEndSave.phi / R2D;
    planeSts.theta = pinfoStepEndSave.theta / R2D;
    planeSts.psi = pinfoStepEndSave.psi / R2D;
    planeSts.h = pinfoStepEndSave.h;
    planeSts.npos = pinfoStepEndSave.npos;
    planeSts.epos = pinfoStepEndSave.epos;
    planeSts.qw = pinfoStepEndSave.qw;
    planeSts.qx = pinfoStepEndSave.qx;
    planeSts.qy = pinfoStepEndSave.qy;
    planeSts.qz = pinfoStepEndSave.qz;
    
    ctrlPos.ail = pinfoStepEndSave.ail;
    ctrlPos.ele = pinfoStepEndSave.ele;
    ctrlPos.rud = pinfoStepEndSave.rud;
    ctrlPos.thrust = pinfoStepEndSave.thrust;
    ctrlPos.lef = pinfoStepEndSave.lef;
    ctrlPos.jsb = pinfoStepEndSave.jsb;
    memcpy(&ctrlCmd, &ctrlPos, sizeof(Actuator));
	latC = pinfoStepEndSave.latC;
	earth_R = pinfoStepEndSave.earth_R;
	loadPosToLag();
	memcpy(&pinfoStepStart, &pinfoStepEndSave, sizeof(PlaneStateInfo));
	memcpy(&pinfoStepEnd, &pinfoStepEndSave, sizeof(PlaneStateInfo));
}

void F16Plane::loadPosToLag()
{
	ailLag->setout(ctrlPos.ail);
	eleLag->setout(ctrlPos.ele);
	rudLag->setout(ctrlPos.rud);
	lefLag->setout(ctrlPos.lef);
	engineLag->setout(ctrlPos.thrust);
}

void F16Plane::loadPlaneInfo(PlaneState* stin, Actuator* ctrlin)
{
    memcpy(&planeSts, stin, sizeof(PlaneState));
	eularToQuater(&planeSts.phi, &planeSts.qw);
    memcpy(&ctrlPos, ctrlin, sizeof(Actuator));
    memcpy(&ctrlCmd, ctrlin, sizeof(Actuator));
    f16AeroIns->calAero(&planeSts, &ctrlPos, &planeFM);
	loadPosToLag();
	initPlaneInfo();
}

void F16Plane::getPlaneState(PlaneState* stin)
{
    memcpy(stin, &planeSts, sizeof(PlaneState));
}

void F16Plane::getActState(Actuator* ctrlin)
{
    memcpy(ctrlin, &ctrlPos, sizeof(Actuator));
}

void F16Plane::getStatesForGame(double* stateOutput)
{
	stateOutput[0] = pinfoStepEnd.lef;
	stateOutput[1] = pinfoStepEnd.npos;
    stateOutput[2] = pinfoStepEnd.epos;
    stateOutput[3] = pinfoStepEnd.h;
    stateOutput[4] = pinfoStepEnd.alpha;
    stateOutput[5] = pinfoStepEnd.beta;
    stateOutput[6] = pinfoStepEnd.phi;
    stateOutput[7] = pinfoStepEnd.theta;
    stateOutput[8] = pinfoStepEnd.psi;
    stateOutput[9] = pinfoStepEnd.p;
    stateOutput[10] = pinfoStepEnd.q;
    stateOutput[11] = pinfoStepEnd.r;
    stateOutput[12] = pinfoStepEnd.vt;
    stateOutput[13] = pinfoStepEnd.vn;
    stateOutput[14] = pinfoStepEnd.ve;
    stateOutput[15] = pinfoStepEnd.vh;
    stateOutput[16] = pinfoStepEnd.nx;
    stateOutput[17] = pinfoStepEnd.ny;
    stateOutput[18] = -pinfoStepEnd.nz;
    stateOutput[19] = pinfoStepEnd.ele;
    stateOutput[20] = pinfoStepEnd.ail;
    stateOutput[21] = pinfoStepEnd.rud;
	stateOutput[22] = pinfoStepEnd.thrust;
	stateOutput[23] = ctrlCmd.ele;
    stateOutput[24] = ctrlCmd.ail;
    stateOutput[25] = ctrlCmd.rud;
    stateOutput[26] = pla;
    stateOutput[27] = pinfoStepEnd.lon;
    stateOutput[28] = pinfoStepEnd.lat;
    stateOutput[29] = pinfoStepEnd.mu;
    stateOutput[30] = pinfoStepEnd.chi;
}

void F16Plane::actuatorModeOn(){actuatorModeFlag = true;}
void F16Plane::actuatorModeOff(){actuatorModeFlag = false;}
bool F16Plane::isActuatorModeOn(){return actuatorModeFlag;}
void F16Plane::engineModeOn(){engineModeFlag = true;}
void F16Plane::engineModeOff(){engineModeFlag = false;}
bool F16Plane::isEngineModeOn(){return engineModeFlag;}

void F16Plane::setDamage(double dmg[])
{
    memcpy(f16AeroIns->damage, dmg, 10*sizeof(double));
}
