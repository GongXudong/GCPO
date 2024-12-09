#include "planeDynUS.h"

MyPlane::MyPlane(double step, int order)
{
    rk1 = new PlaneState;
    rk2 = new PlaneState;
    rk3 = new PlaneState;
    rk4 = new PlaneState;
	tmp = new PlaneState;
    lastout = new PlaneState;
    set(step, order);
	MaxState = 16;
}

MyPlane::~MyPlane(){
    delete rk1;
    delete rk2;
    delete rk3;
    delete rk4;
	delete tmp;
    delete lastout;
}

void MyPlane::set(double step, int order)
{
    Order = order;
    IStep = step;
}

void MyPlane::setPlaneParams(double mass, double Ix_in, double Iy_in, double Iz_in, double Ixz_in, double faip_in)
{
	double Ixx, Iyy, Izz, Ixz;
	m = mass;
	Ixx = Ix_in;
	Iyy = Iy_in;
	Izz = Iz_in;
	Ixz = Ixz_in;

	double Gamma = Ixx * Izz - (Ixz  * Ixz);
    C1 = ((Iyy - Izz)  * Izz  - (Ixz * Ixz))/ Gamma;
    C2 = ((Ixx - Iyy + Izz ) * Ixz ) / Gamma;
    C3 = Izz / Gamma;
    C4 = Ixz / Gamma;
    C5 = (Izz - Ixx) / Iyy;
    C6 = Ixz / Iyy ;
    C7 = 1 / Iyy;
    C8 = (Ixx * (Ixx - Iyy ) + Ixz * Ixz) / Gamma;
    C9 = Ixx / Gamma;

	faip = faip_in;
}

void MyPlane::operator()(FM *fmPtr, PlaneState * stPtr)
{
    int i;

    double *rk1Ptr, *rk2Ptr, *rk3Ptr, *rk4Ptr, *tempPtr; /* 指向RungKutta法的指针对*/
    
    rk1Ptr = (double*)rk1;
    rk2Ptr = (double*)rk2;
    rk3Ptr = (double*)rk3;               /*  指针赋值 */
    rk4Ptr = (double*)rk4;
    tempPtr = (double*)tmp;
    
    
	double *planeOutStructPtr;    /*pointer of PlaneOut struct*/
	planeOutStructPtr = &(stPtr->vt);
	PlaneState stateX;
	double *stateXPtr;
	stateXPtr = &(stateX.vt);
	for(i=0;i<MaxState;i++)
    {
        stateXPtr[i] = planeOutStructPtr[i];
    }

	PlaneState out;
	out = *stPtr;
	double *outputPtr;
	outputPtr = &(stPtr->vt);
    
	if(Order == 4)
    {                         /* 四阶龙格-库塔法 (缺省)*/
        pf(fmPtr, &stateX, rk1);      /*   rk1  */
        for(i=0;i<MaxState;i++)
        {
            tempPtr[i] = stateXPtr[i] + rk1Ptr[i]/2.0 * IStep;
        }

        pf(fmPtr, tmp, rk2);        /*   rk2   */
        for(i=0;i<MaxState;i++)
        {
            tempPtr[i]=stateXPtr[i]+rk2Ptr[i]/2.0*IStep;
        }

        pf(fmPtr, tmp, rk3);        /*   rk3   */
        for(i=0;i<MaxState;i++)
        {
            tempPtr[i]=stateXPtr[i]+rk3Ptr[i]*IStep;
        }

        pf(fmPtr, tmp, rk4);        /*   rk4   */
        for(i=0;i<MaxState;i++)
        {
            outputPtr[i]=stateXPtr[i] + 1/6.0*(rk1Ptr[i] + rk2Ptr[i]*2.0 + rk3Ptr[i]*2.0 + rk4Ptr[i]) * IStep;
        }
    }
    else if(Order == 2)
    {                                /* 二阶龙格-库塔 */
        pf(fmPtr, &stateX, rk1);/*   rk1  */
        for(i=0;i<MaxState;i++)
        {
            tempPtr[i] = stateXPtr[i] + rk1Ptr[i] * IStep;
        }
        pf(fmPtr, tmp, rk2);  /*   rk2   */
        for(i=0;i<MaxState;i++)
        {
            outputPtr[i]=stateXPtr[i] + 1/2.0*(rk1Ptr[i] + rk2Ptr[i]) * IStep;
        }
    }
	else if(Order == 1)
	{
		pf(fmPtr, &stateX, rk1);
        for(i=0;i<MaxState;i++)
        {
            outputPtr[i] = stateXPtr[i] + rk1Ptr[i] * IStep;
        }
	}
    
	pout(stPtr, rk1);
}

void MyPlane::runQuan(FM *fmPtr, PlaneState * stPtr)
{
    int i;
    double *rk1Ptr, *rk2Ptr, *rk3Ptr, *rk4Ptr, *tempPtr; /* 指向RungKutta法的指针对*/
    
    rk1Ptr = (double*)rk1;
    rk2Ptr = (double*)rk2;
    rk3Ptr = (double*)rk3;               /*  指针赋值 */
    rk4Ptr = (double*)rk4;
    tempPtr = (double*)tmp;
    
	double *planeOutStructPtr;    /*pointer of PlaneOut struct*/
	planeOutStructPtr = &(stPtr->vt);
	PlaneState stateX;
	double *stateXPtr;
	stateXPtr = &(stateX.vt);
	for(i=0;i<MaxState;i++)
    {
        stateXPtr[i] = planeOutStructPtr[i];
    }

	PlaneState out;
	out = *stPtr;
	double *outputPtr;
	outputPtr = &(stPtr->vt);
    
	pf_quaternion(fmPtr, &stateX, rk1);
    for(i=0;i<MaxState;i++)
    {
        outputPtr[i] = stateXPtr[i] + rk1Ptr[i] * IStep;
    }
    
	poutQuan(stPtr, rk1);
}

void MyPlane::pf(FM *fmPtr, PlaneState *inPtr, PlaneState *dPtr)
{
	double T = fmPtr->T;		// thrust
	double spa = sin(faip);
    double cpa = cos(faip);

	double Fx = fmPtr->Fx + T * cpa;
	double Fy = fmPtr->Fy;
	double Fz = fmPtr->Fz - T * spa;

    double P  = inPtr->p;
	double Q  = inPtr->q;
    double R  = inPtr->r;

    double sa = sin(inPtr->alpha);
    double ca = cos(inPtr->alpha);
    double sb = sin(inPtr->beta);
    double cb = cos(inPtr->beta);
    double tb = sb / cb;
    double st = sin(inPtr->theta);
    double ct = cos(inPtr->theta);
	double tt = st / ct;
    double sphi = sin(inPtr->phi);
    double cphi = cos(inPtr->phi);
	double spsi = sin(inPtr->psi);
    double cpsi = cos(inPtr->psi);

	double Vt = inPtr->vt;
	double vel_u, vel_v, vel_w;
	vel_u = Vt*ca*cb;
    vel_v = Vt*sb;
    vel_w = Vt*sa*cb;
	
    double nposdot = vel_u*(ct*cpsi) + vel_v*(sphi*cpsi*st - cphi*spsi) + vel_w*(cphi*st*cpsi + sphi*spsi);
    double eposdot = vel_u*(ct*spsi) + vel_v*(sphi*spsi*st + cphi*cpsi) + vel_w*(cphi*st*spsi - sphi*cpsi);
    double altdot = vel_u*st - vel_v*(sphi*ct) - vel_w*(cphi*ct);
	
    // compute Udot,Vdot, Wdot,(as on NASA report p36)
    double Udot = R*vel_v - Q*vel_w - g*st + Fx/m;
    double Vdot = P*vel_w - R*vel_u + g*ct*sphi + Fy/m;
    double Wdot = Q*vel_u - P*vel_v + g*ct*cphi + Fz/m;
	
    // vt_dot equation (from S&L, p82), alpha_dot, beta_dot
    double vtdot = (vel_u*Udot + vel_v*Vdot + vel_w*Wdot)/Vt;
    double alphadot = (vel_u*Wdot - vel_w*Udot)/(vel_u*vel_u + vel_w*vel_w);
    double betadot = (Vdot*Vt - vel_v*vtdot)/(Vt*Vt*cb);

	dPtr->vt = vtdot;
    dPtr->alpha = alphadot;
    dPtr->beta = betadot;

	// compute Pdot, Qdot, and Rdot (as in Stevens and Lewis p32)
    double Mx = fmPtr->Mx;
    double My = fmPtr->My;
	double Mz = fmPtr->Mz;

    double Pdot = C1*Q*R + C2*P*Q + C3*Mx + C4*Mz;
    double Qdot = C5*P*R - C6*(P*P-R*R) + C7*My;
    double Rdot = C8*P*Q - C2*Q*R + C4*Mx + C9*Mz;

	// Kinematic equations
    double phidot = P + tt*(Q*sphi + R*cphi);
    double thetadot = Q*cphi - R*sphi;
    double psidot = (Q*sphi + R*cphi)/ct;

    dPtr->p = Pdot;
    dPtr->q = Qdot;
    dPtr->r = Rdot;

	dPtr->phi = phidot;
    dPtr->theta = thetadot;
    dPtr->psi = psidot;

	dPtr->qw = 0;
    dPtr->qx = 0;
    dPtr->qy = 0;
	dPtr->qz = 0;

	dPtr->h = altdot;
	dPtr->npos = nposdot;
	dPtr->epos = eposdot;
}

void MyPlane::pf_quaternion(FM *fmPtr, PlaneState *inPtr, PlaneState *dPtr)
{
	double T = fmPtr->T;		// thrust
	double spa = sin(faip);
    double cpa = cos(faip);

	double Fx = fmPtr->Fx + T * cpa;
	double Fy = fmPtr->Fy;
	double Fz = fmPtr->Fz - T * spa;

    double P  = inPtr->p;
	double Q  = inPtr->q;
    double R  = inPtr->r;

    double sa = sin(inPtr->alpha);
    double ca = cos(inPtr->alpha);
    double sb = sin(inPtr->beta);
    double cb = cos(inPtr->beta);
    double tb = sb / cb;

	double q0 = inPtr->qw;
	double q1 = inPtr->qx;
	double q2 = inPtr->qy;
	double q3 = inPtr->qz;

	double Vt = inPtr->vt;
	double vel_u, vel_v, vel_w;
	vel_u = Vt*ca*cb;
    vel_v = Vt*sb;
    vel_w = Vt*sa*cb;
	
    // compute Udot,Vdot, Wdot,(as on NASA report p36)
    double Udot = R*vel_v - Q*vel_w + 2*(q1*q3 - q0*q2)*g + Fx/m;
    double Vdot = P*vel_w - R*vel_u + 2*(q2*q3 + q0*q1)*g + Fy/m;
    double Wdot = Q*vel_u - P*vel_v + (q0*q0 - q1*q1 - q2*q2 + q3*q3)*g + Fz/m;
	
    // vt_dot equation (from S&L, p82), alpha_dot, beta_dot
    double vtdot = (vel_u*Udot + vel_v*Vdot + vel_w*Wdot)/Vt;
    double alphadot = (vel_u*Wdot - vel_w*Udot)/(vel_u*vel_u + vel_w*vel_w);
    double betadot = (Vdot*Vt - vel_v*vtdot)/(Vt*Vt*cb);

	dPtr->vt = vtdot;
    dPtr->alpha = alphadot;
    dPtr->beta = betadot;

	double q0_dot = 0.5 * (-P * q1 - Q * q2 - R * q3);
    double q1_dot = 0.5 * ( P * q0 + R * q2 - Q * q3);
    double q2_dot = 0.5 * ( Q * q0 - R * q1 + P * q3);
    double q3_dot = 0.5 * ( R * q0 + Q * q1 - P * q2);

	/* correction term*/
	double dq = 2 * (1 - q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3);
    q0_dot = dq * q0 + q0_dot;
    q1_dot = dq * q1 + q1_dot;
    q2_dot = dq * q2 + q2_dot;
    q3_dot = dq * q3 + q3_dot;

	dPtr->qw = q0_dot;
    dPtr->qx = q1_dot;
    dPtr->qy = q2_dot;
	dPtr->qz = q3_dot;

	dPtr->phi = 0;
    dPtr->theta = 0;
    dPtr->psi = 0;

	// compute Pdot, Qdot, and Rdot (as in Stevens and Lewis p32)
    double Mx = fmPtr->Mx;
    double My = fmPtr->My;
	double Mz = fmPtr->Mz;

    double Pdot = C1*Q*R + C2*P*Q + C3*Mx + C4*Mz;
    double Qdot = C5*P*R - C6*(P*P-R*R) + C7*My;
    double Rdot = C8*P*Q - C2*Q*R + C4*Mx + C9*Mz;
  
    double nposdot = (q0*q0 + q1*q1 - q2*q2 - q3*q3) * vel_u + 2*(q1*q2 - q0*q3) * vel_v + 2*(q1*q3 + q0*q2) * vel_w;
    double eposdot = 2*(q1*q2 + q0*q3) * vel_u + (q0*q0 - q1*q1 + q2*q2 - q3*q3) * vel_v + 2*(q2*q3 - q0*q1) * vel_w;
    double altdot = (-1) * (2*(q1*q3 - q0*q2) * vel_u + 2*(q2*q3 + q0*q1) * vel_v + (q0*q0 - q1*q1 - q2*q2 + q3*q3) * vel_w);

    dPtr->p = Pdot;
    dPtr->q = Qdot;
    dPtr->r = Rdot;

	dPtr->h = altdot;
	dPtr->npos = nposdot;
	dPtr->epos = eposdot;
}

void MyPlane::pout(PlaneState *stsOut, PlaneState *stsOut_d)
{
	int i=0;
	stsOut->alpha_d = stsOut_d->alpha;
	stsOut->beta_d = stsOut_d->beta;
}

void MyPlane::poutQuan(PlaneState *stsOut, PlaneState *stsOut_d)
{
	pout(stsOut, stsOut_d);
	quaterToEular(&stsOut->qw, &stsOut->phi);
}