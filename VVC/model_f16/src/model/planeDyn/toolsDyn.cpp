#include "toolsDyn.h"

/*########################################*/
/*### Added for mex function in matlab ###*/
/*########################################*/

void atmos(double altt, double vtt, double *coeff ){
	double rho0 = 1.225;
    double Re = 6371000;
    double R = 287.05;
    double T0 = 288.15;
    double g0 = 9.80665;
    double gamma = 1.4;
    double temp, rho, mach, qbar, gr;

    temp = T0 - 0.0065 * altt;
    if (altt >= 11000.0) {
       temp = 216.65;
    }

    rho = rho0 * exp((-g0 / (R * temp)) * altt);
    mach = vtt / sqrt(gamma * R * temp);
    qbar = 0.5 * rho * vtt * vtt;
    gr = g0*(Re*Re/((Re+altt)*(Re+altt)));
 
    coeff[0] = mach;
    coeff[1] = qbar;
    coeff[2] = gr;
}

/*########################################*/
/*### Calculate accelerations from states */
/*### and state derivatives. ############ */
/*########################################*/
void accels(double *state, double *xdot, double *y)
{
	double sina, cosa, sinb, cosb ;
	double vel_u, vel_v, vel_w ;
	double u_dot, v_dot, w_dot ;
	double nx_cg, ny_cg, nz_cg ;

	sina = sin(state[7]) ;
	cosa = cos(state[7]) ;
	sinb = sin(state[8]) ;
	cosb = cos(state[8]) ;

	vel_u = state[6]*cosb*cosa ;
	vel_v = state[6]*sinb ;
	vel_w = state[6]*cosb*sina ;

	u_dot = cosb*cosa*xdot[6] - state[6]*sinb*cosa*xdot[8] - state[6]*cosb*sina*xdot[7];
	v_dot = sinb*xdot[6] + state[6]*cosb*xdot[8];
	w_dot = cosb*sina*xdot[6] - state[6]*sinb*sina*xdot[8] + state[6]*cosb*cosa*xdot[7];

	nx_cg = 1.0/g*(u_dot + state[10]*vel_w - state[11]*vel_v) + sin(state[4]);
	ny_cg = 1.0/g*(v_dot + state[11]*vel_u - state[9]*vel_w) - cos(state[4])*sin(state[3]);
	nz_cg = -1.0/g*(w_dot + state[9]*vel_v - state[10]*vel_u) + cos(state[4])*cos(state[3]);

	y[0] = nx_cg ;
	y[1] = ny_cg ;
	y[2] = nz_cg ;
}

/*########################################*/
/*### Port from matlab fix() function ####*/
/*########################################*/
int fix(double in)
{
    int out;

    if (in >= 0.0){
       out = (int)floor(in);
    }
    else if (in < 0.0){
       out = (int)ceil(in);
    }

    return out;
}

/* port from matlab sign() function */
int sign(double in)
{
    int out;

    if (in > 0.0){
       out = 1;
    }
    else if (in < 0.0){
       out = -1;
    }
    else if (in == 0.0){
       out = 0;
    }
    return out;
}

/*
* convert eular angles to quaternion params
* equations are from Russian doc, have been modified to adapt to the US coordinate
*/
void eularToQuater_fromRu(double eangles[], double qangles[])
{
    double spsi2, cpsi2, sphi2, cphi2, st2, ct2;
    sphi2 = sin(eangles[0] / 2);
    cphi2 = cos(eangles[0] / 2);
    st2 = sin(eangles[1] / 2);
    ct2 = cos(eangles[1] / 2);
    spsi2 = sin(eangles[2] / 2);
    cpsi2 = cos(eangles[2] / 2);
    qangles[0] = cpsi2 * ct2 * cphi2 + spsi2 * st2 * sphi2;
    qangles[1] = cpsi2 * ct2 * sphi2 - spsi2 * st2 * cphi2;
    qangles[2] = cpsi2 * st2 * sphi2 - spsi2 * ct2 * cphi2;
    qangles[3] = cpsi2 * st2 * cphi2 + spsi2 * ct2 * sphi2;
}

/*
* convert quaternion params to eular angles
*/
void quaterToEular_fromRu(double qangles[], double eangles[])
{
    double qw, qx, qy, qz, a11, a13;
    qw = qangles[0];
    qx = qangles[1];
    qy = qangles[2];
    qz = qangles[3];
    
    a11 = qw*qw + qx*qx - qy*qy - qz*qz;
    a13 = 2 * (qx * qz - qw * qy);
        
    eangles[0] = - atan2(2 * (qy*qz - qw*qx), qw*qw - qx*qx + qy*qy - qz*qz);    // phi
    eangles[1] = atan(2 * (qw * qz + qx * qy) / sqrt(a11*a11 + a13*a13));      // theta
    eangles[2] = atan2(a13, a11);    // psi
}

/*
* quaternion calculation
*/
void quaterCal_fromRu(double p, double q, double r, double dt, double qangles[])
{
    double qw, qx, qy, qz, ep, qwdot, qxdot, qydot, qzdot;
    qw = qangles[0];
    qx = qangles[1];
    qy = qangles[2];
    qz = qangles[3];
    
    ep = 2 * (1 - qw*qw - qx*qx - qy*qy - qz*qz);

    qwdot = ep * qw - 0.5 * (qx * p - qy * r + qz * q);
    qxdot = ep * qx + 0.5 * (qw * p + qy * q + qz * r);
    qydot = ep * qy + 0.5 * (qz * p - qw * r - qx * q);
    qzdot = ep * qz + 0.5 * (qw * q - qx * r - qy * p);
        
    qw = qw + qwdot * dt;
    qx = qx + qxdot * dt;
    qy = qy + qydot * dt;
    qz = qz + qzdot * dt;
    
    qangles[0] = qw;
    qangles[1] = qx;
    qangles[2] = qy;
    qangles[3] = qz;
}

/*
* convert eular angles to quaternion params
* equations are from F16 NASA code, have been corrected and modified with Russian doc.
*/
void eularToQuater(double eangles[], double qangles[])
{
    double spsi2, cpsi2, sphi2, cphi2, st2, ct2;
    sphi2 = sin(eangles[0] / 2);
    cphi2 = cos(eangles[0] / 2);
    st2 = sin(eangles[1] / 2);
    ct2 = cos(eangles[1] / 2);
    spsi2 = sin(eangles[2] / 2);
    cpsi2 = cos(eangles[2] / 2);
    qangles[0] = cphi2 * ct2 * cpsi2 + sphi2 * st2 * spsi2;
    qangles[1] = sphi2 * ct2 * cpsi2 - cphi2 * st2 * spsi2;
    qangles[2] = cphi2 * st2 * cpsi2 + sphi2 * ct2 * spsi2;
	qangles[3] = cphi2 * ct2 * spsi2 - sphi2 * st2 * cpsi2;
}

void quaterToEular(double qangles[], double eangles[])
{
    double qw, qx, qy, qz;
    qw = qangles[0];
    qx = qangles[1];
    qy = qangles[2];
	qz = qangles[3];

	double a11 = qw*qw + qx*qx - qy*qy - qz*qz;
    double a13 = 2 * (qx * qy + qw * qz);
        
    eangles[0] = atan2(2 * (qy*qz + qw*qx), qw*qw - qx*qx + qz*qz - qy*qy);    // phi
    eangles[1] = atan(2 * (qw * qy - qx * qz) / sqrt(a11*a11 + a13*a13));      // theta
    eangles[2] = atan2(a13, a11);    // psi
}


double TAS2IAS(double TAS, double alt)
{
	double temp = 288.15 - 0.0065 * alt;
	if(alt >= 11000.0)
	{
		temp = 216.65;
	}
	double rho = 1.225 * exp((-9.80665 / (287.05 * temp)) * alt);
	double IAS = TAS * sqrt(rho/ 1.225);
	return IAS;
}

double IAS2TAS(double IAS, double alt)
{
    double temp = 288.15 - 0.0065 * alt;
	if(alt >= 11000.0)
	{
		temp = 216.65;
	}
	double rho = 1.225 * exp((-9.80665 / (287.05 * temp)) * alt);
    double TAS = IAS * sqrt(1.225 / rho);
    return TAS;
}

double soundSpeed(double alt){
	double sound_v;
	if (alt <= 11000)
	    sound_v = 20.0468*sqrt(288.15 - 0.0065*alt); 
	else if (alt<=20000)
		sound_v = 20.0468*sqrt(216.65);	
	else
		sound_v = 20.0468*sqrt(0.001*(alt-20000)+216.65);	

	return sound_v ;
}

// air density
double airDensity(double alt)
{
	double temp = 288.15 - 0.0065 * alt;
	if(alt >= 11000.0)
	{
		temp = 216.65;
	}
	double rho = 1.225 * exp((-9.80665 / (287.05 * temp)) * alt);

    return rho;
}

double muCal(double vh, double v)
{
    if(v <= 0)  return 0;
    else if(fabs(vh) >= v)
    {
        if(vh >0)        return 90;
        else             return -90;
    }
    else return asin(vh / v) * R2D;
}
