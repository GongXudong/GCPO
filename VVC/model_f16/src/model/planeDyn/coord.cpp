
/* 
 * Coordinate conversion function
 * The US mode coordinate
 * Created by whh on 20210427
 */

#include "coord.h"

double r2d = 57.295779513;

void body2ground(double* bc, double* gc, double phi, double theta, double psi, bool deg)
{
	double sphi, cphi, st, ct, spsi, cpsi;
	if(deg)
	{
		sphi = sin(phi / r2d);
		st = sin(theta / r2d);
		spsi = sin(psi / r2d);
		cphi = cos(phi / r2d);
		ct = cos(theta / r2d);
		cpsi = cos(psi / r2d);
	}
	else
	{
		sphi = sin(phi);
		st = sin(theta);
		spsi = sin(psi);
		cphi = cos(phi);
		ct = cos(theta);
		cpsi = cos(psi);
	}
	gc[0] = bc[0]*(ct*cpsi) + bc[1]*(sphi*st*cpsi - cphi*spsi) + bc[2]*(cphi*st*cpsi + sphi*spsi);
    gc[1] = bc[0]*(ct*spsi) + bc[1]*(sphi*st*spsi + cphi*cpsi) + bc[2]*(cphi*st*spsi - sphi*cpsi);
    gc[2] = - bc[0]*st + bc[1]*(sphi*ct) + bc[2]*(cphi*ct);
}

void ground2body(double* gc, double* bc, double phi, double theta, double psi, bool deg)
{
	double sphi, cphi, st, ct, spsi, cpsi;
	if(deg)
	{
		sphi = sin(phi / r2d);
		st = sin(theta / r2d);
		spsi = sin(psi / r2d);
		cphi = cos(phi / r2d);
		ct = cos(theta / r2d);
		cpsi = cos(psi / r2d);
	}
	else
	{
		sphi = sin(phi);
		st = sin(theta);
		spsi = sin(psi);
		cphi = cos(phi);
		ct = cos(theta);
		cpsi = cos(psi);
	}
	bc[0] = gc[0]*(ct*cpsi) + gc[1]*(ct*spsi) - gc[2]*st;
    bc[1] = gc[0]*(sphi*st*cpsi - cphi*spsi) + gc[1]*(sphi*st*spsi + cphi*cpsi) + gc[2]*(sphi*ct);
    bc[2] = gc[0]*(cphi*st*cpsi + sphi*spsi) + gc[1]*(cphi*st*spsi - sphi*cpsi) + gc[2]*(cphi*ct);
}

void body2wind(double bc[3], double wc[3], double alpha, double beta, bool deg)
{
	double sa, ca, sb, cb;
	if(deg)
	{
		sa = sin(alpha / r2d);
		ca = cos(alpha / r2d);
		sb = sin(beta / r2d);
		cb = cos(beta / r2d);
	}
	else
	{
		sa = sin(alpha);
		ca = cos(alpha);
		sb = sin(beta);
		cb = cos(beta);
	}

	double bwmatric[3][3] = 
	{
		{ca*cb, sb,	sa*cb},
		{-ca*sb,cb, -sa*sb},
		{-sa,	0,	ca}
	};

	for(int i=0; i<3; i++)
	{
		wc[i] = 0;
		for(int j=0; j<3; j++)
		{
			wc[i] += bc[j] * bwmatric[i][j];
		}
	}
}

void wind2body(double wc[3], double bc[3], double alpha, double beta, bool deg)
{
	double sa, ca, sb, cb;
	if(deg)
	{
		sa = sin(alpha / r2d);
		ca = cos(alpha / r2d);
		sb = sin(beta / r2d);
		cb = cos(beta / r2d);
	}
	else
	{
		sa = sin(alpha);
		ca = cos(alpha);
		sb = sin(beta);
		cb = cos(beta);
	}

	double wbmatric[3][3] = 
	{
		{ca*cb, -ca*sb, -sa},
		{sb,    cb,    0},
		{sa*cb, -sa*sb, ca}
	};

	for(int i=0; i<3; i++)
	{
		bc[i] = 0;
		for(int j=0; j<3; j++)
		{
			bc[i] += wc[j] * wbmatric[i][j];
		}
	}
}

void stability2body(double sc[3], double bc[3], double alpha, double beta, bool deg)
{
	wind2body(sc, bc, alpha, 0, deg);
}

void body2stability(double bc[3], double sc[3], double alpha, double beta, bool deg)
{
	body2wind(bc, sc, alpha, 0, deg);
}
