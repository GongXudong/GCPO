#ifndef _PLANEDYNUS_H
#define _PLANEDYNUS_H

#include "toolsDyn.h"

// rad
typedef struct{
double  vt,
        alpha,
        beta,
		phi,    // 滚转角
        theta,  // 俯仰角
        psi,    // 偏航角
        p,
        q,
        r,
        h,
		npos,
		epos,
		qw,
        qx,
        qy,
		qz;
double  alpha_d,	// derivative of alpha
		beta_d;
} PlaneState;

// degree
typedef struct{
double  vt,
        alpha,
        beta,
		phi,    // 滚转角
        theta,  // 俯仰角
        psi,    // 偏航角
        p,
        q,
        r,
        h,
		npos,
		epos,
		qw,
        qx,
        qy,
		qz;
double	vn,
		ve,
		vh,
		mu,
		chi,
		nx,
		ny,
		nz,
		nx_ref,
		ny_ref,
		nz_ref,
		lon,
		lat,
		earth_R,
		latC;
double  ail,   //副翼
		ele,   //方向舵
		rud,   //升降舵 
		thrust,//推力
		lef,   //前缘襟翼
		jsb;   //减速板
} PlaneStateInfo;

// degree
typedef struct{
double  ail,   //副翼
		ele,   //方向舵
		rud,   //升降舵 
		thrust,//推力
		lef,   //前缘襟翼
		jsb;   //减速板
} Actuator;

typedef struct ForceMoment{
double  Fx,
		Fy,
		Fz,
		Mx,	// moment of x axis
        My,
        Mz;
double  D,	// 阻力
		Y,	// 升力
		Z,  // 侧力
		T;  // 推力
double  faip;	// 发动机安装角
double  My_trim;
} FM;

class MyPlane
{
public:
    int Order;
	int MaxState;
    double IStep;
	
	double m;
    double C1;
    double C2;
    double C3;
    double C4;
    double C5;
    double C6;
    double C7;
    double C8;
    double C9;
	double faip;	// 发动机安装角

    PlaneState *rk1, *rk2, *rk3, *rk4, *tmp;
    PlaneState *lastout;
    void pf(FM *fmPtr, PlaneState *inPtr, PlaneState *dPtr);
	void pfUVW(FM *fmPtr, PlaneState *inPtr, PlaneState *dPtr);
	void pf_quaternion(FM *fmPtr, PlaneState *inPtr, PlaneState *dPtr);
    void pout(PlaneState * EquValInPtr, PlaneState *OutPtr);

public:
    MyPlane(double step=1/320.0,int order=2);
    ~MyPlane();
    void set(double step,int order);
	void setPlaneParams(double mass, double Ix_in, double Iy_in, double Iz_in, double Ixz_in, double faip_in=0);
    void operator()(FM *fmPtr, PlaneState * stPtr);
	void runQuan(FM *fmPtr, PlaneState * stPtr);
	void poutQuan(PlaneState * EquValInPtr, PlaneState *OutPtr);
};

#endif