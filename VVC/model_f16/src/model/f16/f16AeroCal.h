#ifndef _F16AEROCAL_H
#define _F16AEROCAL_H

#include "f16AeroData.h"
#include "../planeDyn/planeDynUS.h"
#include "../planeDyn/toolsDyn.h"

class F16Aero
{
public:
    F16Aero(const char *fname);
    ~F16Aero();

public:
    F16AeroData *f16data;
    double Cx, Cy, Cz, Cl, Cm, Cn;
    double Cxq, Cyp, Cyr, Czq, Clp, Clr, Cmq, Cnp, Cnr;
    double delta_Cy_a20, delta_Cl_a20, delta_Cn_a20;
    double delta_Cy_r30, delta_Cl_r30, delta_Cn_r30;
    double delta_Clbeta, delta_Cm, delta_Cnbeta;
    double eta_el, delta_Cm_ds;

    // leading edge flap
    double delta_Cx_lef, delta_Cy_lef, delta_Cz_lef;
    double delta_Cl_lef, delta_Cm_lef, delta_Cn_lef;
    double delta_Cxq_lef, delta_Cyp_lef, delta_Cyr_lef, delta_Czq_lef;
    double delta_Clp_lef, delta_Clr_lef, delta_Cmq_lef, delta_Cnp_lef, delta_Cnr_lef;
    double delta_Cy_a20_lef, delta_Cl_a20_lef, delta_Cn_a20_lef;

    double m;
    double B;
    double S;
    double cbar;
    double xcg;
    double xcgr;
    double Heng;
    double Jx;
    double Jy;
    double Jz;
    double Jxz;
    double faip;		// 发动机安装角, rad

    double damage[10];  // damage model

public:
    void planeDef(double mt, double Bt, double St, double cbart, double xcgrt, double xcgt, double Hengt, double Jyt, double Jxzt, double Jzt, double Jxt);
    void getInertia(double &m_out, double &Jx_out, double &Jy_out, double &Jz_out, double &Jxz_out, double &faip_out);
    void lookupTable(double alpha, double beta, double el);
    void calAero(PlaneState* stin, Actuator* ctrlin, FM* fmout);
	void calAeroWithDamage(PlaneState* stin, Actuator* ctrlin, FM* fmout);
    void gettrimval(double alt, double v, double alpha_r, double de_r, double* L, double* V, double* W, double* L_tot, double *M_tot, double *N_tot);
    int trim(double alt, double v, double *ALF, double *dE, double *PLA, double faip = 0, int loop_num = 20, double err = 1e-4);
};

#endif