

#include "f16AeroCal.h"

F16Aero::F16Aero(const char *fname)
{
    /* plane constants (f16 by default) */
    m = 9295.44;    /* mass, kg */
    B = 9.144;      /* span, m */
    S = 27.87;      /* planform area, m^2 */
    cbar = 3.45;    /* mean aero chord, m */
    xcg = 0.30;     /* center of gravity as a fraction of cbar. */
    xcgr = 0.35;    /* reference center of gravity as a fraction of cbar */
    Heng = 216.9;   /* turbine momentum along roll axis. */
    Jx = 12874.8;
    Jy = 75673.6;
    Jz = 85552.1;
    Jxz = 1331.4;
    faip = 0;		// 发动机安装角, rad
	for(int i=0; i<10; i++)
	{
		damage[i] = 1.0;
	}
    f16data = new F16AeroData(fname);
}

F16Aero::~F16Aero()
{
    delete f16data;
}


void F16Aero::planeDef(double mt, double Bt, double St, double cbart, double xcgrt, double xcgt, double Hengt, double Jyt, double Jxzt, double Jzt, double Jxt)
{
    m = mt;             /* mass,  kg */
    B = Bt;             /* span, m */
    S = St;             /* planform area, m^2 */
    cbar = cbart;       /* mean aero chord, m */
    Heng = Hengt;       /* turbine momentum along roll axis. */
	// Heng = 0.0;
    xcg = xcgt;         /* center of gravity as a fraction of cbar. */
    xcgr = xcgrt;       /* reference center of gravity as a fraction of cbar */
    Jx = Jxt;
    Jy = Jyt;
    Jz = Jzt;
    Jxz = Jxzt;
}

void F16Aero::getInertia(double &m_out, double &Jx_out, double &Jy_out, double &Jz_out, double &Jxz_out, double &faip_out)
{
	m_out = m;
	Jx_out = Jx;
	Jy_out = Jy;
	Jz_out = Jz;
	Jxz_out = Jxz;
	faip_out = faip;
}

void F16Aero::lookupTable(double alpha, double beta, double el)
{
	Cx = f16data->_Cx(alpha,beta,el);
    Cy = f16data->_Cy(alpha,beta);
    Cz = f16data->_Cz(alpha,beta,el);
    Cl = f16data->_Cl(alpha,beta,el);
    Cm = f16data->_Cm(alpha,beta,el);
    Cn = f16data->_Cn(alpha,beta,el);

    Cxq = f16data->_Cxq(alpha);
    Cyp = f16data->_Cyp(alpha);
    Cyr = f16data->_Cyr(alpha);
    Czq = f16data->_Czq(alpha);
    Clp = f16data->_Clp(alpha);
    Clr = f16data->_Clr(alpha);
    Cmq = f16data->_Cmq(alpha);
    Cnp = f16data->_Cnp(alpha);
    Cnr = f16data->_Cnr(alpha);

	delta_Cy_a20 = f16data->_Cy_a20(alpha,beta) - f16data->_Cy(alpha,beta);
    delta_Cl_a20 = f16data->_Cl_a20(alpha,beta) - f16data->_Cl(alpha,beta,0);
    delta_Cn_a20 = f16data->_Cn_a20(alpha,beta) - f16data->_Cn(alpha,beta,0);

	delta_Cy_r30 = f16data->_Cy_r30(alpha,beta) - f16data->_Cy(alpha,beta);
    delta_Cl_r30 = f16data->_Cl_r30(alpha,beta) - f16data->_Cl(alpha,beta,0);
    delta_Cn_r30 = f16data->_Cn_r30(alpha,beta) - f16data->_Cn(alpha,beta,0);

	delta_Clbeta = f16data->_delta_Clbeta(alpha);
    delta_Cm     = f16data->_delta_Cm(alpha);
    delta_Cnbeta = f16data->_delta_Cnbeta(alpha);

    eta_el = f16data->_eta_el(el);
    delta_Cm_ds = f16data->_delta_Cm_ds(alpha,el);
	// delta_Cm_ds  = 0;        /* ignore deep-stall effect */

    // leading edge flap
    delta_Cx_lef = f16data->_Cx_lef(alpha,beta) - f16data->_Cx(alpha,beta,0);
    delta_Cy_lef = f16data->_Cy_lef(alpha,beta) - f16data->_Cy(alpha,beta);
    delta_Cz_lef = f16data->_Cz_lef(alpha,beta) - f16data->_Cz(alpha,beta,0);

    delta_Cl_lef = f16data->_Cl_lef(alpha,beta) - f16data->_Cl(alpha,beta,0);
    delta_Cm_lef = f16data->_Cm_lef(alpha,beta) - f16data->_Cm(alpha,beta,0);
    delta_Cn_lef = f16data->_Cn_lef(alpha,beta) - f16data->_Cn(alpha,beta,0);

    delta_Cxq_lef = f16data->_delta_Cxq_lef(alpha);
    delta_Cyp_lef = f16data->_delta_Cyp_lef(alpha);
    delta_Cyr_lef = f16data->_delta_Cyr_lef(alpha);
    delta_Czq_lef = f16data->_delta_Czq_lef(alpha);
    delta_Clp_lef = f16data->_delta_Clp_lef(alpha);
    delta_Clr_lef = f16data->_delta_Clr_lef(alpha);
    delta_Cmq_lef = f16data->_delta_Cmq_lef(alpha);
    delta_Cnp_lef = f16data->_delta_Cnp_lef(alpha);
    delta_Cnr_lef = f16data->_delta_Cnr_lef(alpha);

    delta_Cy_a20_lef = f16data->_Cy_a20_lef(alpha,beta) - f16data->_Cy_lef(alpha,beta) - delta_Cy_a20;
    delta_Cl_a20_lef = f16data->_Cl_a20_lef(alpha,beta) - f16data->_Cl_lef(alpha,beta) - delta_Cl_a20;
    delta_Cn_a20_lef = f16data->_Cn_a20_lef(alpha,beta) - f16data->_Cn_lef(alpha,beta) - delta_Cn_a20;
}

void F16Aero::calAero(PlaneState* stin, Actuator* ctrlin, FM* fmout)
{
	double alpha = stin->alpha * R2D;
	double beta = stin->beta * R2D;
	double ele = ctrlin->ele;
	lookupTable(alpha, beta, ele);
	
	double Cx_tot, dXdQ;
	double Cy_tot, dYdail, dYdR, dYdP;
	double Cz_tot, dZdQ;
	double Cl_tot, dLdail, dLdR, dLdP;
	double Cm_tot, dMdQ;
	double Cn_tot, dNdail, dNdR, dNdP;
	
	double vt = stin->vt;
	double alt = stin->h;
	double P = stin->p;
	double Q = stin->q;
	double R = stin->r;
	
	double dail = ctrlin->ail/21.5;  /* aileron normalized against max angle */
    double drud = ctrlin->rud/30.0;  /* rudder normalized against max angle */
    double dlef = 1 - ctrlin->lef / 25.0;  /* leading edge flap normalized against max angle */
    
	/* compute Cx_tot, Cz_tot, Cm_tot, Cy_tot, Cn_tot, and Cl_tot(as on NASA report p37-40) */
    /* XXXXXXXX Cx_tot XXXXXXXX */
    dXdQ = (cbar/(2*vt)) * (Cxq + delta_Cxq_lef*dlef);
    Cx_tot = Cx + delta_Cx_lef*dlef + dXdQ*Q;
	Cx_tot = 1.6 * Cx_tot;	// wh, 20210315, modified

    /* YYYYYYYY Cy_tot YYYYYYYY */
    dYdail = delta_Cy_a20 + delta_Cy_a20_lef*dlef;
    dYdR = (B/(2*vt))*(Cyr + delta_Cyr_lef*dlef);
    dYdP = (B/(2*vt))*(Cyp + delta_Cyp_lef*dlef);
    Cy_tot = Cy + delta_Cy_lef*dlef + dYdail*dail + delta_Cy_r30*drud + dYdR*R + dYdP*P;
    // printf("Cy_tot: %f, %f, %f, %f, %f, %f\n", Cy_tot, delta_Cy_lef*dlef, dYdail*dail, delta_Cy_r30*drud, dYdR*R, dYdP*P);
    // printf("Cy: %f, %f, %f, %f, %f, %f\n", Cy, delta_Cy_lef, dYdail, delta_Cy_r30, dYdR, dYdP);
    // printf("dlef: %f, %f, %f, %f, %f\n", dlef, dail, drud, R, P);

    /* ZZZZZZZZ Cz_tot ZZZZZZZZ */ 
    dZdQ = (cbar/(2*vt)) * (Czq + delta_Cz_lef*dlef);
    Cz_tot = Cz + delta_Cz_lef*dlef + dZdQ*Q;

    /* LLLLLLLL Cl_tot LLLLLLLL */
    dLdail = delta_Cl_a20 + delta_Cl_a20_lef*dlef;
    dLdR = (B/(2*vt)) * (Clr + delta_Clr_lef*dlef);
    dLdP = (B/(2*vt)) * (Clp + delta_Clp_lef*dlef);
    Cl_tot = Cl + delta_Cl_lef*dlef + dLdail*dail + delta_Cl_r30*drud + dLdR*R + dLdP*P + delta_Clbeta*beta;

    /* MMMMMMMM Cm_tot MMMMMMMM */ 
    dMdQ = (cbar/(2*vt))*(Cmq + delta_Cmq_lef*dlef);
    Cm_tot = Cm*eta_el + Cz_tot*(xcgr-xcg) + delta_Cm_lef*dlef + dMdQ*Q + delta_Cm + delta_Cm_ds;

    /* NNNNNNNN Cn_tot NNNNNNNN */ 
    dNdail = delta_Cn_a20 + delta_Cn_a20_lef*dlef;
    dNdR = (B/(2*vt)) * (Cnr + delta_Cnr_lef*dlef);
    dNdP = (B/(2*vt)) * (Cnp + delta_Cnp_lef*dlef);
    Cn_tot = Cn + delta_Cn_lef*dlef - Cy_tot*(xcgr-xcg)*(cbar/B) + dNdail*dail + delta_Cn_r30*drud + dNdR*R + dNdP*P + delta_Cnbeta*beta;

	double tmp[3];
	atmos(alt, vt, tmp);
	double qbar = tmp[1];

    // printf("Fx: %f, %f, %f, %f, %f, %f\n", Cx_tot, Cy_tot, Cz_tot, Cl_tot, Cm_tot, Cn_tot);
	fmout->Fx = qbar * S * Cx_tot; 
    fmout->Fy = qbar * S * Cy_tot;
    fmout->Fz = qbar * S * Cz_tot;
	fmout->T = ctrlin->thrust;
	fmout->faip = faip;

	fmout->Mx = Cl_tot * qbar * S * B;       /* get moments from coefficients */
    fmout->My = Cm_tot * qbar * S * cbar - R*Heng;
    fmout->Mz = Cn_tot * qbar * S * B + Q*Heng;
}

void F16Aero::calAeroWithDamage(PlaneState* stin, Actuator* ctrlin, FM* fmout)
{
	double alpha = stin->alpha * R2D;
	double beta = stin->beta * R2D;
	double ele = ctrlin->ele;
	lookupTable(alpha, beta, ele);
	
	double Cx_tot, dXdQ;
	double Cy_tot, dYdail, dYdR, dYdP;
	double Cz_tot, dZdQ;
	double Cl_tot, dLdail, dLdR, dLdP;
	double Cm_tot, dMdQ;
	double Cn_tot, dNdail, dNdR, dNdP;
	
	double vt = stin->vt;
	double alt = stin->h;
	double P = stin->p;
	double Q = stin->q;
	double R = stin->r;
	
	double dail = ctrlin->ail/21.5;  /* aileron normalized against max angle */
    double drud = ctrlin->rud/30.0;  /* rudder normalized against max angle */
    double dlef = 1 - ctrlin->lef / 25.0;  /* leading edge flap normalized against max angle */

	/* damage evaluate, evaluate elevator effects*/
    double Cxde0, Czde0, Clde0, Cmde0, Cnde0, Cnde00; // damage evaluate
    double f_da, f_de, f_dr, f_wlon, f_wlat, f_vtail;

	/*damage factor*/
    f_de = 0.9 * (damage[5] + damage[6]) - 0.8;
    f_da = 0.9 * (damage[1] * damage[3] + damage[2] * damage[4]) - 0.8;
    f_dr = damage[7] * damage[8];
    f_wlon = (damage[1] + damage[2]) * 0.5;
    f_wlat = damage[2] - damage[1];
	f_vtail = damage[7];
	dail = f_da * dail;
	drud = f_dr * drud;

    Cxde0 = f16data->_Cx(alpha, beta, 0);
    Czde0 = f16data->_Cz(alpha, beta, 0);
    Clde0 = f16data->_Cl(alpha, beta, 0);
    Cmde0 = f16data->_Cm(alpha, beta, 0);
    Cnde0 = f16data->_Cn(alpha, beta, 0);
	double betatmp;
	double cndelta = 0;
	double cndeltade0 = 0;
	if(f_vtail < 0.9)
	{
		cndelta = f16data->_Cn(alpha, 0, ele) - Cn;
		cndeltade0 = f16data->_Cn(alpha, 0, 0) - Cnde0;
		cndelta = cndeltade0 + (cndelta - cndeltade0) * f_de;
		cndelta = cndelta * (1.8 - 0.4 * f_vtail);
	}

    Cx = Cxde0 + (Cx - Cxde0) * f_de;
    Cz = Czde0 + (Cz - Czde0) * f_de;
    Cl = Clde0 + (Cl - Clde0) * f_de;
    Cm = Cmde0 + (Cm - Cmde0) * f_de;
    Cn = Cnde0 + (Cn - Cnde0) * f_de + cndelta;
    
	/* compute Cx_tot, Cz_tot, Cm_tot, Cy_tot, Cn_tot, and Cl_tot(as on NASA report p37-40) */
    /* XXXXXXXX Cx_tot XXXXXXXX */
    dXdQ = (cbar/(2*vt)) * (Cxq + delta_Cxq_lef*dlef);
    Cx_tot = f_wlon * (Cx + delta_Cx_lef*dlef) + dXdQ*Q;
	Cx_tot = 1.6 * Cx_tot;	// wh, 20210315, modified

    /* YYYYYYYY Cy_tot YYYYYYYY */
    dYdail = delta_Cy_a20 + delta_Cy_a20_lef*dlef;
    dYdR = (B/(2*vt))*(Cyr + delta_Cyr_lef*dlef);
    dYdP = (B/(2*vt))*(Cyp + delta_Cyp_lef*dlef);
    Cy_tot = Cy + delta_Cy_lef*dlef + dYdail*dail + delta_Cy_r30*drud + dYdR*R + dYdP*P;

    /* ZZZZZZZZ Cz_tot ZZZZZZZZ */ 
    dZdQ = (cbar/(2*vt)) * (Czq + delta_Cz_lef*dlef);
    Cz_tot = Cz + delta_Cz_lef*dlef + dZdQ*Q;
	Cz_tot = f_wlon * Cz_tot;

    /* LLLLLLLL Cl_tot LLLLLLLL */
    dLdail = delta_Cl_a20 + delta_Cl_a20_lef*dlef;
    dLdR = (B/(2*vt)) * (Clr + delta_Clr_lef*dlef);
    dLdP = (B/(2*vt)) * (Clp + delta_Clp_lef*dlef);
    Cl_tot = Cl + delta_Cl_lef*dlef + 0.2 * Cz_tot * f_wlat * 0.25 + dLdail*dail + delta_Cl_r30*drud + dLdR*R + dLdP*P + delta_Clbeta*beta;

    /* MMMMMMMM Cm_tot MMMMMMMM */ 
    dMdQ = (cbar/(2*vt))*(Cmq + delta_Cmq_lef*dlef);
    Cm_tot = Cm*eta_el + Cz_tot*(xcgr-xcg) + delta_Cm_lef*dlef + dMdQ*Q + delta_Cm + delta_Cm_ds;

    /* NNNNNNNN Cn_tot NNNNNNNN */ 
    dNdail = delta_Cn_a20 + delta_Cn_a20_lef*dlef;
    dNdR = (B/(2*vt)) * (Cnr + delta_Cnr_lef*dlef);
    dNdP = (B/(2*vt)) * (Cnp + delta_Cnp_lef*dlef);
    Cn_tot = Cn + delta_Cn_lef*dlef - Cy_tot*(xcgr-xcg)*(cbar/B) + dNdail*dail + delta_Cn_r30*drud + dNdR*R + dNdP*P + delta_Cnbeta*beta;

	double tmp[3];
	atmos(alt, vt, tmp);
	double qbar = tmp[1];

    // printf("Fx: %f, %f, %f, %f, %f, %f\n", Cx_tot, Cy_tot, Cz_tot, Cl_tot, Cm_tot, Cn_tot);
	fmout->Fx = qbar * S * Cx_tot; 
    fmout->Fy = qbar * S * Cy_tot;
    fmout->Fz = qbar * S * Cz_tot;
    ctrlin->thrust = ctrlin->thrust * damage[9];// damage evaluate
	fmout->T = ctrlin->thrust;
	fmout->faip = faip;

	fmout->Mx = Cl_tot * qbar * S * B;       /* get moments from coefficients */
    fmout->My = Cm_tot * qbar * S * cbar - R*Heng;
    fmout->Mz = Cn_tot * qbar * S * B + Q*Heng;
}

void F16Aero::gettrimval(double alt, double v, double alpha_r, double de_r, double* L, double* V, double* W, double* L_tot, double *M_tot, double *N_tot)
{
    PlaneState ststmp;
	Actuator ctmp;
	FM fmtmp;
	memset(&ststmp, 0, sizeof(PlaneState));
	memset(&ctmp, 0, sizeof(Actuator));
	memset(&fmtmp, 0, sizeof(FM));

	ststmp.h = alt;
	ststmp.vt = v;
	ststmp.alpha = alpha_r;
	ctmp.ele = de_r * R2D;
	
	calAero(&ststmp, &ctmp, &fmtmp);
    double Fx_tot_body, Fy_tot_body, Fz_tot_body;

    double sa = sin(ststmp.alpha);
    double ca = cos(ststmp.alpha);

	*L_tot = fmtmp.Mx;       /* get moments from coefficients */
    *M_tot = fmtmp.My;
    *N_tot = fmtmp.Mz;

    Fx_tot_body = fmtmp.Fx; 
    Fy_tot_body = fmtmp.Fy;
    Fz_tot_body = fmtmp.Fz;

    //Dcm Body2Wind, beta=0
    *L = Fx_tot_body * ca + 0 + Fz_tot_body * sa;
    *V = 0 + Fy_tot_body + 0;
    *W = Fx_tot_body * (-1) * sa + 0 + Fz_tot_body * ca;
}

// alf: rad, dE: rad, PLA: N
int F16Aero::trim(double alt, double v, double *ALF, double *dE, double *PLA, double faip, int loop_num, double err)
{
    double jac[2][2];
    double jbc[2];
    double t;
    double dalf = 1e-10 / R2D;
    double ddeta_z = 1e-10 / R2D;
    double A = 1000;
    double B = 10000;
    double L, V, W, Lm, Mm, Nm;
    double beta = 0;
    int i;
    *ALF = 0.0;
    *dE = 0.0;

    for(i = 0; i < loop_num; i++ )
    {
        if(fabs(A) < err && fabs(B) < err)
        {
            break;
        }
        gettrimval(alt, v, *ALF, *dE, &L, &V, &W, &Lm, &Mm, &Nm);
        A = -L * tan(faip + *ALF) - W - m*g;     // T*sin(alpha) = L*tan(alpha)
        B = Mm;
        jbc[0] = A; jbc[1] = B;

        *ALF = *ALF + dalf;
        gettrimval(alt, v, *ALF, *dE, &L, &V, &W, &Lm, &Mm, &Nm);

        jac[0][0] = (-L*tan(faip + *ALF + dalf) - W - m*g - A) / dalf;
        jac[1][0] = (Mm - B) / dalf;

        *dE = *dE + ddeta_z;
        gettrimval(alt, v, *ALF, *dE, &L, &V, &W, &Lm, &Mm, &Nm);

        jac[0][1] = (-L*tan(faip + *ALF) - W - m*g - A) / ddeta_z;
        jac[1][1] = (Mm - B) / ddeta_z;

        t = jac[0][0]*jac[1][1] - jac[1][0]*jac[0][1];

        if(t == 0)
        {
            return -2;
        }
        *ALF -= (jbc[0]*jac[1][1] - jbc[1]*jac[0][1]) / t;
        *dE -= (-jbc[0]*jac[1][0] + jbc[1]*jac[0][0]) / t;
    }
    *PLA = -L / cos(faip + *ALF);
    if(i == loop_num ) return -1;
    return 0;
}