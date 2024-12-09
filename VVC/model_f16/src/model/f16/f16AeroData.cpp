
#include "f16AeroData.h"

F16AeroData::F16AeroData(const char *fname)
{
    libMng = new ResManger;
    useLib(fname);
    initTbf();
}

F16AeroData::~F16AeroData()
{
    freeTbf();
    unUseLib();
    delete libMng;
}

void F16AeroData::useLib(const char *fname)
{
    libMng->Use((char*)fname);
}

void F16AeroData::unUseLib()
{
    libMng->UnUse();
}

void F16AeroData::initTbf()
{
    _Cx_tbf = new DataTableExLib("Cx.tbf", libMng);
    _Cy_tbf = new DataTableExLib("Cy.tbf", libMng);
    _Cz_tbf = new DataTableExLib("Cz.tbf", libMng);
    _Cl_tbf = new DataTableExLib("Cl.tbf", libMng);
    _Cm_tbf = new DataTableExLib("Cm.tbf", libMng);
    _Cn_tbf = new DataTableExLib("Cn.tbf", libMng);

    _Cxq_tbf = new DataTableExLib("Cxq.tbf", libMng);
    _Cyp_tbf = new DataTableExLib("Cyp.tbf", libMng);
    _Cyr_tbf = new DataTableExLib("Cyr.tbf", libMng);
    _Czq_tbf = new DataTableExLib("Czq.tbf", libMng);
    _Clp_tbf = new DataTableExLib("Clp.tbf", libMng);
    _Clr_tbf = new DataTableExLib("Clr.tbf", libMng);
    _Cmq_tbf = new DataTableExLib("Cmq.tbf", libMng);
    _Cnp_tbf = new DataTableExLib("Cnp.tbf", libMng);
    _Cnr_tbf = new DataTableExLib("Cnr.tbf", libMng);

    _Cy_a20_tbf = new DataTableExLib("Cyda20.tbf", libMng);
    _Cl_a20_tbf = new DataTableExLib("Clda20.tbf", libMng);
    _Cn_a20_tbf = new DataTableExLib("Cnda20.tbf", libMng);
    _Cy_r30_tbf = new DataTableExLib("Cydr30.tbf", libMng);
    _Cl_r30_tbf = new DataTableExLib("Cldr30.tbf", libMng);
    _Cn_r30_tbf = new DataTableExLib("Cndr30.tbf", libMng);
    
    _Cx_lef_tbf = new DataTableExLib("Cxlf.tbf", libMng);
    _Cy_lef_tbf = new DataTableExLib("Cylf.tbf", libMng);
    _Cz_lef_tbf = new DataTableExLib("Czlf.tbf", libMng);
    _Cl_lef_tbf = new DataTableExLib("Cllf.tbf", libMng);
    _Cm_lef_tbf = new DataTableExLib("Cmlf.tbf", libMng);
    _Cn_lef_tbf = new DataTableExLib("Cnlf.tbf", libMng);
    
    _Cy_a20_lef_tbf = new DataTableExLib("Cyda20lf.tbf", libMng);
    _Cl_a20_lef_tbf = new DataTableExLib("Clda20lf.tbf", libMng);
    _Cn_a20_lef_tbf = new DataTableExLib("Cnda20lf.tbf", libMng);
    
    _delta_Cxq_lef_tbf = new DataTableExLib("dCxqlf.tbf", libMng);
    _delta_Cyp_lef_tbf = new DataTableExLib("dCyplf.tbf", libMng);
    _delta_Cyr_lef_tbf = new DataTableExLib("dCyrlf.tbf", libMng);
    _delta_Czq_lef_tbf = new DataTableExLib("dCzqlf.tbf", libMng);
    _delta_Clp_lef_tbf = new DataTableExLib("dClplf.tbf", libMng);
    _delta_Clr_lef_tbf = new DataTableExLib("dClrlf.tbf", libMng);
    _delta_Cmq_lef_tbf = new DataTableExLib("dCmqlf.tbf", libMng);
    _delta_Cnp_lef_tbf = new DataTableExLib("dCnplf.tbf", libMng);
    _delta_Cnr_lef_tbf = new DataTableExLib("dCnrlf.tbf", libMng);
    
    _delta_Cnbeta_tbf = new DataTableExLib("dCnbeta.tbf", libMng);
    _delta_Clbeta_tbf = new DataTableExLib("dClbeta.tbf", libMng);
    _delta_Cmbeta_tbf = new DataTableExLib("dCm.tbf", libMng);
    
    _delta_Cm_ds_tbf = new DataTableExLib("dCmds.tbf", libMng);
    // _eta_el_tbf = new DataTableExLib("Eta.tbf", libMng);
    _power_tbf = new DataTableExLib("power.tbf", libMng);
}

void F16AeroData::freeTbf()
{
    delete _Cx_tbf;
    delete _Cy_tbf;
    delete _Cz_tbf;
    delete _Cl_tbf;
    delete _Cm_tbf;
    delete _Cn_tbf;

    delete _Cxq_tbf;
    delete _Cyp_tbf;
    delete _Cyr_tbf;
    delete _Czq_tbf;
    delete _Clp_tbf;
    delete _Clr_tbf;
    delete _Cmq_tbf;
    delete _Cnp_tbf;
    delete _Cnr_tbf;

    delete _Cy_a20_tbf;
    delete _Cl_a20_tbf;
    delete _Cn_a20_tbf;
    delete _Cy_r30_tbf;
    delete _Cl_r30_tbf;
    delete _Cn_r30_tbf;

    delete _Cx_lef_tbf;
    delete _Cy_lef_tbf;
    delete _Cz_lef_tbf;
    delete _Cl_lef_tbf;
    delete _Cm_lef_tbf;
    delete _Cn_lef_tbf;

    delete _Cy_a20_lef_tbf;
    delete _Cl_a20_lef_tbf;
    delete _Cn_a20_lef_tbf;

    delete _delta_Cxq_lef_tbf;
    delete _delta_Cyp_lef_tbf;
    delete _delta_Cyr_lef_tbf;
    delete _delta_Czq_lef_tbf;
    delete _delta_Clp_lef_tbf;
    delete _delta_Clr_lef_tbf;
    delete _delta_Cmq_lef_tbf;
    delete _delta_Cnp_lef_tbf;
    delete _delta_Cnr_lef_tbf;

    delete _delta_Cnbeta_tbf;
    delete _delta_Clbeta_tbf;
    delete _delta_Cmbeta_tbf;

    delete _delta_Cm_ds_tbf;
    // delete _eta_el_tbf;
    delete _power_tbf;
}

double F16AeroData::_Cx(double alpha, double beta, double dele)
{
    return _Cx_tbf->GetVal(dele, beta, alpha);
}

double F16AeroData::_Cy(double alpha, double beta)
{
    return _Cy_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cz(double alpha, double beta, double dele)
{
    return _Cz_tbf->GetVal(dele, beta, alpha);
}

double F16AeroData::_Cl(double alpha, double beta, double dele)
{
    return _Cl_tbf->GetVal(dele, beta, alpha);
}

double F16AeroData::_Cm(double alpha, double beta, double dele)
{
    return _Cm_tbf->GetVal(dele, beta, alpha);
}

double F16AeroData::_Cn(double alpha, double beta, double dele)
{
    return _Cn_tbf->GetVal(dele, beta, alpha);
}

double F16AeroData::_Cxq(double alpha)
{
    return _Cxq_tbf->GetVal(alpha);
}

double F16AeroData::_Cyp(double alpha)
{
    return _Cyp_tbf->GetVal(alpha);
}

double F16AeroData::_Cyr(double alpha)
{
    return _Cyr_tbf->GetVal(alpha);
}

double F16AeroData::_Czq(double alpha)
{
    return _Czq_tbf->GetVal(alpha);
}

double F16AeroData::_Clp(double alpha)
{
    return _Clp_tbf->GetVal(alpha);
}

double F16AeroData::_Clr(double alpha)
{
    return _Clr_tbf->GetVal(alpha);
}

double F16AeroData::_Cmq(double alpha)
{
    return _Cmq_tbf->GetVal(alpha);
}

double F16AeroData::_Cnp(double alpha)
{
    return _Cnp_tbf->GetVal(alpha);
}

double F16AeroData::_Cnr(double alpha)
{
    return _Cnr_tbf->GetVal(alpha);
}

double F16AeroData::_Cy_a20(double alpha, double beta)
{
    return _Cy_a20_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cl_a20(double alpha, double beta)
{
    return _Cl_a20_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cn_a20(double alpha, double beta)
{
    return _Cn_a20_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cy_r30(double alpha, double beta)
{
    return _Cy_r30_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cl_r30(double alpha, double beta)
{
    return _Cl_r30_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cn_r30(double alpha, double beta)
{
    return _Cn_r30_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cx_lef(double alpha, double beta)
{
    return _Cx_lef_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cy_lef(double alpha, double beta)
{
    return _Cy_lef_tbf->GetVal( beta, alpha);
}

double F16AeroData::_Cz_lef(double alpha, double beta)
{
    return _Cz_lef_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cl_lef(double alpha, double beta)
{
    return _Cl_lef_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cm_lef(double alpha, double beta)
{
    return _Cm_lef_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cn_lef(double alpha, double beta)
{
    return _Cn_lef_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cy_a20_lef(double alpha, double beta)
{
    return _Cy_a20_lef_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cl_a20_lef(double alpha, double beta)
{
    return _Cl_a20_lef_tbf->GetVal(beta, alpha);
}

double F16AeroData::_Cn_a20_lef(double alpha, double beta)
{
    return _Cn_a20_lef_tbf->GetVal(beta, alpha);
}

double F16AeroData::_delta_Cxq_lef(double alpha)
{
    return _delta_Cxq_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Cyp_lef(double alpha)
{
    return _delta_Cyp_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Cyr_lef(double alpha)
{
    return _delta_Cyr_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Czq_lef(double alpha)
{
    return _delta_Czq_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Clp_lef(double alpha)
{
    return _delta_Clp_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Clr_lef(double alpha)
{
    return _delta_Clr_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Cmq_lef(double alpha)
{
    return _delta_Cmq_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Cnp_lef(double alpha)
{
    return _delta_Cnp_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Cnr_lef(double alpha)
{
    return _delta_Cnr_lef_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Cnbeta(double alpha)
{
    return _delta_Cnbeta_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Clbeta(double alpha)
{
    return _delta_Clbeta_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Cm(double alpha)
{
    return _delta_Cmbeta_tbf->GetVal(alpha);
}

double F16AeroData::_delta_Cm_ds(double dele, double alpha)
{
    return _delta_Cm_ds_tbf->GetVal(dele, alpha);
}

double F16AeroData::_eta_el(double dele)
{
	// return _eta_el_tbf->GetVal(dele);
	if(dele < 20)
		return 1;
	if(dele > 25)
		return 0.95;
	return 1 - 0.01 * (dele - 20);
}

double F16AeroData::_power(double ma)
{
    return _power_tbf->GetVal(ma);
}

double F16AeroData::_power2(double ma)
{
	static double maList[11] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	static double powerFrac[11] = {1.0, 0.98, 0.969, 0.921, 0.898, 0.867, 0.858, 0.856, 0.826, 0.703, 0.56};
	if(ma <= 0)	return powerFrac[0];
	else if(ma >= 1)	return powerFrac[10];
	else
	{
		int pos = 1;
		for(int i=0; i<11; i++)
		{
			if(ma < maList[i])
			{
				pos = i - 1;
				break;
			}
		}
		double frac = powerFrac[pos] + (powerFrac[pos + 1] - powerFrac[pos]) * (ma - maList[pos]) / (maList[pos + 1] - maList[pos]);
		return frac;
	}
}
