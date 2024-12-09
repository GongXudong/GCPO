#ifndef _F16AERODATA_H
#define _F16AERODATA_H

#include "../../tools/utils/datatblx.h"

class F16AeroData
{
public:
    F16AeroData(const char *fname);
    ~F16AeroData();

private:
    ResManger* libMng;
    
    DataTableExLib* _Cx_tbf;
    DataTableExLib* _Cy_tbf;
    DataTableExLib* _Cz_tbf;
    DataTableExLib* _Cl_tbf;
    DataTableExLib* _Cm_tbf;
    DataTableExLib* _Cn_tbf;
    
    DataTableExLib* _Cxq_tbf;
    DataTableExLib* _Cyp_tbf;
    DataTableExLib* _Cyr_tbf;
    DataTableExLib* _Czq_tbf;
    DataTableExLib* _Clp_tbf;
    DataTableExLib* _Clr_tbf;
    DataTableExLib* _Cmq_tbf;
    DataTableExLib* _Cnp_tbf;
    DataTableExLib* _Cnr_tbf;
    
    DataTableExLib* _Cy_a20_tbf;
    DataTableExLib* _Cl_a20_tbf;
    DataTableExLib* _Cn_a20_tbf;
    DataTableExLib* _Cy_r30_tbf;
    DataTableExLib* _Cl_r30_tbf;
    DataTableExLib* _Cn_r30_tbf;

    DataTableExLib* _Cx_lef_tbf;
    DataTableExLib* _Cz_lef_tbf;
    DataTableExLib* _Cm_lef_tbf;
    DataTableExLib* _Cy_lef_tbf;
    DataTableExLib* _Cn_lef_tbf;
    DataTableExLib* _Cl_lef_tbf;

    DataTableExLib* _Cy_a20_lef_tbf;
    DataTableExLib* _Cn_a20_lef_tbf;
    DataTableExLib* _Cl_a20_lef_tbf;

    DataTableExLib* _delta_Cxq_lef_tbf;
    DataTableExLib* _delta_Cyr_lef_tbf;
    DataTableExLib* _delta_Cyp_lef_tbf;
    DataTableExLib* _delta_Czq_lef_tbf;
    DataTableExLib* _delta_Clr_lef_tbf;
    DataTableExLib* _delta_Clp_lef_tbf;
    DataTableExLib* _delta_Cmq_lef_tbf;
    DataTableExLib* _delta_Cnr_lef_tbf;
    DataTableExLib* _delta_Cnp_lef_tbf;

    DataTableExLib* _delta_Cnbeta_tbf;
    DataTableExLib* _delta_Clbeta_tbf;
    DataTableExLib* _delta_Cmbeta_tbf;
    DataTableExLib* _delta_Cm_ds_tbf;
    DataTableExLib* _eta_el_tbf;
    DataTableExLib* _power_tbf;

public:
    void useLib(const char *fname);
    void unUseLib();
    void initTbf();
    void freeTbf();

    double _Cx(double alpha, double beta, double dele);
    double _Cy(double alpha, double beta);
    double _Cz(double alpha, double beta, double dele);
    double _Cl(double alpha, double beta, double dele);
    double _Cm(double alpha, double beta, double dele);
    double _Cn(double alpha, double beta, double dele);

    double _Cxq(double alpha);
    double _Cyp(double alpha);
    double _Cyr(double alpha);
    double _Czq(double alpha);
    double _Clp(double alpha);
    double _Clr(double alpha);
    double _Cmq(double alpha);
    double _Cnp(double alpha);
    double _Cnr(double alpha);

    double _Cy_a20(double alpha, double beta);
    double _Cl_a20(double alpha, double beta);
    double _Cn_a20(double alpha, double beta);
    double _Cy_r30(double alpha, double beta);
    double _Cl_r30(double alpha, double beta);
    double _Cn_r30(double alpha, double beta);

    double _Cx_lef(double alpha, double beta);
    double _Cz_lef(double alpha, double beta);
    double _Cm_lef(double alpha, double beta);
    double _Cy_lef(double alpha, double beta);
    double _Cn_lef(double alpha, double beta);
    double _Cl_lef(double alpha, double beta);

    double _Cy_a20_lef(double alpha, double beta);
    double _Cn_a20_lef(double alpha, double beta);
    double _Cl_a20_lef(double alpha, double beta);

    double _delta_Cxq_lef(double alpha);
    double _delta_Cyr_lef(double alpha);
    double _delta_Cyp_lef(double alpha);
    double _delta_Czq_lef(double alpha);
    double _delta_Clr_lef(double alpha);
    double _delta_Clp_lef(double alpha);
    double _delta_Cmq_lef(double alpha);
    double _delta_Cnr_lef(double alpha);
    double _delta_Cnp_lef(double alpha);

    double _delta_Cnbeta(double alpha);
    double _delta_Clbeta(double alpha);
    double _delta_Cm(double alpha);
    double _delta_Cm_ds(double alpha, double dele);

    double _eta_el(double dele);
    double _power(double ma);
    double _power2(double ma);
};

#endif
