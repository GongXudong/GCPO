#include "control.h"


//OneOrder begin
OneOrder::OneOrder(double A, double B, double C, double T){
    init(A, B, C, T, 1);
}

void OneOrder::setout(double val){
    y0 = val;
    u0 = 0;
}

void OneOrder::init(double A, double B, double C, double T, int reset){
    if(reset == 1)
    {
        u0 = y0 = 0.0;
    }
    a1 = (T*A-2)/(2.0+T*A);
    b0 = (2.0*B+T*C)/(2.0+T*A);
    b1 = (T*C-2.0*B)/(2.0+T*A);
}

double OneOrder::operator()(double InputValue){
    double y;
    y = -a1*y0 + b0*InputValue + b1*u0;
    u0 = InputValue;
    y0 = y;
    return y;
}

void OneOrder::showSelf(){
    printf("u0: %f, y0: %f\n", u0, y0);
}

//Lag begin
Lag::Lag(double K,double T):OneOrder(K,0,K,T){;}

void Lag::setout(double val)
{
    y0 = val;
    u0 = val;
}

//TwoOrder begin
void TwoOrder::init(double A1,double A2,double B0,double B1,double B2,double T,int reset){
    double A,C;
    if(reset == 1)
    {
        u0=y0=u1=y1=0;
    }
    C = 2/T;
    A = A2 + A1*C + C*C;
    b2 = (B2 + B1*C + B0*C*C) / A;
    b1 = (2*B2 - 2*B0*C*C) / A;
    b0 = (B2 - B1*C+B0*C*C) / A;
    a1 = (2*A2 - 2*C*C) / A;
    a0 = (A2 - A1*C + C*C) / A;
}

TwoOrder::TwoOrder(double A1,double A2,double B0,double B1,double B2,double T){
    init(A1,A2,B0,B1,B2,T,1);
}

double TwoOrder::operator ()(double InputValue){
    double y;
    y=-a1*y1-a0*y0+b2*InputValue+b1*u1+b0*u0;
    u0=u1;
    u1=InputValue;
    y0=y1;
    y1=y;
    return y;
}

//TwoLag begin
TwoLag::TwoLag(double A1, double A2, double T):TwoOrder(A1,A2,0,0,A2,T){;}

void TwoLag::setout(double val)
{
    y0 = y1 = u0 = u1 = val;
}