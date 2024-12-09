/*
 * Content: simple version control content from other planes files
 * Author: Wang Hao
 * Created: 20210128
 * Last modified: 20210128
*/

#ifndef _CONTROL_H
#define _CONTROL_H

#include <stdio.h>

//  ***********************************************
//  *       (BS+C)                    b0+b1z(-1)  *
//  *       -------       -->        ----------   *
//  *       (S+A)                     1+a1z(-1)   *
//  ***********************************************
class OneOrder{
protected:
    double a1;
    double b0,b1;
    double u0,y0;
public:
    void init(double A, double B, double C, double T, int reset = 1);
    OneOrder()
    {
        a1=b0=b1=u0=y0=0;
    };
    OneOrder(double A, double B, double C, double T);
    double operator()(double Input);
    void setout(double val);
	void showSelf();
};

//  ***********************************************
//  *         K                       b0+b1z(-1)  *
//  *       -------       -->        ----------   *
//  *       (S+K)                     1+a1z(-1)   *
//  ***********************************************

class Lag:public OneOrder{
public:
    void init(double K, double T, int reset = 1){
        OneOrder::init(K, 0, K, T, reset);
    }
    Lag(double K, double T);
    Lag(){};
    void setout(double val);
};

//  ********************************************************
//  *    b0S*S+b1S+b2                  b0+b1z(-1)+b2z(-2)  *
//  *    ------------     -->         ------------------   *
//  *      S*S+a1S+a2                  1+a1z(-1)+a0z(-2)   *
//  ********************************************************
class TwoOrder{
protected :
    double a0,a1;
    double b0,b1,b2;
    double T;
    double u0,u1;
    double y0,y1;
public:
    void init(double A1, double A2, double B0, double B1, double B2, double T, int reset=1);
    TwoOrder(){;};
    TwoOrder(double A1, double A2, double B0, double B1, double B2, double t);
    double operator()(double Input);
};

//  ********************************************************
//  *          a2                      b0+b1z(-1)+b2z(-2)  *
//  *     ------------     -->         ------------------  *
//  *     S*S+a1S+a2                  1+a1z(-1)+a0z(-2)   *
//  ********************************************************
class TwoLag:public TwoOrder{
public:
    void init(double A1, double A2, double T, int reset = 1){
        TwoOrder::init(A1, A2, 0, 0, A2, T, reset);
    }
    TwoLag(){;};
    TwoLag(double A1, double A2, double T);
    void setout(double val);
};

#endif