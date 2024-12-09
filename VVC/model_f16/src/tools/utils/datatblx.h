#ifndef __DATATBLX_H
#define __DATATBLX_H 1

#include <stdio.h>
#include <stdarg.h>
#include <string>
#include <string.h>
#include <stdlib.h>

#ifdef __unix
#define fopen_s(pFile, filename, mode) ((*(pFile)) = fopen((filename), (mode))) == NULL
#define errno_t int
#define strncpy_s strncpy
#define strcpy_s strcpy
#define strcat_s strcat
#define _stricmp strcasecmp
#endif

#ifndef TRUE
#define TRUE  1
#define FALSE 0
#endif

using namespace std;

typedef struct {
    long len;
    char name[12];
    char *buf;
} Record;

// 64bit
typedef struct{
    int len;
    char name[12];
    int space;
} Record64;

// Data Table inter data stuct
typedef struct{
    int type;
    short dimension;
    short *dimPtr;
    unsigned char **axisPtr;
    unsigned char *tablePtr;
    short *nodePtr;
} InterValTable;

typedef struct{
    int type;
    short dimension;
    short *dimPtr;
    float **axisPtr;
    float *tablePtr;
    short *nodePtr;
    int f;
} InterValTable_f;

typedef struct{
    int type;
    short dimension;
    short *dimPtr;
    double **axisPtr;
    double *tablePtr;
    short *nodePtr;
    int d;
}  InterValTable_d;


// lib read class
// ResManger is designed by Zhu Xue Yao
// 1st  1995.6.1
// 2nd  1996.2.1
class ResManger
{
private:
    FILE* fd;   //file handle
    string libsPath;
    char ResFname[256];
    Record* load_res_table(void);

public:
    short ResCount;
    short ErrorCode;
    Record *ResTBLPtr;

public:
    ResManger();
    ResManger(char *fname);
    ~ResManger();

    FILE* FindResource(char *resname,long*size);
    FILE* FindResource(int i,long*size);

    int Use(char *fname);
    void UnUse(void);
    int Extract(char *fname);
    int Extract(int i);
    int GetResCount(void){return ResCount;};

    void setPath(string path="");
    void *LoadResource(char *resname);
    void *LoadResource(int i);
};

extern ResManger LibManger;

// DataTable Class, tbf file can be used
class DataTable
{
private:
    FILE *openfile(char *fname);

protected:
    void loadfile(FILE *fin, char *fname);

public:
    InterValTable interValTable;
    short dimension;
    int tableSize;

public:
    DataTable(void){dimension=0;};
    DataTable(char *fname){init(fname);};
    ~DataTable();

    void init(char *fname);
    void freeData();
    
    double GetVal(double x1);
    double GetVal(double x1,double x2,...);
    double operator()(double x){return GetVal(x);};
    int getTableSize();
    double* getTable_d();
    float* getTable_f();
    int checkDimension(int axisNum);
    int getAxisSize(int axisNum);
    double* getAxisPtr_d(int axisNum);
    float* getAxisPtr_f(int axisNum);
};

// DataTable Class, lib file can be used
class DataTableEx:public DataTable
{
public:
    DataTableEx(const char *fname){init(fname);};
    void init(const char *fname);
};

// DataTable Class, lib file can be used
class DataTableExLib:public DataTable
{
public:
    DataTableExLib(const char *fname, ResManger *libMng){init(fname, libMng);};
    void init(const char *fname, ResManger *libMng);
};

#endif