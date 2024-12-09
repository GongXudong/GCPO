
#include "datatblx.h"

enum{
    TBL_TYPE_FLOAT = 1,
    TBL_TYPE_DOUBLE = 2
};

// float version
double step2(InterValTable_f *interValTablePtr, double *valuePtr)
{
    int i, sum;
    float x1, x2;
    double xPlane[2];
    
    short d = interValTablePtr->dimension;
    short *ix = interValTablePtr->nodePtr;
    float *xPtr = interValTablePtr->axisPtr[0];
    InterValTable_f temp;

    x1 = xPtr[ix[0]-1];
    x2 = xPtr[ix[0]];
    if(d == 1)
    {
        xPlane[0] = interValTablePtr->tablePtr[ix[0]-1];
        xPlane[1] = interValTablePtr->tablePtr[ix[0]];
        return xPlane[0] + (valuePtr[0] - x1) * (xPlane[1] - xPlane[0]) / (x2 - x1);
    }
    if(d > 1)
    {
        temp.dimension = d - 1;
        temp.dimPtr = interValTablePtr->dimPtr + 1;
        temp.nodePtr = interValTablePtr->nodePtr + 1;
        temp.axisPtr = interValTablePtr->axisPtr + 1;
        for(i=0, sum=1; i<d-1; i++)
        {
            sum *= temp.dimPtr[i];
        }
        for(i=0; i<2; i++)
        {
            temp.tablePtr = interValTablePtr->tablePtr + (ix[0] - i) * sum;
            xPlane[1-i] = step2(&temp, valuePtr + 1);
        }
        return xPlane[0] + (valuePtr[0] - x1) * (xPlane[1] - xPlane[0]) / (x2 - x1);
    }
    return 0.0;
}

// double version
double step2_d(InterValTable_d *interValTablePtr, double *valuePtr)
{
    short i, sum;
    double x1, x2;
    double xPlane[2];
    
    short d = interValTablePtr->dimension;
    short *ix = interValTablePtr->nodePtr;
    double *xPtr = interValTablePtr->axisPtr[0];
    InterValTable_d temp;

    x1 = xPtr[ix[0] - 1];
    x2 = xPtr[ix[0]];

    if(d == 1)
    {
        xPlane[0] = interValTablePtr->tablePtr[ix[0] - 1];
        xPlane[1] = interValTablePtr->tablePtr[ix[0]];
        return xPlane[0] + (valuePtr[0] - x1) * (xPlane[1] - xPlane[0]) / (x2 - x1);
    }
    if(d > 1)
    {
        temp.dimension = d - 1;
        temp.dimPtr = interValTablePtr->dimPtr + 1;
        temp.nodePtr = interValTablePtr->nodePtr + 1;
        temp.axisPtr = interValTablePtr->axisPtr + 1;
        for(i=0,sum=1; i<d-1; i++)
        {
            sum *= temp.dimPtr[i];
        }
        for(i=0; i<2; i++)
        {
            temp.tablePtr = interValTablePtr->tablePtr + (ix[0] - i) * sum;
            xPlane[1-i] = step2_d(&temp, valuePtr + 1);
        }
        return xPlane[0] + (valuePtr[0] - x1) * (xPlane[1] - xPlane[0]) / (x2 - x1);
    }
    return 0.0;
}

// xAxisPtr: an T array pointer contains data in increasing mode
// nx is the size of the array
// function finds an index of a value that is greater than xValue, and the value of index-1 is less than xValue
template<class _T>
short find_node_x(_T *xAxisPtr, short nx, double xValue)
{
    short mid;
    short left = 0;
    short right = nx-1;
    _T *headPtr = xAxisPtr;

    if(xValue <= headPtr[left])
        return left + 1;
    else if(xValue >= headPtr[right])
        return right;

    while(right - left > 1)
    {
        mid = (left + right) / 2;
        if(xValue >= headPtr[mid])
        {
            left = mid;
        }
        else
        {
            right = mid;
        }
    }
    return right;
}

template<class InterValTable, class _T>
double inter_value(InterValTable *interValTablePtr, double *valuePtr)
{
    double result;
    int dataTypeSize = sizeof(_T);
    short d = interValTablePtr->dimension - 1;

    while(d != -1)
    {
        interValTablePtr->nodePtr[d] = find_node_x(interValTablePtr->axisPtr[d], interValTablePtr->dimPtr[d], valuePtr[d]);
        if(valuePtr[d] < interValTablePtr->axisPtr[d][0])
        {
            valuePtr[d] = interValTablePtr->axisPtr[d][0];
        }
        else if(valuePtr[d] > interValTablePtr->axisPtr[d][interValTablePtr->dimPtr[d] - 1])
        {
            valuePtr[d] = interValTablePtr->axisPtr[d][interValTablePtr->dimPtr[d] - 1];
        }
        d--;
    }
    if(dataTypeSize == 4)
        result = step2((InterValTable_f*)interValTablePtr, valuePtr);
    else
        result = step2_d((InterValTable_d*)interValTablePtr, valuePtr);
    return result;
}

// Data Table Class is designed by Zhu Xueyao, 1996.2.1
// tbf file can be used
void DataTable::init(char *fname){
    FILE* fin;
    fin = openfile(fname);
    loadfile(fin, fname);
    fclose(fin);
}

// open data file fname
FILE* DataTable::openfile(char *fname){
    FILE* fin = NULL;
	errno_t err;
    err = fopen_s(&fin, fname, "rb");
    if(err != 0)
    {
        printf( "can not open %s\n", fname);
        dimension = 0;
        exit(1);
    }
    return fin;
}

// load data from fin
// version means the data type in fin, 0 is float, 2 is double
// size_type is the storage size of one data, double is 8 bytes, float is 4 bytes
// dimension is the dimension number of the data table
// tableSize is the numbers size of data table
// size_table is the storage size of data table
// bptsize is the sum of every axis numbers size
// nodePtr is used for interpolation function, find_node_x
void DataTable::loadfile(FILE *fin, char *fname){
    int size1, bptsize;
    int size_type, size_table;
    char mark[5];
    mark[4] = '\0';
    unsigned char *ptr;

    short i;
    short ibuf[10];
    short version, commentlen;
    short* DimPtr;
    short* NodePtr;
    unsigned char** AxisPtr;
    unsigned char* TablePtr;

    fread(mark, sizeof(char), 4, fin);
    if(strncmp(mark, "TDBF", 4) != 0){
        printf("No TDBF mark, %s is not DataTable file\n", fname);
        dimension = 0;
        exit(1);
    }

    fread(&version, sizeof(short), 1, fin);
    fread(&commentlen, sizeof(short), 1, fin);
    fread(&dimension, sizeof(short), 1, fin);
    fread(ibuf, sizeof(short), dimension, fin);

    for(i = 0, tableSize = 1; i < dimension; i++)
        tableSize *= ibuf[i];
    for(i = 0, bptsize = 0; i < dimension; i++)
        bptsize += ibuf[i];

    size_type = (version==TBL_TYPE_DOUBLE)?sizeof(double):sizeof(float);
    size_table = (tableSize + bptsize) * size_type;
    size1 = (sizeof(float *) + sizeof(short)*2) * dimension + size_table;
    ptr = (unsigned char*)malloc(size1);
    
    if(ptr == NULL){
        printf("can not allocate memory in %s \n",fname);
        dimension = 0,
        exit(1);
    }

    // Data Struct
    //      Ptr         len
    //      TablePtr    tableSize*sizeof(float)
    //                  AxisPtr
    //                  DimPtr
    //                  NodePtr

    TablePtr = (unsigned char *)ptr;
    AxisPtr = (unsigned char **)(ptr + size_table);
    DimPtr = (short *)(ptr + sizeof(float *) * dimension + size_table);
    NodePtr = (short *)(ptr + (sizeof(float *) + sizeof(short)) * dimension + size_table);

    for(i = 0; i < dimension; i++)
       DimPtr[i] = ibuf[i];

    AxisPtr[0] = TablePtr + tableSize*size_type;
    for(i = 1;i < dimension; i++)
        AxisPtr[i] = AxisPtr[i-1] + DimPtr[i-1]*size_type;

    fread(TablePtr, size_type, tableSize, fin);
    fread(AxisPtr[0], size_type, bptsize, fin);

    interValTable.type = version;
    interValTable.dimension = dimension;
    interValTable.nodePtr = NodePtr;
    interValTable.dimPtr = DimPtr;
    interValTable.axisPtr = AxisPtr;
    interValTable.tablePtr = TablePtr;
    //  printf("%s  free : %lu bytes\n", fname,farcoreleft());
}

DataTable::~DataTable()
{
    if(dimension != 0)
    {
        dimension = 0;
    }
    freeData();
}

void DataTable::freeData()
{
    free(interValTable.tablePtr);
}

double DataTable::GetVal(double x){
    if(dimension != 1)
    {
        printf("Dimension Error\n");
        exit(1);
    }
    if(interValTable.type == TBL_TYPE_DOUBLE)
    {
        return inter_value<InterValTable_d, double>((InterValTable_d*)&interValTable, &x);
    }
    else
    {
        return inter_value<InterValTable_f, float>((InterValTable_f*)&interValTable, &x);
    }
}

double DataTable::GetVal(double x1,double x2,...){
    double _x[10];
    va_list ap;

    if(dimension < 2)
    {
        printf("Dimension Error in interval model\n");
        exit(1);
    }

    _x[0] = x1;
    _x[1] = x2;

    if(dimension > 2)
    {
        va_start(ap, x2);
        for(short i = 2; i < dimension; i++)
        {
            _x[i] = va_arg(ap, double);
        }
        va_end(ap);
    }
    if(interValTable.type == TBL_TYPE_DOUBLE)
        return inter_value<InterValTable_d, double>((InterValTable_d*)&interValTable, _x);
    else
        return inter_value<InterValTable_f, float>((InterValTable_f*)&interValTable, _x);
}

int DataTable::getTableSize()
{
    return tableSize;
}

int DataTable::getAxisSize(int axisNum)
{
    axisNum = checkDimension(axisNum);
    return (int)(interValTable.dimPtr[axisNum]);
}

double* DataTable::getTable_d()
{
    if(interValTable.type == TBL_TYPE_DOUBLE)
        return (double*)(interValTable.tablePtr);
    printf("table data type is float %d\n", interValTable.type);
    return NULL;
}

float* DataTable::getTable_f()
{
    if(interValTable.type == TBL_TYPE_FLOAT)
        return (float*)(interValTable.tablePtr);
    printf("table data type is double %d\n", interValTable.type);
    return NULL;
}

double* DataTable::getAxisPtr_d(int axisNum)
{
    axisNum = checkDimension(axisNum);
    if(interValTable.type != TBL_TYPE_DOUBLE)
    {
        printf("table data type is float %d\n", interValTable.type);
        return NULL;
    }
    return (double*)(interValTable.axisPtr[axisNum]);
}

float* DataTable::getAxisPtr_f(int axisNum)
{
    axisNum = checkDimension(axisNum);
    if(interValTable.type != TBL_TYPE_FLOAT)
    {
        printf("table data type is double %d\n", interValTable.type);
        return NULL;
    }
    return (float*)(interValTable.axisPtr[axisNum]);
}

int DataTable::checkDimension(int axisNum)
{
    if(axisNum >= interValTable.dimension)
    {
        printf("axis number is larger than data dimension!\n");
        axisNum = 0;
    }
    return axisNum;
}

ResManger::ResManger()
{
    ResCount = 0;
    ResTBLPtr = NULL;
    fd = NULL;
    libsPath = "";
}

void *ResManger::LoadResource(char *fname)
{
    char* buf;
    long lsize;
    FILE *fp;
    fp = FindResource(fname, &lsize);
    if(fp == NULL) 
    {
        return NULL;
    }
    buf = (char*)malloc(lsize);
    if(buf == NULL)
    {
        return NULL;
    }
    fread(buf, lsize, 1, fp);
    return buf;
}

void *ResManger::LoadResource(int i)
{
    char* buf;
    long lsize;
    FILE *fp;
    fp = FindResource(i, &lsize);
    if(fp == NULL)
    {
        return NULL;
    }
    buf = (char*)malloc(lsize);
    if(buf == NULL)
    {
        return NULL;
    }
    fread(buf, lsize, 1, fp);
    return buf;
}


int ResManger::Extract(char *fname)
{
    long lsize;
    FILE *fp, *fout;
    short i;
    char *buf;
    const short maxlen = 30000;
    if(fname != NULL)
    {
        fp = FindResource(fname, &lsize);
        if(fp == NULL)
        {
            return FALSE;
        }
        if(lsize > maxlen)
        {
            printf("too large size");
            return FALSE;
        }
        buf=(char *)malloc(lsize);
        if(buf == NULL)
        {
            printf("can not alloc memory");
            return FALSE;
        }
        fread(buf, lsize, 1, fp);
        fopen_s(&fout, fname, "wb");
        fwrite(buf, lsize, 1, fout);
        fclose(fout);
        free(buf);
        return TRUE;
    }
    printf("resource = %d\n", ResCount);
    buf=(char *)malloc(maxlen);

    for(i=0; i<ResCount; i++)
    {
        char str[20];
        fp = FindResource(ResTBLPtr[i].name, &lsize);
        if(lsize > maxlen)
        {
            printf("too large size:%ld\n", lsize);
            return FALSE;
        }
        strncpy_s(str, ResTBLPtr[i].name, 13);
        fread(buf, lsize, 1, fp);
        fopen_s(&fout, str, "wb");
        fwrite(buf, lsize, 1, fout);
        fclose(fout);
    }
    free(buf);
    return TRUE;
}

ResManger::~ResManger()
{
    UnUse();
}

void ResManger::UnUse()
{
    if(ResTBLPtr!=NULL)
    {
        delete ResTBLPtr;
        fclose(fd);
        fd = NULL;
        ResTBLPtr = NULL;
        ResCount = 0;
        ResFname[0] = 0;
    }
}

void ResManger::setPath(string path)
{
    libsPath = path;
}

int ResManger::Use(char *fname)
{
    UnUse();
    strcpy_s(ResFname, libsPath.c_str());
    strcat_s(ResFname, fname);

    if(fd!=NULL)
    {
        fclose(fd);
    }
    fopen_s(&fd, ResFname, "rb");
    if(fd == NULL)
    {
        printf("open error: %s\n", ResFname);
        ErrorCode = -1;
    }
    else
    {
        ErrorCode = 0;
    }
    ResTBLPtr = load_res_table();
    return ErrorCode;
}

ResManger::ResManger(char *fname){
    Use(fname);
}

Record *ResManger::load_res_table(void)//,char *name)
{
    short i, count;
    Record *r;
    Record64 *r32;
    int pos, flen;
    ResCount = 0;
    fseek(fd, -2L, SEEK_END);
    flen = ftell(fd) + 2;
    fread(&count, sizeof(short), 1, fd);
    
    r = (Record *)malloc(sizeof(Record)*count);
    r32 = (Record64 *)malloc(sizeof(Record64)*count);
    if((r==NULL)||(r32==NULL)) return NULL;
    
    pos = flen - (int)(2L + sizeof(Record64)*count);
    fseek(fd, pos, SEEK_SET);
    fread(r32, sizeof(Record64), count, fd);
    ResCount = count;
    
    for(i=0; i<count; i++)
    {
        memset(&r[i], 0, sizeof(r[i]));
        r[i].len = r32[i].len;
        for(int j=0; j<12; j++)
        {
            r[i].name[j] = r32[i].name[j];
        }
    }
    free(r32);
    return r;
}

FILE *ResManger::FindResource(int i,long *size)
{
    short num;
    long pos;
    if(i<0 || i>ResCount-1) return NULL;
    num = i;
    pos = 0;
    *size = ResTBLPtr[i].len;

    for(i=num;i<ResCount;i++)
    {
        pos += ResTBLPtr[i].len;
    }
    pos = (2L + sizeof(Record64)*ResCount + pos);
    pos =- pos;
    fseek(fd, pos, SEEK_END);
    return fd;
}

FILE *ResManger::FindResource(char *resname,long *size)
{
    short i,num;
    long pos;
    for(i=0;i<ResCount;i++)
    {
        if(_stricmp(resname, ResTBLPtr[i].name)==0) break;
    }
    if(i == ResCount) return NULL;
    num = i;
    pos = 0;
    *size = ResTBLPtr[i].len;

    for(i=num;i<ResCount;i++)
    {
        pos += ResTBLPtr[i].len;
    }
    fseek(fd,-(long)(2L + sizeof(Record64)*ResCount + pos), SEEK_END);
    return fd;
}

ResManger LibManger;
void DataTableEx::init(const char *fname)
{
    FILE *fp;
    long size;
    char str[16];
    fp = LibManger.FindResource((char*)fname, &size);
    if(fp==NULL)
    {
        strncpy_s(str, fname, 8);
        str[8] = 0;
        strcat_s(str,".TBF");
        fp = LibManger.FindResource(str, &size);
    }
    if(fp==NULL)
    {
        printf("Can not find file %s\n",str);
        exit(1);
    }
    loadfile(fp, (char*)fname);
};

void DataTableExLib::init(const char *fname, ResManger *libMng)
{
    FILE *fp;
    long size;
    char str[16];
    fp = libMng->FindResource((char*)fname, &size);
    if(fp==NULL)
    {
        strncpy_s(str, fname, 8);
        str[8] = 0;
        strcat_s(str,".TBF");
        fp = libMng->FindResource(str, &size);
    }
    if(fp==NULL)
    {
        printf("Can not find file %s\n",str);
        exit(1);
    }
    loadfile(fp, (char*)fname);
};
