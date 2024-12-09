CC          = gcc 
CCPP        = g++
CCFLAGS     = -Wno-strict-aliasing
LDFLAGS     = $(CCFLAGS) -L.

HOST_COMPILER   ?= g++

INCLUDE         = -I$(DIR_SRC) -I$(DIR_SRC)tools/utils/ -I$(DIR_SRC)tools/linuxOS/
INCLUDE_DATA    = -I$(DIR_SRC)tools/data/

RM          = rm -f 
MAKE        = make

DIR_SRC     = src/
DIR_MAIN    = main/
DIR_BIN     = bin/
DIR_LIB     = libs/

SRC_BASE        = $(wildcard $(DIR_SRC)tools/utils/*.cpp)
OBJ_BASE        = $(SRC_BASE:.cpp=.o)
SRC_F16         = $(wildcard $(DIR_SRC)model/f16/*.cpp)
OBJ_F16         = $(SRC_F16:.cpp=.o)
SRC_DYN         = $(wildcard $(DIR_SRC)model/planeDyn/*.cpp)
OBJ_DYN         = $(SRC_DYN:.cpp=.o)

TARGET      = f16model#f16model#runf16

all: 
	$(MAKE) $(TARGET)
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:.

runf16:$(OBJ_BASE) $(SRC_F16) $(SRC_DYN) $(DIR_MAIN)f16simple/main.cpp $(DIR_MAIN)f16simple/testMng.cpp
	$(CCPP) $(LDFLAGS) -o $(DIR_BIN)f16simple/$@ $^

f16model:$(SRC_BASE) $(SRC_F16) $(SRC_DYN) $(DIR_MAIN)f16dll/generalPlane.cpp
	$(CCPP) $(LDFLAGS) -fPIC -shared -o $(DIR_BIN)f16simple/$@.so $^

%.o: %.cpp
	$(CC) -o $@ -c $^ $(CCFLAGS)

%.o: %.c
	$(CC) -o $@ -c $^ $(CCFLAGS)

clean:
	$(RM) $(OBJ_BASE) $(OBJ_F16) $(OBJ_DYN)

distclean:
	$(MAKE) clean 
	$(RM) $(TARGET)
