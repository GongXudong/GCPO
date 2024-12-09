
#include "testMng.h"

void testF16Plane()
{
	double h = 0;
	double IAS = 120;
	double v = IAS2TAS(IAS, h);
	F16Plane f16tmp(h, 220, 0.01, 1);
	f16tmp.setLog("logs/trace_f16.csv");
	// f16tmp.setCtrl(0, 0, -10, 60000);	// ail, ele, rud(deg), thrust(N)
	
	for(int i=0; i<2000; i++)
	{
		// f16tmp.setAltitude(h);
		// f16tmp.setAlpha(30);
		// f16tmp.setTheta(0);
		f16tmp.ctrlCmd.thrust = 10000000;
		// f16tmp.ctrlPos.thrust = 0;
		// f16tmp.ctrlCmd.ele = -5;
		f16tmp.step();
		// f16tmp.reset();
	}
}