//#include "StdAfx.h"
#include "string.h"
#include "Multinomial.h"
//#include "Tri_Mult.h"
#include <vector>
#include <string>
using namespace std;

#ifdef TRI_MULT_DIST
typedef Tri_Mult DIST;
typedef HH_Tri_Mult HH; // type for prior parameters
typedef SS_Tri_Mult SS; // type for prior parameters
typedef QQ_Tri_Mult QQ; // type for prior parameters
typedef Stat_Tri_Mult STAT;
#else
typedef Multinomial DIST;
typedef vector<double> HH; // type for prior parameters
typedef int SS; // type for an obseration
typedef vector<int> QQ; // type for sufficient statistic充分统计量
typedef Stat_Mult STAT;
#endif
