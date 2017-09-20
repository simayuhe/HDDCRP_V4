//#include "StdAfx.h"
#include "rand_utils.h"


int rand_mult_0(vector<double> & pi) 
{
	double sum = 0.0;
	vector<double>::iterator it;
	for (it = pi.begin(); it != pi.end(); it++)
	{
		sum += *it;
	}
	double mass = genrand_real2() * sum;
	for (it = pi.begin(); it != pi.end(); it++)
	{
		mass -= *it;
		if ( mass <= 0.00000000000 ) 
		{
			break;
		}
	}
	return( it - pi.begin() );
}

int rand_mult_1(vector<double> & pi) 
{
	//double sum = pi.back();
	double mass = genrand_real2() * pi.back();
	vector<double>::iterator it;
	vector<double>::iterator end = pi.end() - 1;
	for (it = pi.begin(); it != end; it++)
	{
		mass -= *it;
		if ( mass <= 0.00000000000 ) 
		{
			break;
		}
	}
	return( it - pi.begin() );
}

int rand_mult_1(vector<int> & pi) 
{
	//double sum = pi.back();
	double mass = genrand_real2() * (double)(pi.back());
	vector<int>::iterator it;
	vector<int>::iterator end = pi.end() - 1;
	for (it = pi.begin(); it != end; it++)
	{
		mass -= (double)(*it);
		if ( mass <= 0.00000000000 ) 
		{
			break;
		}
	}
	return( it - pi.begin() );
}
