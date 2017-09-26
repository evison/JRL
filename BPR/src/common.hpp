#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <sstream>
#include "gzstream.h"
#include "sys/time.h"
#include <cfloat>
#include <climits>
#include <unordered_set>
#include <random>

using namespace std;

bool pairCompare(const pair<int, double>& firstElem, const pair<int, double>& secondElem);
bool dimCompare (const pair<int, double*>& firstElem, const pair<int, double*>& secondElem);


/// Safely open a file
FILE* fopen_(const char* p, const char* m);

/// Data associated with a rating
typedef struct vote
{
	int user; // ID of the user
	int item; // ID of the item
	int label; // rated or not

	long long voteTime; // Unix time of the rating
} vote;

/// Data associated with a rating
typedef struct epoch
{
	int bin_from; 	// ID of the 1st bin
	int bin_to; 	// ID of the last bin
} epoch;

inline double inner(double* x, double* y, int K)
{
  //inner product
  //sum_^N_k x_k*y_k
	double res = 0;
	for (int k = 0; k < K; k ++) {
		res += x[k]*y[k];
	}
	return res;
}

inline double square(double x)
{
	return x*x;
}

inline double dsquare(double x)
{
	return 2*x;
}

inline double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

inline double clock_(void)
{
	timeval tim;
	gettimeofday(&tim, NULL);
	return tim.tv_sec + (tim.tv_usec / 1000000.0);
}

static inline string &ltrim(string &s)
{
	s.erase(s.begin(), find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace))));
	return s;
}

// trim from end
static inline string &rtrim(string &s)
{
	s.erase(find_if(s.rbegin(), s.rend(), not1(ptr_fun<int, int>(isspace))).base(), s.end());
	return s;
}

// trim from both ends
static inline string &trim(string &s)
{
	return ltrim(rtrim(s));
}