#include "common.hpp"

/// Safely open a file
FILE* fopen_(const char* p, const char* m)
{
	FILE* f = fopen(p, m);
	if (!f) {
		printf("Failed to open %s\n", p);
		exit(1);
	}
	return f;
}

bool pairCompare(const pair<int, double>& firstElem, const pair<int, double>& secondElem) 
{
	return firstElem.second > secondElem.second;
}

bool dimCompare(const pair<int, double*>& firstElem, const pair<int, double*>& secondElem) 
{
	return *(firstElem.second) > *(secondElem.second);
}