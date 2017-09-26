#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include "gzstream.h"
#include <cfloat>
#include <unordered_set>
#include <unordered_map>
#include <ctype.h>
#include "sys/time.h"
#include <climits>
#include <cmath>

using namespace std;


unordered_set<string> women, men, boys, girls, baby;

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

static inline std::string &ltrim(std::string &s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
  return ltrim(rtrim(s));
}

/// To sort words by frequency in a corpus
bool wordCountCompare(pair<string, int> p1, pair<string, int> p2)
{
  return p1.second > p2.second;
}

/// Parse category info for all products
void loadCategories(char* categoryPath, unordered_map<string,string>& itemCate)
{
  fprintf(stderr, "Loading category data ");
  igzstream in;
  in.open(categoryPath);
  if (! in.good()) {
    fprintf(stderr, "\n  Can't load category from %s.\n", categoryPath);
    exit(1);
  }

  string line;

  string currentProduct = "NULL";
  int count = 0;
  while (getline(in, line)) {
    istringstream ss(line);

    if (line.c_str()[0] != ' ') {
      double price = -1;
      string brand("unknown_brand");
      ss >> currentProduct >> price >> brand;

      count ++;
      if (not (count % 100000)) {
        fprintf(stderr, ".");
        fflush(stderr);
      }
      continue;
    }

    vector<string> category;
    string cat;
    while (getline(ss, cat, ',')) {
      category.push_back(trim(cat));
    }

    if (category.size() < 2 || category[0] != "Clothing Shoes & Jewelry") {
      continue;
    }

    if (category[1] == "Men") men.insert(currentProduct);
    else if (category[1] == "Women") women.insert(currentProduct);
    else if (category[1] == "Boys") boys.insert(currentProduct);
    else if (category[1] == "Girls") girls.insert(currentProduct);
    else if (category[1] == "Baby") baby.insert(currentProduct);
  }

  in.close();
  printf("\nLoaded.\n");
}


int main(int argc, char** argv)
{
  if (argc != 3)
  {
    printf("Files required are:\n");
    printf("  1: INPUT: Clothing image feature file\n");
    printf("  2: INPUT: Category file path\n");
    exit(1);
  }
  char* imgFeatPath = argv[1];
  char* catePath = argv[2];
  int imgFeatDim = 4096; 

  unordered_map<string,string> itemCate;
  loadCategories(catePath, itemCate);


  ofstream of_Men("image_features_Men.b", ios::binary);
  if (!of_Men.is_open()) {
    printf("Can't open image_features_Men.b for writing.\n");
    exit(1);
  }

  ofstream of_Women("image_features_Women.b", ios::binary);
  if (!of_Women.is_open()) {
    printf("Can't open image_features_Women.b for writing.\n");
    exit(1);
  }

  ofstream of_Boys("image_features_Boys.b", ios::binary);
  if (!of_Boys.is_open()) {
    printf("Can't open image_features_Boys.b for writing.\n");
    exit(1);
  }

  ofstream of_Girls("image_features_Girls.b", ios::binary);
  if (!of_Girls.is_open()) {
    printf("Can't open image_features_Girls.b for writing.\n");
    exit(1);
  }

  ofstream of_Baby("image_features_Baby.b", ios::binary);
  if (!of_Baby.is_open()) {
    printf("Can't open image_features_Baby.b for writing.\n");
    exit(1);
  }

  FILE* f = fopen_(imgFeatPath, "rb");
  int a;
  char asin[10];
  float* feat = new float[imgFeatDim];


  int counter = 0;
  while (!feof(f)) {
    if ((a = fread(asin, sizeof(*asin), 10, f)) != 10) {
      printf("Expected to read %d chars, got %d\n", 10, a);
      continue;
    }

    for (int c = 0; c < 10; c ++) {
      if (not isascii(asin[c])) {
        printf("Expected asin to be 10-digit ascii\n");
        exit(1);
      }
    }

    string sAsin(asin);

    if ((a = fread(feat, sizeof(*feat), imgFeatDim, f)) != imgFeatDim) {
      printf("Expected to read %d floats, got %d\n", imgFeatDim, a);
      exit(1);
    }

    if (men.find(sAsin) != men.end()) {
      of_Men.write(sAsin.c_str(), 10);
      for (int j = 0; j < imgFeatDim; j ++) {
        of_Men.write((char*) &(feat[j]), sizeof(float));
      }
    }
    if (women.find(sAsin) != women.end()) {
      of_Women.write(sAsin.c_str(), 10);
      for (int j = 0; j < imgFeatDim; j ++) {
        of_Women.write((char*) &(feat[j]), sizeof(float));
      }
    } 
    if (boys.find(sAsin) != boys.end()) {
      of_Boys.write(sAsin.c_str(), 10);
      for (int j = 0; j < imgFeatDim; j ++) {
        of_Boys.write((char*) &(feat[j]), sizeof(float));
      }
    } 
    if (girls.find(sAsin) != girls.end()) {
      of_Girls.write(sAsin.c_str(), 10);
      for (int j = 0; j < imgFeatDim; j ++) {
        of_Girls.write((char*) &(feat[j]), sizeof(float));
      }
    } 
    if (baby.find(sAsin) != baby.end()) {
      of_Baby.write(sAsin.c_str(), 10);
      for (int j = 0; j < imgFeatDim; j ++) {
        of_Baby.write((char*) &(feat[j]), sizeof(float));
      }
    }

    counter ++;
    if (counter % 10000 == 0) {
        fprintf(stderr, ".");
        fflush(stderr);
    }
  }

  fclose(f);
  delete [] feat;

  of_Men.close();
  of_Women.close();
  of_Boys.close();
  of_Girls.close();
  of_Baby.close();

  printf("\n  Congrats. All written successfully.\n");
}

