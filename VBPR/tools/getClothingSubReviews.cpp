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
  unordered_set<string> subcat = {"Women", "Men", "Girls", "Boys", "Baby"};

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

    if (category.size() < 2
      || category[0] != "Clothing Shoes & Jewelry" 
      || subcat.find(category[1]) == subcat.end()) {
      continue;
    }
    
    itemCate[currentProduct] = category[1];
  }

  in.close();
  printf("\n  Loaded.\n");
}


int main(int argc, char** argv)
{
  if (argc != 3)
  {
    printf("Files required are:\n");
    printf("  1: INPUT: Review file (old format)\n");
    printf("  2: INPUT: Category file path\n");
    exit(1);
  }
  char* reviewPath = argv[1]; //review blobs
  char* catePath = argv[2]; // what is this???

  unordered_map<string,string> itemCate;
  loadCategories(catePath, itemCate);

  string uName; // User name
  string bName; // Item name
  float value; // Rating
  int voteTime; // Time rating was entered
  int nw; // Number of words
  int nRead = 0; // Progress
  string line;

  igzstream in;
  in.open(reviewPath);
  if (!in.good()) {
    printf("review file read error!");
    exit(1);
  }

  ofstream of_Men;
  of_Men.open ("reviews_Men.txt");
  if (!of_Men.is_open()) {
    printf("Can't open reviews_Men.txt for writing.\n");
    exit(1);
  }

  ofstream of_Women;
  of_Women.open ("reviews_Women.txt");
  if (!of_Women.is_open()) {
    printf("Can't open reviews_Women.txt for writing.\n");
    exit(1);
  }

  ofstream of_Boys;
  of_Boys.open ("reviews_Boys.txt");
  if (!of_Boys.is_open()) {
    printf("Can't open reviews_Boys.txt for writing.\n");
    exit(1);
  }

  ofstream of_Girls;
  of_Girls.open ("reviews_Girls.txt");
  if (!of_Girls.is_open()) {
    printf("Can't open reviews_Girls.txt for writing.\n");
    exit(1);
  }

  ofstream of_Baby;
  of_Baby.open ("reviews_Baby.txt");
  if (!of_Baby.is_open()) {
    printf("Can't open reviews_Baby.txt for writing.\n");
    exit(1);
  }

  int lineCount = 0;
  fprintf(stderr, "\n  Reading from the Clothing full set ");
  while (getline(in, line)) {
    stringstream ss(line);
    ss >> uName >> bName >> value >> voteTime >> nw;

    if (value > 5 or value < 0 or bName.size() != 10) { // Ratings should be in the range [0,5]
      printf("Got bad value of %f\nOther fields were %s %s %d\n", value, uName.c_str(), bName.c_str(), voteTime);
      continue;
    }

    if (itemCate.find(bName) == itemCate.end()) {
//      printf("Item not found.\n");
      continue;
    }

    string cat = itemCate[bName];
    if (cat == "Men") {
      of_Men << line << "\n";
    } else if (cat == "Women") {
      of_Women << line << "\n";
    } else if (cat == "Boys") {
      of_Boys << line << "\n";
    } else if (cat == "Girls") {
      of_Girls << line << "\n";
    } else if (cat == "Baby") {
      of_Baby << line << "\n";
    } else {
      printf("What is this category: %s? \n", cat.c_str());
      exit(1);
    }

    lineCount ++;
    if (not (lineCount % 100000)) {
      fprintf(stderr, ".");
      fflush(stderr);
    }
  }

  in.close();

  of_Men.close();
  of_Women.close();
  of_Boys.close();
  of_Girls.close();
  of_Baby.close();

  printf("\n  Congrats. All written successfully.\n");
}
