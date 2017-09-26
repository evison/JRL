#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "vector"
#include "math.h"
#include "string.h"
#include <string>
#include <iostream>
#include "omp.h"
#include "map"
#include "set"
#include "vector"
#include "common.hpp"
#include "algorithm"
#include "lbfgs.h"
#include "sstream"
#include "gzstream.h"

/// Safely open a file
FILE* fopen_(const char* p, const char* m)
{
  FILE* f = fopen(p, m);
  if (!f)
  {
    printf("Failed to open %s\n", p);
    exit(1);
  }
  return f;
}

/// Data associated with a rating
struct vote
{
  int user; // ID of the user
  int item; // ID of the item
  float value; // Rating

  int voteTime; // Unix-time of the rating
  std::vector<int> words; // IDs of the words in the review
};

typedef struct vote vote;

/// To sort words by frequency in a corpus
bool wordCountCompare(std::pair<std::string, int> p1, std::pair<std::string, int> p2)
{
  return p1.second > p2.second;
}

/// To sort votes by product ID
bool voteCompare(vote* v1, vote* v2)
{
  return v1->item > v2->item;
}

/// Sign (-1, 0, or 1)
template<typename T> int sgn(T val)
{
  return (val > T(0)) - (val < T(0));
}

class corpus
{
public:
  corpus(std::string voteWholeFile, std::string voteTrainFile, std::string voteValidFile, std::string voteTestFile, int max)
  {
    std::map<std::string, int> uCounts;
    std::map<std::string, int> bCounts;

    std::string uName;
    std::string bName;
    float value;
    int voteTime;
    int nw;
    int nRead = 0;

    igzstream in;
    in.open(voteWholeFile.c_str());
    std::string line;
    std::string sWord;
    
    // Read the input file. The first time the file is read it is only to compute word counts, in order to select the top "maxWords" words to include in the dictionary
    while (std::getline(in, line))
    {
      std::stringstream ss(line);
      ss >> uName >> bName >> value >> voteTime >> nw;
      if (value > 5 or value < 0)
      { // Ratings should be in the range [0,5]
        printf("Got bad value of %f\nOther fields were %s %s %d\n", value, uName.c_str(), bName.c_str(), voteTime);
        exit(0);
      }
      for (int w = 0; w < nw; w++)
      {
        ss >> sWord;
        if (wordCount.find(sWord) == wordCount.end())
          wordCount[sWord] = 0;
        wordCount[sWord]++;
      }

      if (uCounts.find(uName) == uCounts.end())
        uCounts[uName] = 0;
      if (bCounts.find(bName) == bCounts.end())
        bCounts[bName] = 0;
      uCounts[uName]++;
      bCounts[bName]++;

      nRead++;
      if (nRead % 100000 == 0)
      { 
        printf(".");
        fflush(stdout);
      }

      if (max > 0 and (int) nRead >= max)
        break;
    }
    in.close();

    printf("\nnWholeUsers = %d, nWholeItems = %d, nWholeRatings = %d\n", (int) uCounts.size(), (int) bCounts.size(), nRead);

    wholeV = new std::vector<vote*>();
    vote* v = new vote();
    std::map<std::string, int> userIds;
    std::map<std::string, int> beerIds;

    nUsers = 0;
    nBeers = 0;

    int userMin = 0;
    int beerMin = 0;

    int maxWords = 5000; // Dictionary size
    std::vector < std::pair<std::string, int> > whichWords;
    for (std::map<std::string, int>::iterator it = wordCount.begin(); it != wordCount.end(); it++)
      whichWords.push_back(*it);
    sort(whichWords.begin(), whichWords.end(), wordCountCompare);
    if ((int) whichWords.size() < maxWords)
      maxWords = (int) whichWords.size();
    nWords = maxWords;
    for (int w = 0; w < maxWords; w++)
    {
      wordId[whichWords[w].first] = w;
      idWord[w] = whichWords[w].first;
    }

    // Re-read the entire file, this time building structures from those words in the dictionary
    printf("\nbegin reading whole file");
    igzstream in2;
    in2.open(voteWholeFile.c_str());
    nRead = 0;
    while (std::getline(in2, line))
    {
      std::stringstream ss(line);
      ss >> uName >> bName >> value >> voteTime >> nw;

      for (int w = 0; w < nw; w++)
      {
        ss >> sWord;
        if (wordId.find(sWord) != wordId.end())
          v->words.push_back(wordId[sWord]);
      }

      if (uCounts[uName] >= userMin)
      {
        if (userIds.find(uName) == userIds.end())
        {
          rUserIds[nUsers] = uName;
          userIds[uName] = nUsers++;
        }
        v->user = userIds[uName];
      }
      else
        continue;

      if (bCounts[bName] >= beerMin)
      {
        if (beerIds.find(bName) == beerIds.end())
        {
          rBeerIds[nBeers] = bName;
          beerIds[bName] = nBeers++;
        }
        v->item = beerIds[bName];
      }
      else
        continue;

      v->value = value;
      v->voteTime = voteTime;

      wholeV->push_back(v);
      v = new vote();

      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush( stdout);
      }

      if (max > 0 and (int) nRead >= max)
        break;
    }

    printf("\n");
    delete v;

    in2.close();
    printf("\nend reading whole file");
    

    
    // read train dataset
    trainV = new std::vector<vote*>();
    v = new vote();
    
    printf("\nbegin reading train file"); 
    igzstream in3;
    in3.open(voteTrainFile.c_str());
    nRead = 0;
    while (std::getline(in3, line))
    {
      std::stringstream ss(line);
      ss >> uName >> bName >> value >> voteTime >> nw;

      for (int w = 0; w < nw; w++)
      {
        ss >> sWord;
        if (wordId.find(sWord) != wordId.end())
          v->words.push_back(wordId[sWord]);
      }

      if (uCounts[uName] >= userMin)
      {
        v->user = userIds[uName];
      }
      else
        continue;

      if (bCounts[bName] >= beerMin)
      {
        v->item = beerIds[bName];
      }
      else
        continue;

      v->value = value;
      v->voteTime = voteTime;

      trainV->push_back(v);
      v = new vote();
 
      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush( stdout);
      }

      if (max > 0 and (int) nRead >= max)
        break;
    }

    printf("\n");
    delete v;
    in3.close();     
    printf("\nend reading train file");
    
    // read valid dataset
    validV = new std::vector<vote*>();
    v = new vote();

    printf("\nbegin reading valid file");
    igzstream in4;
    in4.open(voteValidFile.c_str());
    nRead = 0;
    while (std::getline(in4, line))
    {
      std::stringstream ss(line);
      ss >> uName >> bName >> value >> voteTime >> nw;
      //printf("%s", bName);
      for (int w = 0; w < nw; w++)
      {
        ss >> sWord;
        if (wordId.find(sWord) != wordId.end())
          v->words.push_back(wordId[sWord]);
      }

      if (uCounts[uName] >= userMin)
      {
        v->user = userIds[uName];
      }
      else
        continue;

      if (bCounts[bName] >= beerMin)
      {
        v->item = beerIds[bName];
      }
      else
        continue;

      v->value = value;
      v->voteTime = voteTime;
      validV->push_back(v);
      v = new vote();

      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush( stdout);
      }

      if (max > 0 and (int) nRead >= max)
        break;
    }

    printf("\n");
    delete v;
    in4.close();
    printf("\nend reading valid file");

    // read test dataset 
    testV = new std::vector<vote*>();
    v = new vote();

    printf("\nend reading test file");
    igzstream in5;
    in5.open(voteTestFile.c_str());
    nRead = 0;
    while (std::getline(in5, line))
    {
      std::stringstream ss(line);
      ss >> uName >> bName >> value >> voteTime >> nw;

      for (int w = 0; w < nw; w++)
      {
        ss >> sWord;
        if (wordId.find(sWord) != wordId.end())
          v->words.push_back(wordId[sWord]);
      }

      if (uCounts[uName] >= userMin)
      {
        v->user = userIds[uName];
      }
      else
        continue;

      if (bCounts[bName] >= beerMin)
      {
        v->item = beerIds[bName];
      }
      else
        continue;

      v->value = value;
      v->voteTime = voteTime;

      testV->push_back(v);
      v = new vote();

      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush( stdout);
      }

      if (max > 0 and (int) nRead >= max)
        break;
    }

    printf("\n");
    delete v;
    in5.close(); 
    printf("\nend reading test file");
  }

  ~corpus()
  {
    for (std::vector<vote*>::iterator it = wholeV->begin(); it != wholeV->end(); it++)
      delete *it;
    delete wholeV;
    for (std::vector<vote*>::iterator it = trainV->begin(); it != trainV->end(); it++)
      delete *it;
    delete trainV;
    for (std::vector<vote*>::iterator it = validV->begin(); it != validV->end(); it++)
      delete *it;
    delete validV;
    for (std::vector<vote*>::iterator it = testV->begin(); it != testV->end(); it++)
      delete *it;
    delete testV;

  }

  std::vector<vote*>* wholeV;
  std::vector<vote*>* trainV;
  std::vector<vote*>* validV;
  std::vector<vote*>* testV;


  int nUsers; // Number of users
  int nBeers; // Number of items
  int nWords; // Number of words

  std::map<std::string, int> userIds; // Maps a user's string-valued ID to an integer
  std::map<std::string, int> beerIds; // Maps an item's string-valued ID to an integer

  std::map<int, std::string> rUserIds; // Inverse of the above map
  std::map<int, std::string> rBeerIds;

  std::map<std::string, int> wordCount; // Frequency of each word in the corpus
  std::map<std::string, int> wordId; // Map each word to its integer ID
  std::map<int, std::string> idWord; // Inverse of the above map
};
