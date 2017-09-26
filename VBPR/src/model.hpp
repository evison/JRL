#pragma once

#include "common.hpp"
#include "corpus.hpp"
#include "categoryTree.hpp"

enum action_t { COPY, INIT, FREE };

class model
{
public:
	model(corpus* corp) 
	: corp(corp)
	{
		nUsers = corp->nWholeUsers;
		nItems = corp->nWholeItems;
		nVotes = corp->nWholeVotes;

		train_nUsers = corp->nTrainUsers;
		train_nItems = corp->nTrainItems;
		train_nVotes = corp->nTrainVotes;

		test_nUsers = corp->nTestUsers;
		test_nItems = corp->nTestItems;
		test_nVotes = corp->nTestVotes;

		
		
		// leave out `two' for each user
		//test_per_user = new pair<int,long long> [nUsers];
		//val_per_user  = new pair<int,long long> [nUsers];
		//for (int u = 0; u < nUsers; u ++) {
		//	test_per_user[u] = make_pair(-1, -1);  // -1 denotes empty
		//	val_per_user[u]  = make_pair(-1, -1);
		//}

		// split into training set AND valid set AND test set 
		// NOTE: never use corp->V as the training set
		pos_per_user = new map<int,long long>[nUsers];
		pos_per_item = new map<int,long long>[nItems];
		for (int x = 0; x < train_nVotes; x ++) {
			vote* v = corp->trainV.at(x);
			int user = v->user;
			int item = v->item;
			long long voteTime = v->voteTime; 
			pos_per_user[user][item] = voteTime;
			pos_per_item[item][user] = voteTime;
		}


		test_per_user = new map<int,long long>[nUsers];
		test_per_item = new map<int,long long>[nItems];
		for (int x = 0; x < test_nVotes; x ++) {
			vote* v = corp->testV.at(x);
			int user = v->user;
			int item = v->item;
			long long voteTime = v->voteTime; 

			test_per_user[user][item] = voteTime;
			test_per_item[item][user] = voteTime;
		}


		// sanity check
		//for (int u = 0; u < nUsers; u ++) {
		//	if (test_per_user[u].first == -1 || val_per_user[u].first == -1) {
		//		fprintf(stderr, "\n\n Corpus split exception.\n");
		//		exit(1);
		//	}
		//}
                //recommend_per_user = new map<int,int>[nUsers];
		// calculate num_pos_events
		num_pos_events = 0;
		for (int u = 0; u < nUsers; u ++) {
			num_pos_events += pos_per_user[u].size();
		}
	}

	~model()
	{
		delete [] pos_per_user;
		delete [] pos_per_item;

		delete [] test_per_user;
		delete [] test_per_item;
	}

	/* Model parameters */
	int NW; // Total number of parameters
	double* W; // Contiguous version of all parameters
	double* bestW;

	/* Corpus related */
	corpus* corp; // dangerous
	corpus* trainCorp;
	corpus* testCorp;

	int nUsers; // Number of users
	int nItems; // Number of items
	int nVotes; // Number of ratings

	int train_nUsers; // Number of users
	int train_nItems; // Number of items
	int train_nVotes; // Number of ratings

	int test_nUsers; // Number of users
	int test_nItems; // Number of items
	int test_nVotes; // Number of ratings

	map<int,long long>* pos_per_user;
	map<int,long long>* pos_per_item;

	map<int,long long>* test_per_user;
	map<int,long long>* test_per_item;

        //int** recommend_per_user;

	//pair<int,long long>* val_per_user;
	//pair<int,long long>* test_per_user;

	int num_pos_events;

	/* Category information */
	int  nCategory;
	int* itemCategoryId;
	map<int,int> nodeIds;
	map<int,string> rNodeIds;

	/* additional information for demo paper */
	map<int, double> itemPrice;
	map<int, string> itemBrand;
	
	void loadCategories(const char* categoryPath, string subcategoryName, string rootName, bool skipRoot);

	virtual void AUCCX(double* AUC_test, double* std);
	virtual void P(double* test, double* std, int flag);
	virtual void AUC(double* AUC_test, double* std);
	virtual void AUC_coldItem(double* AUC_test, double* std, int* num_user);
	virtual void copyBestModel();
	virtual void saveModel(const char* path);
	virtual void loadModel(const char* path);
	
	virtual string toString();

private:
	virtual double prediction(int user, int item) = 0;
};
