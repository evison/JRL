#pragma once

#include "common.hpp"

class corpus
{
public:
	corpus() {}
	~corpus() {}

	vector<vote*> wholeV; // vote
        vector<vote*> trainV; // vote
        vector<vote*> testV; // vote


	int nWholeUsers; // Number of users
	int nWholeItems; // Number of items
	int nWholeVotes; // Number of ratings

        int nTrainUsers; // Number of users
        int nTrainItems; // Number of items
        int nTrainVotes; // Number of ratings

        int nTestUsers; // Number of users
        int nTestItems; // Number of items
        int nTestVotes; // Number of ratings


	map<string, int> userIds; // Maps a user's string-valued ID to an integer
	map<string, int> itemIds; // Maps an item's string-valued ID to an integer

	map<int, string> rUserIds; // Inverse of the above maps
	map<int, string> rItemIds;
        
        map<string, int> userTrainIds;
        map<string, int> itemTrainIds;

        map<string, int> userTestIds;
        map<string, int> itemTestIds;

	/* For pre-load */
	map<string, int> imgAsins;
	map<string, int> uCounts;
	map<string, int> bCounts;

	vector<vector<pair<int, float> > > imageFeatures;
	int imFeatureDim;  // fixed to 4096

	/* For WWW demo */
	vector<double> avgRatingPerItem;
	vector<int> numReviewsPerItem;

	virtual void loadData(const char* voteWholeFile, const char* voteTrainFile, const char* voteTestFile, const char* imgFeatPath, int userMin, int itemMin, int mode);
	virtual void cleanUp();

private:
	void loadVotes(const char* imgFeatPath, const char* voteFile, int userMin, int itemMin);
	void loadBPRVotes(const char* voteWholeFile, const char* voteTrainFile, const char* voteTestFile, int userMin, int itemMin);
        void loadVBPRVotes(const char* voteWholeFile, const char* voteTrainFile, const char* voteTestFile, const char* imgFeatPath, int userMin, int itemMin);
        void loadImgFeatures(const char* imgFeatPath);
	void generateVotes(map<pair<int, int>, long long>& voteMap, int mode);
};
