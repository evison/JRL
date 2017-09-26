#include "corpus.hpp"

void corpus::loadData(const char* voteWholeFile, const char* voteTrainFile, const char* voteTestFile, const char* imgFeatPath, int userMin, int itemMin, int mode)
{
    nWholeItems = 0;
    nWholeUsers = 0;
    nWholeVotes = 0;
    
    nTrainItems = 0;
    nTrainUsers = 0;
    nTrainVotes = 0;
    
    nTestItems = 0;
    nTestUsers = 0;
    nTestVotes = 0;


    imFeatureDim = 4096;
    
    if(mode == 0)
    {
        loadBPRVotes(voteWholeFile, voteTrainFile, voteTestFile, userMin, itemMin);
    }
    else
    {
        loadVBPRVotes(voteWholeFile, voteTrainFile, voteTestFile, imgFeatPath, userMin, itemMin);     
        loadImgFeatures(imgFeatPath);
    }    	
    fprintf(stderr, "\n  \"nUsers\": %d, \"nItems\": %d, \"nVotes\": %d\n", nWholeUsers, nWholeItems, nWholeVotes);
    fprintf(stderr, "\n  \"nTrainUsers\": %d, \"nTrainItems\": %d, \"nTrainVotes\": %d\n", nTrainUsers, nTrainItems, nTrainVotes);
    fprintf(stderr, "\n  \"nTestUsers\": %d, \"nTestItems\": %d, \"nTestVotes\": %d\n", nTestUsers, nTestItems, nTestVotes);
}

void corpus::cleanUp()
{
	for (vector<vote*>::iterator it = wholeV.begin(); it != wholeV.end(); it++) {
		delete *it;
	}
       for (vector<vote*>::iterator it = trainV.begin(); it != trainV.end(); it++) {
                delete *it;
        }
    for (vector<vote*>::iterator it = testV.begin(); it != testV.end(); it++) {
                delete *it;
        }
}

void corpus::loadBPRVotes(const char* voteWholeFile, const char* voteTrainFile, const char* voteTestFile, int userMin, int itemMin){
    
    fprintf(stderr, "  Loading votes from %s,%s,%s, userMin = %d, itemMin = %d  ", voteWholeFile, voteTrainFile, voteTestFile, userMin, itemMin);

    string uName; // User name
    string bName; // Item name
    float value; // Rating
    long long voteTime; // Time rating was entered
    map<pair<int, int>, long long> voteWholeMap;
    map<pair<int, int>, long long> voteTrainMap;
    map<pair<int, int>, long long> voteTestMap;

    int nRead = 0; // Progress
    string line;

    igzstream in;
    in.open(voteWholeFile);
    if (! in.good()) {
        fprintf(stderr, "Can't read votes from %s.\n", voteWholeFile);
        exit(1);
    }

    // The first pass is for filtering
    while (getline(in, line)) {
        stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime;

        nRead++;
        if (nRead % 100000 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }

        //if (imgAsins.find(bName) == imgAsins.end()) {
        //  continue;
        //}

        if (value > 5 or value < 0) { // Ratings should be in the range [0,5]
            printf("Got bad value of %f\nOther fields were %s %s %lld\n", value, uName.c_str(), bName.c_str(), voteTime);
            exit(1);
        }

        if (uCounts.find(uName) == uCounts.end()) {
            uCounts[uName] = 0;
        }
        if (bCounts.find(bName) == bCounts.end()) {
            bCounts[bName] = 0;
        }
        uCounts[uName]++;
        bCounts[bName]++;
    }
    in.close();




    // Re-read the whole file .
    nWholeUsers = 0;
    nWholeItems = 0;

    igzstream in2;
    in2.open(voteWholeFile);
    if (! in2.good()) {
        fprintf(stderr, "Can't read votes from %s.\n", voteWholeFile);
        exit(1);
    }

    vector<vector<double> > ratingPerItem;

    nRead = 0;

    while (getline(in2, line)) {
        stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime;
        // reviewer_id, asin, overall, time, text

        nRead++;
        if (nRead % 100000 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }
        

        //if (imgAsins.find(bName) == imgAsins.end()) {
        // continue;
        //}

        if (uCounts[uName] < userMin or bCounts[bName] < itemMin) {
            continue;
        }

        // new item
        if (itemIds.find(bName) == itemIds.end()) {
            rItemIds[nWholeItems] = bName;
            itemIds[bName] = nWholeItems++;
            vector<double> vec;
            ratingPerItem.push_back(vec);
        }
        // new user
        if (userIds.find(uName) == userIds.end()) {
            rUserIds[nWholeUsers] = uName;
            userIds[uName] = nWholeUsers++;
        }

        ratingPerItem[itemIds[bName]].push_back(value);

        // this is a diction of u,i pairs w/ a value of time
        // (u,i)=>time
        voteWholeMap[make_pair(userIds[uName], itemIds[bName])] = voteTime;
    }
    in2.close();

    
    // Read and map train file
    nTrainUsers = 0;
    nTrainItems = 0;

    igzstream in3;
    in3.open(voteTrainFile);
    if (! in3.good()) {
        fprintf(stderr, "Can't read votes from %s.\n", voteTrainFile);
        exit(1);
    }

    nRead = 0;

    while (getline(in3, line)) {
        stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime;
        // reviewer_id, asin, overall, time, text

        nRead++;
        if (nRead % 100000 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }

        if (itemTrainIds.find(bName) == itemTrainIds.end()) {
            itemTrainIds[bName] = nTrainItems++;
        }
        // new user
        if (userTrainIds.find(uName) == userTrainIds.end()) {
            userTrainIds[uName] = nTrainUsers++;
        }



        voteTrainMap[make_pair(userIds[uName], itemIds[bName])] = voteTime;
    }
    in3.close();


    // Read and map test data
    nTestUsers = 0;
    nTestItems = 0;

    igzstream in4;
    in4.open(voteTestFile);
    if (! in4.good()) {
        fprintf(stderr, "Can't read votes from %s.\n", voteTestFile);
        exit(1);
    }


    nRead = 0;

    while (getline(in4, line)) {
        stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime;
        // reviewer_id, asin, overall, time, text

        nRead++;
        if (nRead % 100000 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }

        if (itemTestIds.find(bName) == itemTestIds.end()) {
            itemTestIds[bName] = nTestItems++;
        }
        // new user
        if (userTestIds.find(uName) == userTestIds.end()) {
            userTestIds[uName] = nTestUsers++;
        }

         
        voteTestMap[make_pair(userIds[uName], itemIds[bName])] = voteTime;
    }
    in4.close();





    for (int x = 0; x < nWholeItems; x ++) {
        numReviewsPerItem.push_back(ratingPerItem[x].size());
        double sum = 0;
        for (int j = 0; j < (int)ratingPerItem[x].size(); j ++) {
            sum += ratingPerItem[x].at(j);
        }
        if (ratingPerItem[x].size() > 0) {
            avgRatingPerItem.push_back(sum / ratingPerItem[x].size());
        } 
        else {
            avgRatingPerItem.push_back(0);
        }
    }

    fprintf(stderr, "\n");
    generateVotes(voteWholeMap, 0);
    generateVotes(voteTrainMap, 1);
    generateVotes(voteTestMap, 2);

}



void corpus::loadVBPRVotes(const char* voteWholeFile, const char* voteTrainFile, const char* voteTestFile, const char* imgFeatPath, int userMin, int itemMin)
{
    //image part
    FILE* f = fopen_(imgFeatPath, "rb");
    fprintf(stderr, "\n  Pre-loading image asins from %s  ", imgFeatPath);

    float* feat = new float [imFeatureDim];
    char* asin = new char [11];
    asin[10] = '\0';
    int a;
    int counter = 0;
    while (!feof(f)) {
        if ((a = fread(asin, sizeof(*asin), 10, f)) != 10) { // last line might be empty
            continue;
        }

        // trim right space
        string sAsin(asin);
        size_t found = sAsin.find(" ");
        if (found != string::npos) {
            sAsin = sAsin.substr(0, found);
        }

        for (unsigned c = 0; c < sAsin.size(); c ++) {
            if (not isascii(asin[c])) {
            printf("Expected asin to be 10-digit ascii\n");
            exit(1);
            }
        }
        if (not (counter % 10000)) {
            fprintf(stderr, ".");
            fflush(stderr);
        }

        if ((a = fread(feat, sizeof(*feat), imFeatureDim, f)) != imFeatureDim) {
            printf("Expected to read %d floats, got %d\n", imFeatureDim, a);
            exit(1);
        }
        imgAsins[sAsin] = 1;
        counter ++;
    }
    fprintf(stderr, "\n");

    delete[] asin;
    delete [] feat;
    fclose(f);
    // end image part
    
    // begin review part
    fprintf(stderr, "  Loading votes from %s,%s,%s, userMin = %d, itemMin = %d  ", voteWholeFile, voteTrainFile, voteTestFile, userMin, itemMin);

    string uName; // User name
    string bName; // Item name
    float value; // Rating
    long long voteTime; // Time rating was entered
    map<pair<int, int>, long long> voteWholeMap;
    map<pair<int, int>, long long> voteTrainMap;
    map<pair<int, int>, long long> voteTestMap;

    int nRead = 0; // Progress
    string line;

    igzstream in;
    in.open(voteWholeFile);
    if (! in.good()) {
        fprintf(stderr, "Can't read votes from %s.\n", voteWholeFile);
        exit(1);
    }

    // The first pass is for filtering
    while (getline(in, line)) {
        stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime;

        nRead++;
        if (nRead % 100000 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }

        if (imgAsins.find(bName) == imgAsins.end()) {
          continue;
        }

        if (value > 5 or value < 0) { // Ratings should be in the range [0,5]
            printf("Got bad value of %f\nOther fields were %s %s %lld\n", value, uName.c_str(), bName.c_str(), voteTime);
            exit(1);
        }

        if (uCounts.find(uName) == uCounts.end()) {
            uCounts[uName] = 0;
        }
        if (bCounts.find(bName) == bCounts.end()) {
            bCounts[bName] = 0;
        }
        uCounts[uName]++;
        bCounts[bName]++;
    }
    in.close();

    // Re-read the whole file
    nWholeUsers = 0;
    nWholeItems = 0;

    igzstream in2;
    in2.open(voteWholeFile);
    if (! in2.good()) {
        fprintf(stderr, "Can't read votes from %s.\n", voteWholeFile);
        exit(1);
    }

    vector<vector<double> > ratingPerItem;

    nRead = 0;

    while (getline(in2, line)) {
        stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime;
        // reviewer_id, asin, overall, time, text

        nRead++;
        if (nRead % 100000 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }


        if (imgAsins.find(bName) == imgAsins.end()) {
            continue;
        }

        if (uCounts[uName] < userMin or bCounts[bName] < itemMin) {
            continue;
        }

        // new item
        if (itemIds.find(bName) == itemIds.end()) {
            rItemIds[nWholeItems] = bName;
            itemIds[bName] = nWholeItems++;
            vector<double> vec;
            ratingPerItem.push_back(vec);
        }
        // new user
        if (userIds.find(uName) == userIds.end()) {
            rUserIds[nWholeUsers] = uName;
            userIds[uName] = nWholeUsers++;
        }

        ratingPerItem[itemIds[bName]].push_back(value);

        // this is a diction of u,i pairs w/ a value of time
        // (u,i)=>time
        voteWholeMap[make_pair(userIds[uName], itemIds[bName])] = voteTime;
    }
    in2.close();
 

    // Read and map train file
    nTrainUsers = 0;
    nTrainItems = 0;

    igzstream in3;
    in3.open(voteTrainFile);
    if (! in3.good()) {
        fprintf(stderr, "Can't read votes from %s.\n", voteTrainFile);
        exit(1);
    }

    nRead = 0;

    while (getline(in3, line)) {
        stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime;
        // reviewer_id, asin, overall, time, text
        
        if (imgAsins.find(bName) == imgAsins.end()) {
            continue;
        }
   
        nRead++;
        if (nRead % 100000 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }

        if (itemTrainIds.find(bName) == itemTrainIds.end()) {
            itemTrainIds[bName] = nTrainItems++;
        }
        // new user
        if (userTrainIds.find(uName) == userTrainIds.end()) {
            userTrainIds[uName] = nTrainUsers++;
        }



        voteTrainMap[make_pair(userIds[uName], itemIds[bName])] = voteTime;
    }
    in3.close();


    // Read and map test data
    nTestUsers = 0;
    nTestItems = 0;

    igzstream in4;
    in4.open(voteTestFile);
    if (! in4.good()) {
        fprintf(stderr, "Can't read votes from %s.\n", voteTestFile);
        exit(1);
    }


    nRead = 0;

    while (getline(in4, line)) {
        stringstream ss(line);
        ss >> uName >> bName >> value >> voteTime;
        // reviewer_id, asin, overall, time, text

        if (imgAsins.find(bName) == imgAsins.end()) {
            continue;
        }  

        nRead++;
        if (nRead % 100000 == 0) {
            fprintf(stderr, ".");
            fflush(stderr);
        }

        if (itemTestIds.find(bName) == itemTestIds.end()) {
            itemTestIds[bName] = nTestItems++;
        }
        // new user
        if (userTestIds.find(uName) == userTestIds.end()) {
            userTestIds[uName] = nTestUsers++;
        }


        voteTestMap[make_pair(userIds[uName], itemIds[bName])] = voteTime;
    }
    in4.close();
    // end review part

    for (int x = 0; x < nWholeItems; x ++) {
        numReviewsPerItem.push_back(ratingPerItem[x].size());
        double sum = 0;
        for (int j = 0; j < (int)ratingPerItem[x].size(); j ++) {
            sum += ratingPerItem[x].at(j);
        }
        if (ratingPerItem[x].size() > 0) {
            avgRatingPerItem.push_back(sum / ratingPerItem[x].size());
        }
        else {
            avgRatingPerItem.push_back(0);
        }
    }

    fprintf(stderr, "\n");
    generateVotes(voteWholeMap, 0);
    generateVotes(voteTrainMap, 1);
    generateVotes(voteTestMap, 2);

}





void corpus::loadVotes(const char* imgFeatPath, const char* voteFile, int userMin, int itemMin)
{

  //image part
  FILE* f = fopen_(imgFeatPath, "rb");
  fprintf(stderr, "\n  Pre-loading image asins from %s  ", imgFeatPath);

  float* feat = new float [imFeatureDim];
  char* asin = new char [11];
  asin[10] = '\0';
  int a;
  int counter = 0;
  while (!feof(f)) {
    
    if ((a = fread(asin, sizeof(*asin), 10, f)) != 10) { // last line might be empty
      continue;
    }
    
    
    
    // trim right space
    string sAsin(asin);
    size_t found = sAsin.find(" ");
    if (found != string::npos) {
        sAsin = sAsin.substr(0, found);
    }

    for (unsigned c = 0; c < sAsin.size(); c ++) {
      if (not isascii(asin[c])) {
        printf("Expected asin to be 10-digit ascii\n");
        exit(1);
      }
    }
    if (not (counter % 10000)) {
      fprintf(stderr, ".");
      fflush(stderr);
    }

    if ((a = fread(feat, sizeof(*feat), imFeatureDim, f)) != imFeatureDim) {
      printf("Expected to read %d floats, got %d\n", imFeatureDim, a);
      exit(1);
    }
    imgAsins[sAsin] = 1;
    counter ++;
  }
  fprintf(stderr, "\n");

  delete[] asin;
  delete [] feat;
  fclose(f);
  
  //end image part

	fprintf(stderr, "  Loading votes from %s, userMin = %d, itemMin = %d  ", voteFile, userMin, itemMin);

	string uName; // User name
	string bName; // Item name
	float value; // Rating
	long long voteTime; // Time rating was entered
	map<pair<int, int>, long long> voteMap;

	int nRead = 0; // Progress
	string line;

	igzstream in;
	in.open(voteFile);
	if (! in.good()) {
		fprintf(stderr, "Can't read votes from %s.\n", voteFile);
		exit(1);
	}

	// The first pass is for filtering
	while (getline(in, line)) {
		stringstream ss(line);
		ss >> uName >> bName >> value;

		nRead++;
		if (nRead % 100000 == 0) {
			fprintf(stderr, ".");
			fflush(stderr);
		}

    //if (imgAsins.find(bName) == imgAsins.end()) {
    //  continue;
    //}

		if (value > 5 or value < 0) { // Ratings should be in the range [0,5]
			printf("Got bad value of %f\nOther fields were %s %s %lld\n", value, uName.c_str(), bName.c_str(), voteTime);
			exit(1);
		}

		if (uCounts.find(uName) == uCounts.end()) {
			uCounts[uName] = 0;
		}
		if (bCounts.find(bName) == bCounts.end()) {
			bCounts[bName] = 0;
		}
		uCounts[uName]++;
		bCounts[bName]++;
	}
	in.close();

	// Re-read
	nWholeUsers = 0;
	nWholeItems = 0;
	
	igzstream in2;
	in2.open(voteFile);
	if (! in2.good()) {
		fprintf(stderr, "Can't read votes from %s.\n", voteFile);
		exit(1);
	}

	vector<vector<double> > ratingPerItem;

	nRead = 0;
        
	while (getline(in2, line)) {
		stringstream ss(line);
		ss >> uName >> bName >> value >> voteTime;
    // reviewer_id, asin, overall, time, text

		nRead++;
		if (nRead % 100000 == 0) {
			fprintf(stderr, ".");
			fflush(stderr);
		}

    //if (imgAsins.find(bName) == imgAsins.end()) {
    //    continue;
    //}

    if (uCounts[uName] < userMin or bCounts[bName] < itemMin) {
      continue;
    }

		// new item
		if (itemIds.find(bName) == itemIds.end()) {
			rItemIds[nWholeItems] = bName;
			itemIds[bName] = nWholeItems++;
			vector<double> vec;
			ratingPerItem.push_back(vec);
		}
		// new user
		if (userIds.find(uName) == userIds.end()) {
			rUserIds[nWholeUsers] = uName;
			userIds[uName] = nWholeUsers++;
		}

		ratingPerItem[itemIds[bName]].push_back(value);

    // this is a diction of u,i pairs w/ a value of time
    //(u,i)=>time
		voteMap[make_pair(userIds[uName], itemIds[bName])] = voteTime;	
	}
	in2.close();

	for (int x = 0; x < nWholeItems; x ++) {
		numReviewsPerItem.push_back(ratingPerItem[x].size());
		double sum = 0;
		for (int j = 0; j < (int)ratingPerItem[x].size(); j ++) {
			sum += ratingPerItem[x].at(j);
		}
		if (ratingPerItem[x].size() > 0) {
			avgRatingPerItem.push_back(sum / ratingPerItem[x].size());
		} else {
			avgRatingPerItem.push_back(0);
		}
	}

	fprintf(stderr, "\n");
	generateVotes(voteMap, 0);
}

void corpus::generateVotes(map<pair<int, int>, long long>& voteMap, int mode)
{
	fprintf(stderr, "\n  Generating %d votes data ", mode);
	if (mode == 0){
	    for(map<pair<int, int>, long long>::iterator it = voteMap.begin(); it != voteMap.end(); it ++) {
	    	vote* v = new vote();
		v->user = it->first.first;
		v->item = it->first.second;
		v->voteTime = it->second;
		v->label = 1; // positive
		wholeV.push_back(v);
	    }
	    nWholeVotes = wholeV.size();
	    random_shuffle(wholeV.begin(), wholeV.end());
        }
        else if (mode == 1) {
            for(map<pair<int, int>, long long>::iterator it = voteMap.begin(); it != voteMap.end(); it ++) {
                vote* v = new vote();
                v->user = it->first.first;
                v->item = it->first.second;
                v->voteTime = it->second;
                v->label = 1; // positive
                trainV.push_back(v);
            }
            nTrainVotes = trainV.size();
            random_shuffle(trainV.begin(), trainV.end());
        }
        else if (mode == 2) {
            for(map<pair<int, int>, long long>::iterator it = voteMap.begin(); it != voteMap.end(); it ++) {
                vote* v = new vote();
                v->user = it->first.first;
                v->item = it->first.second;
                v->voteTime = it->second;
                v->label = 1; // positive
                testV.push_back(v);
            }
            nTestVotes = testV.size();
            random_shuffle(testV.begin(), testV.end());
        }
        else {
            printf("not valid mode.");
        }    
}

void corpus::loadImgFeatures(const char* imgFeatPath)
{
  for (int i = 0; i < nWholeItems; i ++) {
    vector<pair<int, float> > vec;
    imageFeatures.push_back(vec);
  }

  FILE* f = fopen_(imgFeatPath, "rb");
  fprintf(stderr, "\n  Loading image features from %s  ", imgFeatPath);

  float ma = 58.388599; // Largest feature observed

  float* feat = new float [imFeatureDim];
  char* asin = new char [11];
  asin[10] = '\0';
  int a;
  int counter = 0;
  while (!feof(f)) {
    if ((a = fread(asin, sizeof(*asin), 10, f)) != 10) {
      //printf("Expected to read %d chars, got %d\n", 10, a);
      continue;
    }
        // trim right space
        string sAsin(asin);
        size_t found = sAsin.find(" ");
        if (found != string::npos) {
            sAsin = sAsin.substr(0, found);
        }

    //read 4096 float-sized bytes
    if ((a = fread(feat, sizeof(*feat), imFeatureDim, f)) != imFeatureDim) {
      printf("Expected to read %d floats, got %d\n", imFeatureDim, a);
      exit(1);
    }

    if (itemIds.find(sAsin) == itemIds.end()) {
      continue;
    }

    vector<pair<int, float> > &vec = imageFeatures.at(itemIds[sAsin]);
    for (int f = 0; f < imFeatureDim; f ++) {
      if (feat[f] != 0) {  // compression
        vec.push_back(std::make_pair(f, feat[f]/ma));
      }
    }

    // print process
    counter ++;
    if (not (counter % 10000)) {
      fprintf(stderr, ".");
      fflush(stderr);
    }
  }
  fprintf(stderr, "\n");

  delete [] asin;
  delete [] feat;
  fclose(f);
}



