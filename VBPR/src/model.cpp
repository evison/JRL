#include "model.hpp"

// Parse category info for all products
void model::loadCategories(const char* categoryPath, string subcategoryName, string rootName, bool skipRoot)
{
	itemCategoryId = new int [nItems];
	for (int i = 0; i < nItems; i ++) {
		itemCategoryId[i] = -1;
	}

	fprintf(stderr, "\n  Loading category data");
	categoryTree* ct = new categoryTree(rootName, skipRoot);

	igzstream in;
	in.open(categoryPath);
	if (! in.good()) {
		fprintf(stderr, "\n  Can't load category from %s.\n", categoryPath);
		exit(1);
	}

	string line;

	int item = -1;
	int count = 0;
	nCategory = 0;

	while (getline(in, line)) {
		istringstream ss(line);

		if (line.c_str()[0] != ' ') {
			string itemId;
			double price = -1;
			string brand("unknown_brand");
			ss >> itemId >> price >> brand;
			if (corp->itemIds.find(itemId) == corp->itemIds.end()) {
				item = -1;
				continue;
			}

			item = corp->itemIds[itemId];
			itemPrice[item] = price;
			itemBrand[item] = brand;

			// print process
			count ++;
			if (not (count % 10000)) {
				fprintf(stderr, ".");
			}
			continue;
		}

		if (item == -1) {
			continue;
		}
		vector<string> category;

		// Category for each product is a comma-separated list of strings
		string cat;
		while (getline(ss, cat, ',')) {
			category.push_back(trim(cat));
		}

		if (category[0] != "Clothing Shoes & Jewelry" || category[1] != subcategoryName) {
			continue;
		}

		ct->addPath(category);

		if (category.size() < 4) {
			continue;
		}
		string* categoryP = &(category[0]);
		categoryNode* targetNode = ct->root->find(categoryP, 4);
		if (targetNode == 0) {
			fprintf(stderr, "Can't find the category node.\n");
			exit(1);
		}

		if (nodeIds.find(targetNode->nodeId) == nodeIds.end()) {
			nodeIds[targetNode->nodeId] = nCategory;
			rNodeIds[nCategory] = category[2] + "," + category[3];
			nCategory ++;
		}
		itemCategoryId[item] = nodeIds[targetNode->nodeId];
	}

	fprintf(stderr, "\n");
	in.close();

	int total = 0;
	for (int i = 0; i < nItems; i ++) {
		if (itemCategoryId[i] != -1) {
			total ++;
		}
	}
	fprintf(stderr, "  #Items with category: %d\n", total);
	if (1.0 * total / nItems < 0.5) {
		fprintf(stderr, "So few items are having category info. Sth wrong may have happened.\n");
		exit(1);
	}
}

void model::AUCCX(double* AUC_test, double* std)
{
	double* AUC_u_test = new double[nUsers];
        int* indicator = new int[nUsers];
        for (int u = 0; u < nUsers; u ++) {
            indicator[u] = 1;
        }

        //iterates every user by every item (not in users's rankings)
	#pragma omp parallel for schedule(dynamic)
	for (int u = 0; u < nUsers; u ++) {
		double auc_u = 0.0;
		int n_it = 0;
                for (map<int,long long>::iterator it = test_per_user[u].begin(); it != test_per_user[u].end(); it ++) {
		    int item_test = it->first;	
	            n_it ++;
		    double x_u_test = prediction(u, item_test);
		    int count_test = 0;
		    int max = 0;
		    for (int j = 0; j < nItems; j ++) {
	                if (pos_per_user[u].find(j) != pos_per_user[u].end() || test_per_user[u].find(j) != test_per_user[u].end()) {   
		            continue;
			}
			max ++;
			double x_uj = prediction(u, j);
			if (x_u_test > x_uj) {
		            count_test ++;
			}
		    }
		    auc_u += 1.0 * count_test / max;
		}
		if(n_it == 0) {
		    indicator[u] = 0;
		}
		else {
		    AUC_u_test[u] = 1.0 * auc_u / n_it;
		}
	}

	// sum up AUC
	*AUC_test = 0;
        int u_num = 0;
	for (int u = 0; u < nUsers; u ++) {
	    *AUC_test += AUC_u_test[u];
            if ( indicator[u] == 1)
                u_num ++;
	}
        fprintf(stderr, "%d\n", u_num);
	*AUC_test /= u_num;
        
	// calculate standard deviation
	double variance = 0;
	for (int u = 0; u < nUsers; u ++) {
            if ( indicator[u] == 1) { 
                variance += square(AUC_u_test[u] - *AUC_test);
            }
	}
	*std = sqrt(variance/u_num);

	delete [] AUC_u_test;
}

/*
void model::P(double* test, double* std, int flag)
{       
    int top_K = 50;
    double* u_test = new double[nUsers];
    int* indicator = new int[nUsers];
    int recommend_per_user[nUsers][top_K];

    //iterates every user by every item (not in users's rankings)
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < nUsers; u ++) {
        int* recommend_items = new int[top_K];
        float* recommend_item_values = new float[top_K];

        for (int i = 0; i < top_K; i ++) {
            recommend_items[i] = 0;
            recommend_item_values[i] = -100000000000000000;
        }
        
        for (int j = 0; j < nItems; j ++) {	
            if (pos_per_user[u].find(j) == pos_per_user[u].end()) { // only consider items not in the train dataset
                double x_uj = prediction(u, j);
                // update the recommended items, recommending items with the highest score
                int ind = -1;
                for (int i = 0; i < top_K; i ++) {
		    if (recommend_item_values[i] > x_uj) {
                        break;
		    }
		    else {
                        
		        ind = i;
		    }
	        } 
                if (ind != -1) {
                    for (int l = 0; l < ind; l++) {
                        recommend_items[l] = recommend_items[l+1];
                        recommend_item_values[l] = recommend_item_values[l+1];
                    }
            	    recommend_items[ind] = j;
            	    recommend_item_values[ind] = x_uj;
                }
            }
	}

        // copy recommended items for further model saving
        for (int i = 0; i < top_K; i ++) {
            recommend_per_user[u][i] = recommend_items[i];
        }
        
        // compute a user's precision
        int hit = 0;
        int true_num = 0;
        for (map<int,long long>::iterator it = test_per_user[u].begin(); it != test_per_user[u].end(); it ++) {
	    int item_test = it->first;
            for (int i = 0; i < top_K; i ++) {
	        if (recommend_items[i] == item_test) {
                    hit ++;
		}
	    } 
	    true_num ++;	
	}
        // if there is a user without any item, then label it
	if (true_num == 0){
            indicator[u] = 1;
	}
	else {
            u_test[u] = 1.0 * hit / top_K;
	}  
        // release the memory
        delete [] recommend_items;
        delete [] recommend_item_values;
    }
    
    // if flag==1, save the model (recommened items, purchased items in train dataset and test dataset)
    if (flag == 1){
        // save recommend items
        FILE* f = fopen_("recommend", "w");
        fprintf(f, "{\n");
        for (int u = 0; u < nUsers; u ++) {
            fprintf(f, "%d: [", u);
            for (int i = 0; i<top_K; i++) {
                int item = recommend_per_user[u][i];
                fprintf(f, "%d,", item);
            }
            fprintf(f, "],\n");
        }
        fprintf(f, "}\n");
        fclose(f);
        fprintf(stderr, "\n All recommend items saved.");
        
        // save purchased items in the test dataset
        FILE* f1 = fopen_("true_test", "w");
        fprintf(f1, "{\n");
        for (int u = 0; u < nUsers; u ++) {
            fprintf(f1, "%d: [", u);
            for (map<int,long long>::iterator it = test_per_user[u].begin(); it != test_per_user[u].end(); it ++) {
                int item1 = it->first;
                fprintf(f1, "%d,", item1);
            }
            fprintf(f1, "],\n");
        }
        fprintf(f1, "}\n");
        fclose(f1);
        fprintf(stderr, "\n Purchased items in the test dataset saved.");

        // save purchased items in the train dataset
        FILE* f2 = fopen_("true_train", "w");
        fprintf(f2, "{\n");
        for (int u = 0; u < nUsers; u ++) {
            fprintf(f2, "%d: [", u);
            for (map<int,long long>::iterator it = pos_per_user[u].begin(); it != pos_per_user[u].end(); it ++) {
                int item2 = it->first;
                fprintf(f2, "%d,", item2);
            }
            fprintf(f2, "],\n");
        }
        fprintf(f2, "}\n");
        fclose(f2);
        fprintf(stderr, "\n Purchased items in the train dataset saved.");
    }


    // sum up P
    *test = 0;
    int u_num = 0;
    for (int u = 0; u < nUsers; u ++) {
	if(indicator[u] != 1){
            *test += u_test[u];
            u_num ++;
        }            
	    
    }
    *test /= u_num;
    printf("%d\n", u_num);

    // calculate standard deviation
    double variance = 0;
    for (int u = 0; u < nUsers; u ++) {
        if (indicator[u] != 1)
            variance += square(u_test[u] - *test);
    }
    *std = sqrt(variance/u_num);
    delete [] u_test;
    delete [] indicator;
}

*/


void model::P(double* test, double* std, int flag)
{       
    int top_K = 50;
    double* u_test = new double[nUsers];
    int* indicator = new int[nUsers];
    //fprintf(stderr, "njtu.");
    int** recommend_per_user = new int*[nUsers];
    for (int i = 0; i < nUsers; i++){
        recommend_per_user[i] = new int[top_K];
    }
    //fprintf(stderr, "njtu..");
    //iterates every user by every item (not in users's rankings)
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < nUsers; u ++) {
        int* recommend_items = new int[top_K];
        float* recommend_item_values = new float[top_K];

        for (int i = 0; i < top_K; i ++) {
            recommend_items[i] = 0;
            recommend_item_values[i] = -100000000000000000;
        }
        
        for (int j = 0; j < nItems; j ++) {    
            if (pos_per_user[u].find(j) == pos_per_user[u].end()) { // only consider items not in the train dataset
                double x_uj = prediction(u, j);
                // update the recommended items, recommending items with the highest score
                int ind = -1;
                for (int i = 0; i < top_K; i ++) {
            if (recommend_item_values[i] > x_uj) {
                        break;
            }
            else {
                        
                ind = i;
            }
            } 
                if (ind != -1) {
                    for (int l = 0; l < ind; l++) {
                        recommend_items[l] = recommend_items[l+1];
                        recommend_item_values[l] = recommend_item_values[l+1];
                    }
                    recommend_items[ind] = j;
                    recommend_item_values[ind] = x_uj;
                }
            }
    }

        // copy recommended items for further model saving
        for (int i = 0; i < top_K; i ++) {
            recommend_per_user[u][i] = recommend_items[i];
        }
        
        // compute a user's precision
        int hit = 0;
        int true_num = 0;
        for (map<int,long long>::iterator it = test_per_user[u].begin(); it != test_per_user[u].end(); it ++) {
        int item_test = it->first;
            for (int i = 0; i < top_K; i ++) {
            if (recommend_items[i] == item_test) {
                    hit ++;
        }
        } 
        true_num ++;    
    }
        // if there is a user without any item, then label it
    if (true_num == 0){
            indicator[u] = 1;
    }
    else {
            u_test[u] = 1.0 * hit / top_K;
    }  
        // release the memory
        delete [] recommend_items;
        delete [] recommend_item_values;
    }
    
    // if flag==1, save the model (recommened items, purchased items in train dataset and test dataset)
    if (flag == 1){
        // save recommend items
        FILE* f = fopen_("recommend.txt", "w");
        fprintf(f, "{\n");
        for (int u = 0; u < nUsers; u ++) {
            fprintf(f, "%d: [", u);
            for (int i = 0; i<top_K; i++) {
                int item = recommend_per_user[u][i];
                fprintf(f, "%d,", item);
            }
            fprintf(f, "],\n");
        }
        fprintf(f, "}\n");
        fclose(f);
        fprintf(stderr, "\n All recommend items saved.");
        
        // save purchased items in the test dataset
        FILE* f1 = fopen_("true_test.txt", "w");
        fprintf(f1, "{\n");
        for (int u = 0; u < nUsers; u ++) {
            fprintf(f1, "%d: [", u);
            for (map<int,long long>::iterator it = test_per_user[u].begin(); it != test_per_user[u].end(); it ++) {
                int item1 = it->first;
                fprintf(f1, "%d,", item1);
            }
            fprintf(f1, "],\n");
        }
        fprintf(f1, "}\n");
        fclose(f1);
        fprintf(stderr, "\n Purchased items in the test dataset saved.");

        // save purchased items in the train dataset
        FILE* f2 = fopen_("true_train.txt", "w");
        fprintf(f2, "{\n");
        for (int u = 0; u < nUsers; u ++) {
            fprintf(f2, "%d: [", u);
            for (map<int,long long>::iterator it = pos_per_user[u].begin(); it != pos_per_user[u].end(); it ++) {
                int item2 = it->first;
                fprintf(f2, "%d,", item2);
            }
            fprintf(f2, "],\n");
        }
        fprintf(f2, "}\n");
        fclose(f2);
        fprintf(stderr, "\n Purchased items in the train dataset saved.");
    }


    // sum up P
    *test = 0;
    int u_num = 0;
    for (int u = 0; u < nUsers; u ++) {
    if(indicator[u] != 1){
            *test += u_test[u];
            u_num ++;
        }            
        
    }
    *test /= u_num;
    printf("%d\n", u_num);

    // calculate standard deviation
    double variance = 0;
    for (int u = 0; u < nUsers; u ++) {
        if (indicator[u] != 1)
            variance += square(u_test[u] - *test);
    }
    *std = sqrt(variance/u_num);
    delete [] u_test;
    delete [] indicator;
    for(int i = 0; i < nUsers; i++){
        delete[] recommend_per_user[i];
    }
    delete[] recommend_per_user;
}





void model::AUC(double* AUC_test, double* std)
{
}

void model::AUC_coldItem(double* AUC_test, double* std, int* num_user)
{
}

void model::copyBestModel()
{
	for (int w = 0; w < NW; w ++) {
		bestW[w] = W[w];
	}
}

void model::saveModel(const char* path)
{
	FILE* f = fopen_(path, "w");
	fprintf(f, "{\n");
	fprintf(f, "  \"NW\": %d,\n", NW);

	fprintf(f, "  \"W\": [");
	for (int w = 0; w < NW; w ++) {
		fprintf(f, "%f", bestW[w]);
		if (w < NW - 1) fprintf(f, ", ");
	}
        
	fprintf(f, "]\n");
        fprintf(f, "}\n");
	fclose(f);
	fprintf(stderr, "\nModel saved to %s.\n", path);
}

/// model must be first initialized before calling this function
void model::loadModel(const char* path)
{
	fprintf(stderr, "\n  loading parameters from %s.\n", path);
	ifstream in;
	in.open(path);
	if (! in.good()){
		fprintf(stderr, "Can't read init solution from %s.\n", path);
		exit(1);
	}
	string line;
	string st;
	char ch;
	while(getline(in, line)) {
		stringstream ss(line);
		ss >> st;
		if (st == "\"NW\":") {
			int nw;
			ss >> nw;
			if (nw != NW) {
				fprintf(stderr, "NW not match.");
				exit(1);
			}
			continue;
		}

		if (st == "\"W\":") {
			ss >> ch; // skip '['
			for (int w = 0; w < NW; w ++) {
				if (! (ss >> W[w] >> ch)) {
					fprintf(stderr, "Read W[] error.");
					exit(1);
				}
			}
			break;
		}
	}
	in.close();
}

string model::toString()
{
	return "Empty Model!";
}
