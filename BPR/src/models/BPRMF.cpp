#include "BPRMF.hpp"

void BPRMF::init()
{
	NW = nItems + K  * (nUsers + nItems); // total count of bias and latent factors -> count of all parameters
	W = new double[NW]; // contigious array of parameters
	bestW = new double[NW];

  //check to see if this is a pre-trained instatance loaded from disk
	getParametersFromVector(W, &beta_item, &gamma_user, &gamma_item, INIT);

  //initialize latent factor vectors gamma_u & gamma_i
  //matrix dim: user x K (latent factor dimension [20])
	for (int u = 0; u < nUsers; u ++) {
		for (int k = 0; k < K; k ++) {
			gamma_user[u][k] = rand() * 1.0 / RAND_MAX;
                        //printf("%f ",gamma_user[u][k]);
		}
	}
	for (int i = 0; i < nItems; i ++) {
		beta_item[i] = 0;
		for (int k = 0; k < K; k ++) {
			gamma_item[i][k] = rand() * 1.0 / RAND_MAX;
		}
	}
}

void BPRMF::cleanUp()
{
	getParametersFromVector(W, &beta_item, &gamma_user, &gamma_item, FREE);
	
	delete [] W;
	delete [] bestW;
}

void BPRMF::getParametersFromVector(	double*   g,
										double**  beta_item, 
										double*** gamma_user, 
										double*** gamma_item,
										action_t  action)
{
	if (action == FREE) {
		delete [] (*gamma_user);
		delete [] (*gamma_item);
		return;
	}

	if (action == INIT)	{
		*gamma_user = new double* [nUsers];
		*gamma_item = new double* [nItems];
	}

	int ind = 0;

	*beta_item = g + ind;
	ind += nItems;

	for (int u = 0; u < nUsers; u ++) {
		(*gamma_user)[u] = g + ind;
		ind += K;
	}
	for (int i = 0; i < nItems; i ++) {
		(*gamma_item)[i] = g + ind;
		ind += K;
	}

	if (ind != NW) {
		printf("Got bad index (BPRMF.cpp, line %d)", __LINE__);
		exit(1);
	}
}

double BPRMF::prediction(int user, int item)
{
	return beta_item[item] + inner(gamma_user[user], gamma_item[item], K);
}

int BPRMF::sampleUser()
{
	while (true) {
		int user_id = rand() % nUsers; //random user_id
    // try another id if this user doesn't have any ratings or if # of ratings is size of all items???
		if (pos_per_user[user_id].size() == 0 || (int) pos_per_user[user_id].size() == nItems) {
			continue;
		}
		return user_id;
	}
}

//HERE is where the gradent is taken
// tuple u,i,j are first 3 args
void BPRMF::updateFactors(int user_id, int pos_item_id, int neg_item_id, double learn_rate)
{
  //equation 7
	double x_uij = beta_item[pos_item_id] - beta_item[neg_item_id];
        //double x_uij = 0.0;
	x_uij += inner(gamma_user[user_id], gamma_item[pos_item_id], K) - inner(gamma_user[user_id], gamma_item[neg_item_id], K);

	double deri = 1 / (1 + exp(x_uij));
  //eq 8
	beta_item[pos_item_id] += learn_rate * (deri - biasReg * beta_item[pos_item_id]);
	beta_item[neg_item_id] += learn_rate * (-deri - biasReg * beta_item[neg_item_id]);

	// adjust latent factors
	for (int f = 0; f < K; f ++) {
		double w_uf = gamma_user[user_id][f];     // w_uf - latent factors for user
		double h_if = gamma_item[pos_item_id][f]; // h_if - latent factors for pos item
		double h_jf = gamma_item[neg_item_id][f]; // h_jf - letent factors for neg item

		gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - lambda * w_uf);
		gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - lambda * h_if);
		gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - lambda / 1.0 * h_jf);
	}
}

//this is the the gradient routine called from inner loop of train method
//this whole routine iterates n_votes times
//so once for every feedback observation, we sample a random user u.
//then we sample a reandom positive item from u
//then we sample a random negative item from u (item uesr hasnt' bought)
//then for that triple we run the GD routine
void BPRMF::oneiteration(double learn_rate)
{
	// uniformally sample users in order to approximatelly optimize AUC for all users
	int user_id, pos_item_id, neg_item_id;

	// working memory
	//vector<int>* user_matrix = new vector<int> [nUsers];
  
        //for each user u
        //  for k,v in pos_per_user
        //    user_matrix[u].append(k) // fills user list w/ item ids
	//for (int u = 0; u < nUsers; u ++) {
	//	for (map<int,long long>::iterator it = pos_per_user[u].begin(); it != pos_per_user[u].end(); it ++) {
	//		user_matrix[u].push_back(it->first);
	//	}
	//}

	// now it begins!
        //iterate all implicit ratings
        /*
	for (int i = 0; i < 2*num_pos_events; i++) {
		
		// sample user (random)
		user_id = sampleUser();
		vector<int>& user_items = user_matrix[user_id]; //get item ids from routine above

		// reset user if already exhausted
		if (user_items.size() == 0) {
			for (map<int,long long>::iterator it = pos_per_user[user_id].begin(); it != pos_per_user[user_id].end(); it ++) {
				user_items.push_back(it->first);
			}
		}

		// sample positive item and remove it from this list
		int rand_num = rand() % user_items.size(); //choose random item in users items
		pos_item_id = user_items.at(rand_num);
		user_items.at(rand_num) = user_items.back();
		user_items.pop_back();

		// sample negative item by choosing random item id and seeing if it's in the positive
                //if not, continue
		do {
			neg_item_id = rand() % nItems;
		} while (pos_per_user[user_id].find(neg_item_id) != pos_per_user[user_id].end());

		// now got tuple (user_id, pos_item, neg_item)
                / /check the gradient
		updateFactors(user_id, pos_item_id, neg_item_id, learn_rate);
	}*/
        
    for (int u = 0; u < nUsers; u ++){
        user_id = u;
        for (map<int,long long>::iterator it = pos_per_user[u].begin(); it != pos_per_user[u].end(); it ++) {
            pos_item_id = it->first;
            do {
                neg_item_id = rand() % nItems;
            } while (pos_per_user[user_id].find(neg_item_id) != pos_per_user[user_id].end());
            updateFactors(user_id, pos_item_id, neg_item_id, learn_rate);
        }
    }             

    //delete [] user_matrix;
}

void BPRMF::train(int iterations, double learn_rate)
{
	fprintf(stderr, "%s", ("\n<<< " + toString() + " >>>\n\n").c_str());

	double bestValidAUC = -1;
	int best_iter = 0;

	// SGD begins
	for (int iter = 1; iter <= iterations; iter ++) {
		
		// perform one iter of SGD
		double l_dlStart = clock_();
		oneiteration(learn_rate);
		fprintf(stderr, "Iter: %d, took %f\n", iter, clock_() - l_dlStart);

		if(iter % 1 == 0) {
			double test, std;
			P(&test, &std, 0);
                        //copyBestModel();
			fprintf(stderr, "Test P = %f(%f), Test Std = %f\n", test, bestValidAUC,  std);
			
                        //AUC should increase each iteration or else cancel
			if (bestValidAUC < test) {
				bestValidAUC = test;
				best_iter = iter;
				copyBestModel();
			} else if (test < bestValidAUC && iter > best_iter + 50) {
				fprintf(stderr, "Overfitted. Exiting... \n");
				break;
			}
		}
	}

	// copy back best parameters
	for (int w = 0; w < NW; w ++) {
		W[w] = bestW[w];
	}

	double test, std;
	P(&test, &std, 1);
	fprintf(stderr, "\n\n <<< BPR-MF >>> Test P = %f, Test Std = %f\n", test, std);

}

string BPRMF::toString()
{
	char str[10000];
	sprintf(str, "BPR-MF__K_%d_lambda_%.2f_biasReg_%.2f", K, lambda, biasReg);
	return str;
}
