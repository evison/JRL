#include "VBPR.hpp"

void VBPR::init()
{
	NW = K * nUsers + (K + 1) * nItems 	// latent factors
		 + K2 * nUsers + K2 * corp->imFeatureDim  // visual factors
		 + corp->imFeatureDim; // beta_cnn

	W = new double[NW];
	bestW = new double[NW];

	// parameter initialization
	for (int w = 0; w < NW; w ++) {
		if (w < nItems || w >= NW - corp->imFeatureDim) {
			W[w] = 0;
		} else {
			W[w] = 1.0 * rand() / RAND_MAX;
		}
	}
	
	getParametersFromVector(W, &beta_item, &gamma_user, &gamma_item, &theta_user, &U, &beta_cnn, INIT);

	/* for speed up */
	beta_item_visual = new double [nItems];
	theta_item = new double* [nItems];
	for (int i = 0; i < nItems; i ++) {
		theta_item[i] = new double [K2];
	}
}

void VBPR::cleanUp()
{
	getParametersFromVector(W, &beta_item, &gamma_user, &gamma_item, &theta_user, &U, &beta_cnn, FREE);

	delete [] W;
	delete [] bestW;

	for (int i = 0; i < nItems; i ++) {
		delete [] theta_item[i];
	}
	delete [] theta_item;
	delete [] beta_item_visual;
}

void VBPR::getParametersFromVector(	double*   g,
									double**  beta_item,
									double*** gamma_user, 
									double*** gamma_item,
									double*** theta_user,
									double*** U,
									double**  beta_cnn,
									action_t action)
{
	if (action == FREE) {
		delete [] (*gamma_user);
		delete [] (*gamma_item);
		delete [] (*theta_user);
		delete [] (*U);
		return;
	}

	if (action == INIT)	{
		*gamma_user = new double* [nUsers];
		*gamma_item = new double* [nItems];
		*theta_user = new double* [nUsers];
		*U = new double* [K2];
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

	for (int u = 0; u < nUsers; u ++) {
		(*theta_user)[u] = g + ind;
		ind += K2;
	}

	for (int k = 0; k < K2; k ++) {
		(*U)[k] = g + ind;
		ind += corp->imFeatureDim;
	}

	*beta_cnn = g + ind;
	ind += corp->imFeatureDim;

	if (ind != NW) {
		printf("Got bad index (VBPR.cpp, line %d)", __LINE__);
		exit(1);
	}
}

void VBPR::getVisualFactors()
{
	#pragma omp parallel for
	for (int x = 0; x < nItems; x ++) {
		// cnn
		vector<pair<int, float> >& feat = corp->imageFeatures[x];
		
		// visual factors
		for (int k = 0; k < K2; ++k) {
			theta_item[x][k] = 0;  // embed features to K2-dim		
			for (unsigned i = 0; i < feat.size(); i ++) {
				theta_item[x][k] += U[k][feat[i].first] * feat[i].second;
			}
		}

		// visual bias
		beta_item_visual[x] = 0;
		for (unsigned i = 0; i < feat.size(); i ++) {
			beta_item_visual[x] += beta_cnn[feat[i].first] * feat[i].second;
		}
	}
}

double VBPR::prediction(int user, int item)
{
	return  beta_item[item]
			+ inner(gamma_user[user], gamma_item[item], K)
			+ inner(theta_item[item], theta_user[user], K2)
			+ beta_item_visual[item];
}

void VBPR::updateFactors(int user_id, int pos_item_id, int neg_item_id, double learn_rate)
{
	// sparse representation of f_i - f_j
	vector<pair<int, float> > diff;
	vector<pair<int, float> >& feat_i = corp->imageFeatures[pos_item_id];
	vector<pair<int, float> >& feat_j = corp->imageFeatures[neg_item_id];
	unsigned p_i = 0, p_j = 0;
	while (p_i < feat_i.size() && p_j < feat_j.size()) {
		int ind_i = feat_i[p_i].first;
		int ind_j = feat_j[p_j].first;
		if (ind_i < ind_j) {
			diff.push_back(make_pair(ind_i, feat_i[p_i].second));
			p_i ++;
		} else if (ind_i > ind_j) {
			diff.push_back(make_pair(ind_j, - feat_j[p_j].second));
			p_j ++;
		} else {
			diff.push_back(make_pair(ind_i, feat_i[p_i].second - feat_j[p_j].second));
			p_i ++; p_j ++;
		}
	}
	while (p_i < feat_i.size()) {
		diff.push_back(feat_i[p_i]);
		p_i ++;
	}
	while (p_j < feat_j.size()) {
		diff.push_back(make_pair(feat_j[p_j].first, - feat_j[p_j].second));
		p_j ++;
	}

	// U * (x_i - x_j)
	for (int r = 0; r < K2; ++ r) {
		theta_item[0][r] = 0;  // borrow the memory at index 0		
		for (unsigned ind = 0; ind < diff.size(); ind ++) {
			int c = diff[ind].first;
			theta_item[0][r] += U[r][c] * diff[ind].second;
		}
	}

	// visual bias
	double visual_bias = 0;
	for (unsigned ind = 0; ind < diff.size(); ind ++) {
		int c = diff[ind].first;
		visual_bias += beta_cnn[c] * diff[ind].second;
	}

	// x_uij = prediction(user_id, pos_item_id) - prediction(user_id, neg_item_id);
	double x_uij = beta_item[pos_item_id] - beta_item[neg_item_id];
	x_uij += inner(gamma_user[user_id], gamma_item[pos_item_id], K) - inner(gamma_user[user_id], gamma_item[neg_item_id], K);
	x_uij += inner(theta_user[user_id], theta_item[0], K2);
	x_uij += visual_bias;

	double deri = 1 / (1 + exp(x_uij));

	beta_item[pos_item_id] += learn_rate * ( deri - biasReg * beta_item[pos_item_id]);
	beta_item[neg_item_id] += learn_rate * (-deri - biasReg * beta_item[neg_item_id]);

	// adjust latent factors
	for (int f = 0; f < K; f ++) {
		double w_uf = gamma_user[user_id][f];
		double h_if = gamma_item[pos_item_id][f];
		double h_jf = gamma_item[neg_item_id][f];
 
		gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - lambda * w_uf);
		gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - lambda * h_if);
		gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - lambda / 10.0 * h_jf);
	}

	// adjust visual factors
	for (int f = 0; f < K2; f ++) {
		for (unsigned ind = 0; ind < diff.size(); ind ++) {
			int c = diff[ind].first;
			U[f][c] += learn_rate * (deri * theta_user[user_id][f] * diff[ind].second - lambda2 * U[f][c]);
		}
		theta_user[user_id][f] += learn_rate * (deri * theta_item[0][f] - lambda * theta_user[user_id][f]);
	}

	// adjust visual bias
	for (unsigned ind = 0; ind < diff.size(); ind ++) {
		int c = diff[ind].first;
		beta_cnn[c] += learn_rate * (deri * diff[ind].second - lambda2 * beta_cnn[c]);
	}
}

void VBPR::train(int iterations, double learn_rate)
{
	fprintf(stderr, "%s", ("\n<<< " + toString() + " >>>\n\n").c_str());

	double bestValidAUC = -1;
    int best_iter = -1;

	// SGD begins
	for (int iter = 1; iter <= iterations; iter ++) {
		
		// perform one iter of SGD
		double l_dlStart = clock_();
		oneiteration(learn_rate);
		fprintf(stderr, "Iter: %d, took %f\n", iter, clock_() - l_dlStart);

		if (iter % 1 == 0) {
			getVisualFactors(); // essential to get correct prediction

			double test, std;
			P(&test, &std,0);
			fprintf(stderr, "Test P = %f, Test Std = %f\n", test, std);
			
			if (bestValidAUC < test) {
				bestValidAUC = test;
				best_iter = iter;
				copyBestModel();				
			} else if (test < bestValidAUC && iter >= best_iter + 20) {
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
	P(&test, &std,1);
	fprintf(stderr, "\n\n === [All Item]: Test P = %f, Test Std = %f\n", test, std);
	
	
}

string VBPR::toString()
{
	char str[10000];
	sprintf(str, "VBPR__K_%d_K2_%d_lambda_%.2f_lambda2_%.6f_biasReg_%.2f", K, K2, lambda, lambda2, biasReg);
	return str;
}
