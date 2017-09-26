#pragma once

#include "BPRMF.hpp"

class VBPR : public BPRMF 
{
public:
	VBPR(corpus* corp, int K, int K2, double lambda, double lambda2, double biasReg) 
		: BPRMF(corp, K, lambda, biasReg) 
		, K2(K2)
		, lambda2(lambda2) {}
	
	~VBPR(){}
	
	void init();
	void cleanUp();

	void getParametersFromVector(	double*   g, 
									double**  beta_item,
									double*** gamma_user, 
									double*** gamma_item,
									double*** theta_user,
									double*** U,
									double**  beta_cnn,
									action_t  action);
	
	void getVisualFactors();
	double prediction(int user, int item);

	void train(int iterations, double learn_rate);
	void updateFactors(int user_id, int pos_item_id, int neg_item_id, double learn_rate);

	string toString();

	/* hyper-parameters */
	int K2;
	double lambda2;

	/* auxiliary variables */
	double** theta_user;
	double*  beta_cnn;

	/* Model parameters */
	double** U;  	// embedding matrix (K2 by 4096)

	/* for speep-up */
	double** theta_item;
	double*  beta_item_visual;
};
