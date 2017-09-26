#include "corpus.hpp"
#include "BPRMF.hpp"
#include "VBPR.hpp"


void go_BPRMF(corpus* corp, int K, double lambda, double biasReg, int iterations, const char* corp_name)
{
	BPRMF md(corp, K, lambda, biasReg);
	md.init();
	md.train(iterations, 0.1);
	md.saveModel((string(corp_name) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}

void go_VBPR(corpus* corp, int K, int K2, double lambda, double lambda2, double biasReg, int iterations, const char* corp_name)
{
	VBPR md(corp, K, K2, lambda, lambda2, biasReg);
	md.init();
	md.train(iterations, 0.01);
	md.saveModel((string(corp_name) + "__" + md.toString() + ".txt").c_str());
	md.cleanUp();
}


int main(int argc, char** argv)
{
	srand(0);
	
	if (argc != 13) {
		printf(" Parameters as following: \n");
		printf(" 1. Whole file path\n");
		printf(" 2. Train file path\n");
		printf(" 3. Test file path\n");
		printf(" 4. Img feature path\n");
		printf(" 5. Latent Feature Dim. (K)\n");
		printf(" 6. Visual Feature Dim. (K')\n");
		//printf(" 7. alpha (for WRMF only)\n");
		printf(" 7. biasReg (regularizer for bias terms)\n");
		printf(" 8. lambda  (regularizer for general terms)\n");
		printf(" 9. lambda2 (regularizer for \"sparse\" terms)\n");
		printf(" 10. #Epoch (number of epochs) \n");
		printf("11. Max #iter \n");
		printf("12. Corpus/Category name under \"Clothing Shoes & Jewelry\" (e.g. Women)\n\n");
		exit(1);
	}

	char* reviewPath = argv[1];
	char* trainPath = argv[2];
	char* testPath = argv[3];
	char* imgFeatPath = argv[4];
	int K  = atoi(argv[5]);
	int K2 = atoi(argv[6]);
	//double alpha = atof(argv[7]);
	double biasReg = atof(argv[7]);
	double lambda = atof(argv[8]);
	double lambda2 = atof(argv[9]);
	double nEpoch = atoi(argv[10]);
	int iter = atoi(argv[11]);
	char* corp_name = argv[12];

	fprintf(stderr, "{\n");
	//fprintf(stderr, "  \"corpus\": \"%s\",\n", reviewPath);

	//corpus bprCorp;
	//bprCorp.loadData(reviewPath, trainPath, testPath, imgFeatPath, -1, -1, 0);
        //corpus trainCorp;
	//trainCorp.loadData(trainPath, imgFeatPath, -1, -1);
	//corpus testCorp;
	//testCorp.loadData(testPath, imgFeatPath, -1, -1);
        //go_BPRMF(&bprCorp, K, lambda, biasReg, iter, corp_name);
        //bprCorp.cleanUp();
        //fprintf(stderr, "}\n");


        corpus vbprCorp;
        vbprCorp.loadData(reviewPath, trainPath, testPath, imgFeatPath, -1, -1, 1);
        go_VBPR(&vbprCorp, K, K2, lambda, lambda2, biasReg, iter, corp_name);
	
	vbprCorp.cleanUp();
	fprintf(stderr, "}\n");

	fflush(stderr);

	return 0;
}
