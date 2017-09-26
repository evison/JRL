#include "corpus.hpp"
#include "BPRMF.hpp"



void go_BPRMF(corpus* corp, corpus* trainCorp, corpus* validCorp, corpus* testCorp, int K, double lambda, double biasReg, int iterations, const char* corp_name,  double learn_rate)
{
	BPRMF md(corp, K, lambda, biasReg);
  //load here
	md.init();
	md.train(iterations, learn_rate);
	double valid, test, std;
	md.AUC(&valid, &test, &std);
  char score[8];
  sprintf(score,"%f",test);
	md.saveModel((string(corp_name) + "__" + md.toString() + "="+score + ".txt").c_str());  
  
	md.cleanUp();
}


int main(int argc, char** argv)
{
	srand(0);



	const char* reviewPath = "simple_out.gz"; 
	int K  = 20;
	int iter = 10;
	const char* corp_name = "Clothing";
  
  
	corpus corp;
	corp.loadData(reviewPath, "", 5, 0);
  
  double learning_rates[4]={0.005, 0.05, 0.5, 1.0};

  #pragma omp parallel for schedule(dynamic)
  for(int biasReg=0; biasReg<10; biasReg++){
      for(int lambda=0; lambda<10; lambda++){
        for(int i=0; i<3; i++){
          double learn_rate = learning_rates[i];
          go_BPRMF(&corp, K, lambda, biasReg, iter, corp_name, learn_rate);
        }

      }
  }



	corp.cleanUp();
	fflush(stderr);

	return 0;
}
