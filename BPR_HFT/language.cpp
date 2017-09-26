#include "common.hpp"
#include "vector"
#include "map"
#include "limits"
#include "omp.h"
#include "lbfgs.h"
#include "sys/time.h"
#include "language.hpp"


using namespace std;

inline double square(double x)
{
  return x * x;
}

inline double dsquare(double x)
{
  return 2 * x;
}

double clock_()
{
  timeval tim;
  gettimeofday(&tim, NULL);
  return tim.tv_sec + (tim.tv_usec / 1000000.0);
}

/// Recover all parameters from a vector (g)
int topicCorpus::getG(double* g,
                      double** alpha,
                      double** kappa,
                      double** beta_user,
                      double** beta_beer,
                      double*** gamma_user,
                      double*** gamma_beer,
                      double*** topicWords,
                      bool init)
{
  if (init)
  {
    *gamma_user = new double*[nUsers];
    *gamma_beer = new double*[nBeers];
    *topicWords = new double*[nWords];
  }

  int ind = 0;
  *alpha = g + ind;
  ind++;
  *kappa = g + ind;
  ind++;

  *beta_user = g + ind;
  ind += nUsers;
  *beta_beer = g + ind;
  ind += nBeers;

  for (int u = 0; u < nUsers; u++)
  {
    (*gamma_user)[u] = g + ind;
    ind += K;
  }
  for (int b = 0; b < nBeers; b++)
  {
    (*gamma_beer)[b] = g + ind;
    ind += K;
  }
  for (int w = 0; w < nWords; w++)
  {
    (*topicWords)[w] = g + ind;
    ind += K;
  }

  if (ind != NW)
  {
    printf("Got incorrect index at line %d\n", __LINE__);
    exit(1);
  }
  return ind;
}

/// Free parameters
void topicCorpus::clearG(double** alpha,
                         double** kappa,
                         double** beta_user,
                         double** beta_beer,
                         double*** gamma_user,
                         double*** gamma_beer,
                         double*** topicWords)
{
  delete[] (*gamma_user);
  delete[] (*gamma_beer);
  delete[] (*topicWords);
}

/// Compute energy
static lbfgsfloatval_t evaluate(void *instance,
                                const lbfgsfloatval_t *x,
                                lbfgsfloatval_t *g,
                                const int n,
                                const lbfgsfloatval_t step)
{
  topicCorpus* ec = (topicCorpus*) instance;

  for (int i = 0; i < ec->NW; i++)
    ec->W[i] = x[i];

  double* grad = new double[ec->NW];
  ec->dl(grad);
  for (int i = 0; i < ec->NW; i++)
    g[i] = grad[i];
  delete[] grad;

  lbfgsfloatval_t fx = ec->lsq();
  return fx;
}

static int progress(void *instance,
                    const lbfgsfloatval_t *x,
                    const lbfgsfloatval_t *g,
                    const lbfgsfloatval_t fx,
                    const lbfgsfloatval_t xnorm,
                    const lbfgsfloatval_t gnorm,
                    const lbfgsfloatval_t step,
                    int n,
                    int k,
                    int ls)
{
  static double gtime = clock_();
  printf(".");
  fflush(stdout);
  double tdiff = clock_();
  gtime = tdiff;
  return 0;
}

/// Predict a particular rating given the current parameter values
double topicCorpus::prediction(vote* vi)
{
  int user = vi->user;
  int beer = vi->item;
  double res = *alpha + beta_user[user] + beta_beer[beer];
  for (int k = 0; k < K; k++)
    res += gamma_user[user][k] * gamma_beer[beer][k];
  return res;
}

double topicCorpus::prediction_ui(int user, int beer)
{
  double res = *alpha + beta_user[user] + beta_beer[beer];
  for (int k = 0; k < K; k++)
    res += gamma_user[user][k] * gamma_beer[beer][k];
  return res;
}

double topicCorpus::prediction_ui_bpr(int user, int beer)
{
  double res = beta_beer[beer];
  for (int k = 0; k < K; k++)
    res += gamma_user[user][k] * gamma_beer[beer][k];
  return res;
}



/// Compute normalization constant for a particular item
void topicCorpus::topicZ(int beer, double& res)
{
  res = 0;
  for (int k = 0; k < K; k++)
    res += exp(*kappa * gamma_beer[beer][k]);
}

/// Compute normalization constants for all K topics
void topicCorpus::wordZ(double* res)
{
  for (int k = 0; k < K; k++)
  {
    res[k] = 0;
    for (int w = 0; w < nWords; w++)
      res[k] += exp(backgroundWords[w] + topicWords[w][k]);
  }
}

/// Update topic assignments for each word. If sample==true, this is done by sampling, otherwise it's done by maximum likelihood (which doesn't work very well)
void topicCorpus::updateTopics(bool sample)
{
  double updateStart = clock_();

  for (int x = 0; x < (int) trainVotes.size(); x++)
  {
    if (x > 0 and x % 100000 == 0)
    {
      printf(".");
      fflush(stdout);
    }
    vote* vi = trainVotes[x];
    int beer = vi->item;

    int* topics = wordTopics[vi];

    for (int wp = 0; wp < (int) vi->words.size(); wp++)
    { // For each word position
      int wi = vi->words[wp]; // The word
      double* topicScores = new double[K];
      double topicTotal = 0;
      for (int k = 0; k < K; k++)
      {
        topicScores[k] = exp(*kappa * gamma_beer[beer][k] + backgroundWords[wi] + topicWords[wi][k]);
        topicTotal += topicScores[k];
      }

      for (int k = 0; k < K; k++)
        topicScores[k] /= topicTotal;

      int newTopic = 0;
      if (sample)
      {
        double x = rand() * 1.0 / (1.0 + RAND_MAX);
        while (true)
        {
          x -= topicScores[newTopic];
          if (x < 0)
            break;
          newTopic++;
        }
      }
      else
      {
        double bestScore = -numeric_limits<double>::max();
        for (int k = 0; k < K; k++)
          if (topicScores[k] > bestScore)
          {
            bestScore = topicScores[k];
            newTopic = k;
          }
      }
      delete[] topicScores;

      if (newTopic != topics[wp])
      { // Update topic counts if the topic for this word position changed
        {
          int t = topics[wp];
          wordTopicCounts[wi][t]--;
          wordTopicCounts[wi][newTopic]++;
          topicCounts[t]--;
          topicCounts[newTopic]++;
          beerTopicCounts[beer][t]--;
          beerTopicCounts[beer][newTopic]++;
          topics[wp] = newTopic;
        }
      }
    }
  }
  printf("\n");
}


void topicCorpus::generate_train_samples()
{ 
  int user_id = 0;
  int pos_item_id = 0;
  int neg_item_id = 0;
  for (int u = 0; u < nUsers; u ++){
    user_id = u;
    for (map<int,long long>::iterator it = train_per_user[u].begin(); it != train_per_user[u].end(); it ++) {
        pos_item_id = it->first;
        do {
            neg_item_id = rand() % nBeers;
        } while (train_per_user[user_id].find(neg_item_id) != train_per_user[user_id].end());
        train_samples[user_id][pos_item_id] = neg_item_id;
    }
  }          
}



void topicCorpus::generate_loss_samples()
{
  int user_id = 0;
  int pos_item_id = 0;
  int neg_item_id = 0;
  for (int u = 0; u < nUsers; u ++){
    user_id = u;
    for (map<int,long long>::iterator it = train_per_user[u].begin(); it != train_per_user[u].end(); it ++) {
        pos_item_id = it->first;
        do {
            neg_item_id = rand() % nBeers;
        } while (train_per_user[user_id].find(neg_item_id) != train_per_user[user_id].end());
        loss_samples[user_id][pos_item_id] = neg_item_id;
    }
  }
}



/// Derivative of the energy function
void topicCorpus::dl(double* grad)
{
  double dlStart = clock_();

  for (int w = 0; w < NW; w ++)
    grad[w] = 0;

  double* dalpha;
  double* dkappa;
  double* dbeta_user;
  double* dbeta_beer;
  double** dgamma_user;
  double** dgamma_beer;
  double** dtopicWords;
  
  getG(grad, &(dalpha), &(dkappa), &(dbeta_user), &(dbeta_beer), &(dgamma_user), &(dgamma_beer), &(dtopicWords), true);
  /*
  double da = 0;
#pragma omp parallel for reduction(+:da)
  for (int u = 0; u < nUsers; u ++)
  {
    for (vector<vote*>::iterator it = trainVotesPerUser[u].begin(); it != trainVotesPerUser[u].end(); it ++)
    {
      vote* vi = *it;
      double p = prediction(vi);
      double dl = dsquare(p - vi->value);

      da += dl;
      dbeta_user[u] += dl;
      for (int k = 0; k < K; k++)
        dgamma_user[u][k] += dl * gamma_beer[vi->item][k];
    }
  }
  (*dalpha) = da;

#pragma omp parallel for
  for (int b = 0; b < nBeers; b ++)
  {
    for (vector<vote*>::iterator it = trainVotesPerBeer[b].begin(); it != trainVotesPerBeer[b].end(); it ++)
    {
      vote* vi = *it;
      double p = prediction(vi);
      double dl = dsquare(p - vi->value);

      dbeta_beer[b] += dl;
      for (int k = 0; k < K; k++)
        dgamma_beer[b][k] += dl * gamma_user[vi->user][k];
    }
  }
  */
    
  generate_train_samples();
  double da = 0;
  double learn_rate = 1;
  double biasReg = 0.01;
  int user_id = 0;
  int pos_item_id = 0;
  int neg_item_id = 0;
  //#pragma omp parallel for 
  for (int u = 0; u < nUsers; u ++){
        user_id = u;
        
        for (map<int,int>::iterator it = train_samples[user_id].begin(); it != train_samples[user_id].end(); it ++) {
            
            pos_item_id = it->first;
            //a = it->second;
             
            neg_item_id = it->second;
            //printf("%d,%d,%d", user_id, pos_item_id, neg_item_id);       
            double x_uij = prediction_ui_bpr(user_id, pos_item_id) - prediction_ui_bpr(user_id, neg_item_id);
            //double x_uij = 0.0;
            //x_uij += inner(gamma_user[user_id], gamma_beer[pos_item_id], K) - inner(gamma_user[user_id], gamma_beer[neg_item_id], K);

            double deri = 1 / (1 + exp(x_uij));
  
            dbeta_beer[pos_item_id] -= learn_rate * (deri - biasReg * beta_beer[pos_item_id]);
            dbeta_beer[neg_item_id] -= learn_rate * (-deri - biasReg * beta_beer[neg_item_id]);
            
            // adjust latent factors
            for (int f = 0; f < K; f ++) {
                double w_uf = gamma_user[user_id][f];     // w_uf - latent factors for user
                double h_if = gamma_beer[pos_item_id][f]; // h_if - latent factors for pos item
                double h_jf = gamma_beer[neg_item_id][f]; // h_jf - letent factors for neg item

                dgamma_user[user_id][f]     -= learn_rate * ( deri * (h_if - h_jf) );
                dgamma_beer[pos_item_id][f] -= learn_rate * ( deri * w_uf );
                dgamma_beer[neg_item_id][f] -= learn_rate * (-deri * w_uf );
            }
            
        }
        
    }
    //printf("bpr optimization end");             




  double dk = 0;
#pragma omp parallel for reduction(+:dk)
  for (int b = 0; b < nBeers; b++)
  {
    double tZ;
    topicZ(b, tZ);

    for (int k = 0; k < K; k++)
    {
      double q = -lambda * (beerTopicCounts[b][k] - beerWords[b] * exp(*kappa * gamma_beer[b][k]) / tZ);
      dgamma_beer[b][k] += *kappa * q;
      dk += gamma_beer[b][k] * q;
    }
  }
  (*dkappa) = dk;

  // Add the derivative of the regularizer
  if (latentReg > 0)
  {
    for (int u = 0; u < nUsers; u++)
      for (int k = 0; k < K; k++)
        dgamma_user[u][k] += latentReg * dsquare(gamma_user[u][k]);
    for (int b = 0; b < nBeers; b++)
      for (int k = 0; k < K; k++)
        dgamma_beer[b][k] += latentReg * dsquare(gamma_beer[b][k]);
  }

  double* wZ = new double[K];
  wordZ(wZ);

#pragma omp parallel for
  for (int w = 0; w < nWords; w++)
    for (int k = 0; k < K; k++)
    {
      int twC = wordTopicCounts[w][k];
      double ex = exp(backgroundWords[w] + topicWords[w][k]);
      dtopicWords[w][k] += -lambda * (twC - topicCounts[k] * ex / wZ[k]);
    }

  delete[] wZ;
  clearG(&(dalpha), &(dkappa), &(dbeta_user), &(dbeta_beer), &(dgamma_user), &(dgamma_beer), &(dtopicWords));
}





/// Compute the energy according to the least-squares criterion

double topicCorpus::lsq()
{
  double lsqStart = clock_();
  double res = 0;
  int user_id = 0;
  int pos_item_id = 0;
  int neg_item_id = 0;

  /*
  #pragma omp parallel for reduction(+:res)
  for (int x = 0; x < (int) trainVotes.size(); x++)
  {
    vote* vi = trainVotes[x];
    res += square(prediction(vi) - vi->value);
  }
  */
  for (int u = 0; u < nUsers; u ++){
        user_id = u;

        for (map<int,int>::iterator it = loss_samples[user_id].begin(); it != loss_samples[user_id].end(); it ++) {
            pos_item_id = it->first;
            neg_item_id = it->second;
            double x_uij = prediction_ui_bpr(user_id, pos_item_id) - prediction_ui_bpr(user_id, neg_item_id);
            double deri = 1 / (1 + exp(x_uij));
            res -= deri;
        }
    }




  for (int b = 0; b < nBeers; b++)
  {
    double tZ;
    topicZ(b, tZ);
    double lZ = log(tZ);

    for (int k = 0; k < K; k++)
      res += -lambda * beerTopicCounts[b][k] * (*kappa * gamma_beer[b][k] - lZ);
  }





  // Add the regularizer to the energy
  if (latentReg > 0)
  {
    for (int u = 0; u < nUsers; u++)
      for (int k = 0; k < K; k++)
        res += latentReg * square(gamma_user[u][k]);
    for (int b = 0; b < nBeers; b++)
      for (int k = 0; k < K; k++)
        res += latentReg * square(gamma_beer[b][k]);
  }

  double* wZ = new double[K];
  wordZ(wZ); 
  for (int k = 0; k < K; k++)
  {
    double lZ = log(wZ[k]);
    for (int w = 0; w < nWords; w++)
      res += -lambda * wordTopicCounts[w][k] * (backgroundWords[w] + topicWords[w][k] - lZ);
  }
  delete[] wZ;

  double lsqEnd = clock_();

  return res;
}










/*

double topicCorpus::lsq()
{
  double lsqStart = clock_();
  double res = 0;

#pragma omp parallel for reduction(+:res)
  for (int x = 0; x < (int) trainVotes.size(); x++)
  {
    vote* vi = trainVotes[x];
    res += square(prediction(vi) - vi->value);
  }

  for (int b = 0; b < nBeers; b++)
  {
    double tZ;
    topicZ(b, tZ);
    double lZ = log(tZ);

    for (int k = 0; k < K; k++)
      res += -lambda * beerTopicCounts[b][k] * (*kappa * gamma_beer[b][k] - lZ);
  }





  // Add the regularizer to the energy
  if (latentReg > 0)
  {
    for (int u = 0; u < nUsers; u++)
      for (int k = 0; k < K; k++)
        res += latentReg * square(gamma_user[u][k]);
    for (int b = 0; b < nBeers; b++)
      for (int k = 0; k < K; k++)
        res += latentReg * square(gamma_beer[b][k]);
  }

  double* wZ = new double[K];
  wordZ(wZ);
  for (int k = 0; k < K; k++)
  {
    double lZ = log(wZ[k]);
    for (int w = 0; w < nWords; w++)
      res += -lambda * wordTopicCounts[w][k] * (backgroundWords[w] + topicWords[w][k] - lZ);
  }
  delete[] wZ;

  double lsqEnd = clock_();

  return res;
}
*/
/// Compute the average and the variance
void averageVar(vector<double>& values, double& av, double& var)
{
  double sq = 0;
  av = 0;
  for (vector<double>::iterator it = values.begin(); it != values.end(); it++)
  {
    av += *it;
    sq += (*it) * (*it);
  }
  av /= values.size();
  sq /= values.size();
  var = sq - av * av;
}


void topicCorpus::precision(double* test, double* std, int flag)
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
        
        for (int j = 0; j < nBeers; j ++) {    
            if (train_per_user[u].find(j) == train_per_user[u].end()) { // only consider items not in the train dataset
                double x_uj = prediction_ui_bpr(u, j);
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
            for (map<int,long long>::iterator it = train_per_user[u].begin(); it != train_per_user[u].end(); it ++) {
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



/// Compute the validation and test error (and testing standard error)
void topicCorpus::validTestError(double& train, double& valid, double& test, double& testSte)
{
  train = 0;
  valid = 0;
  test = 0;
  testSte = 0;

  map<int, vector<double> > errorVsTrainingUser;
  map<int, vector<double> > errorVsTrainingBeer;

  for (vector<vote*>::iterator it = trainVotes.begin(); it != trainVotes.end(); it++)
    train += square(prediction(*it) - (*it)->value);
  for (vector<vote*>::iterator it = validVotes.begin(); it != validVotes.end(); it++)
    valid += square(prediction(*it) - (*it)->value);
  for (set<vote*>::iterator it = testVotes.begin(); it != testVotes.end(); it++)
  {
    double err = square(prediction(*it) - (*it)->value);
    test += err;
    testSte += err*err;
    if (nTrainingPerUser.find((*it)->user) != nTrainingPerUser.end())
    {
      int nu = nTrainingPerUser[(*it)->user];
      if (errorVsTrainingUser.find(nu) == errorVsTrainingUser.end())
        errorVsTrainingUser[nu] = vector<double> ();
      errorVsTrainingUser[nu].push_back(err);
    }
    if (nTrainingPerBeer.find((*it)->item) != nTrainingPerBeer.end())
    {
      int nb = nTrainingPerBeer[(*it)->item];
      if (errorVsTrainingBeer.find(nb) == errorVsTrainingBeer.end())
        errorVsTrainingBeer[nb] = vector<double> ();
      errorVsTrainingBeer[nb].push_back(err);
    }
  }

  // Standard error
  for (map<int, vector<double> >::iterator it = errorVsTrainingBeer.begin(); it != errorVsTrainingBeer.end(); it++)
  {
    if (it->first > 100)
      continue;
    double av, var;
    averageVar(it->second, av, var);
  }
  train /= trainVotes.size();
  valid /= validVotes.size();
  test /= testVotes.size();
  testSte /= testVotes.size();
  testSte = sqrt((testSte - test*test) / testVotes.size());
}

/// Print out the top words for each topic
void topicCorpus::topWords()
{
  printf("Top words for each topic:\n");
  for (int k = 0; k < K; k++)
  {
    vector < pair<double, int> > bestWords;
    for (int w = 0; w < nWords; w++)
      bestWords.push_back(pair<double, int> (-topicWords[w][k], w));
    sort(bestWords.begin(), bestWords.end());
    for (int w = 0; w < 10; w++)
    {
      printf("%s (%f) ", corp->idWord[bestWords[w].second].c_str(), -bestWords[w].first);
    }
    printf("\n");
  }
}

/// Subtract averages from word weights so that each word has average weight zero across all topics (the remaining weight is stored in "backgroundWords")
void topicCorpus::normalizeWordWeights(void)
{
  for (int w = 0; w < nWords; w++)
  {
    double av = 0;
    for (int k = 0; k < K; k++)
      av += topicWords[w][k];
    av /= K;
    for (int k = 0; k < K; k++)
      topicWords[w][k] -= av;
    backgroundWords[w] += av;
  }
}

/// Save a model and predictions to two files
void topicCorpus::save(char* modelPath, char* predictionPath)
{
  if (modelPath)
  {
    FILE* f = fopen_(modelPath, "w");
    if (lambda > 0)
      for (int k = 0; k < K; k++)
      {
        vector < pair<double, int> > bestWords;
        for (int w = 0; w < nWords; w++)
          bestWords.push_back(pair<double, int> (-topicWords[w][k], w));
        sort(bestWords.begin(), bestWords.end());
        for (int w = 0; w < nWords; w++)
          fprintf(f, "%s %f\n", corp->idWord[bestWords[w].second].c_str(), -bestWords[w].first);
        if (k < K - 1)
          fprintf(f, "\n");
      }
    fclose(f);
  }

  if (predictionPath)
  {
    FILE* f = fopen_(predictionPath, "w");
    for (set<vote*>::iterator it = testVotes.begin(); it != testVotes.end(); it++)
      fprintf(f, "%s %s %f %f\n", corp->rUserIds[(*it)->user].c_str(), corp->rBeerIds[(*it)->item].c_str(),
              (*it)->value, bestValidPredictions[*it]);
    fclose(f);
  }
}

/// Train a model for "emIterations" with "gradIterations" of gradient descent at each step
void topicCorpus::train(int emIterations, int gradIterations)
{
  double bestValid = numeric_limits<double>::max();
  double bestPrecision = -1;
  for (int emi = 0; emi < emIterations; emi++)
  {
    //printf("%f", gamma_beer[0][0]);
    lbfgsfloatval_t fx = 0;
    lbfgsfloatval_t* x = lbfgs_malloc(NW);
    for (int i = 0; i < NW; i++)
      x[i] = W[i];

    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.max_iterations = gradIterations;
    param.epsilon = 1e-2;
    param.delta = 1e-2;
    lbfgs(NW, x, &fx, evaluate, progress, (void*) this, &param);
    printf("\nenergy after gradient step = %f\n", fx);
    lbfgs_free(x);

    if (lambda > 0)
    {
      updateTopics(true);
      normalizeWordWeights();
      topWords();
    }
    
    double train_rmse, valid_rmse, test_rmse, testSte_rmse;
    validTestError(train_rmse, valid_rmse, test_rmse, testSte_rmse);
    printf("Error (train/valid/test) = %f/%f/%f (%f)\n", train_rmse, valid_rmse, test_rmse, testSte_rmse);    

    double valid_p, std_p;
    precision(&valid_p, &std_p, 0);
    fprintf(stderr, "Test P = %f(%f), Test Std = %f\n", valid_p, bestPrecision, std_p);
    
    //double bestPrecision = -1;
    if (valid_p > bestPrecision)
    {
      bestPrecision = valid_p;
      precision(&valid_p, &std_p, 1);
    }
    else
    {
      printf("not saved!");
    }   
  }
}

int main(int argc, char** argv)
{
  srand(0);

  if (argc < 2)
  {
    printf("Input files ave required\n");
    exit(0);
  }

  double latentReg = 0;
  double lambda = 0.1;
  int K = 5;
  char* modelPath = "model.out";
  char* predictionPath = "predictions.out";

  if (argc == 10)
  {
    latentReg = atof(argv[5]);
    lambda = atof(argv[6]);
    K = atoi(argv[7]);
    modelPath = argv[8];
    predictionPath = argv[9];
  }

  printf("corpus = %s\n", argv[1]);
  printf("latentReg = %f\n", latentReg);
  printf("lambda = %f\n", lambda);
  printf("K = %d\n", K);

  corpus corp(argv[1], argv[2], argv[3], argv[4], 0);

  topicCorpus ec(&corp, K, // K
                 latentReg, // latent topic regularizer
                 lambda); // lambda
  ec.train(50, 50);
  ec.save(modelPath, predictionPath);

  return 0;
}
