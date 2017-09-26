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
  fflush( stdout);
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
    //for (vector<vote*>::iterator it = trainVotes.begin(); it != trainVotes.end(); it++)
      //fprintf(f, "%s %s %f %f\n", corp->rUserIds[(*it)->user].c_str(), corp->rBeerIds[(*it)->item].c_str(),
        //      (*it)->value, bestValidPredictions[*it]);
    //fprintf(f, "\n");
    //for (vector<vote*>::iterator it = validVotes.begin(); it != validVotes.end(); it++)
      //fprintf(f, "%s %s %f %f\n", corp->rUserIds[(*it)->user].c_str(), corp->rBeerIds[(*it)->item].c_str(),
        //      (*it)->value, bestValidPredictions[*it]);
    //fprintf(f, "\n");
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
  for (int emi = 0; emi < emIterations; emi++)
  {
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

    double train, valid, test, testSte;
    validTestError(train, valid, test, testSte);
    printf("Error (train/valid/test) = %f/%f/%f (%f)\n", train, valid, test, testSte);
 
      
    if (valid < bestValid)
    {
      bestValid = valid;
      for (set<vote*>::iterator it = testVotes.begin(); it != testVotes.end(); it++)
         bestValidPredictions[*it] = prediction(*it);
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
  //corpus trainCorp(argv[2], 0);
  //corpus validCorp(argv[3], 0);
  //corpus testCorp(argv[4], 0);

  topicCorpus ec(&corp, K, // K
                 latentReg, // latent topic regularizer
                 lambda); // lambda
  ec.train(50, 50);
  ec.save(modelPath, predictionPath);

  return 0;
}
