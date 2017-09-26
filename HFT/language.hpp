#include "common.hpp"

class topicCorpus
{
public:
  topicCorpus(corpus* corp, // The corpus
              int K, // The number of latent factors
              double latentReg, // Parameter regularizer used by the "standard" recommender system
              double lambda) : // Word regularizer used by HFT
    corp(corp), K(K), latentReg(latentReg), lambda(lambda)
  {
    srand(0);

    nUsers = corp->nUsers;
    nBeers = corp->nBeers;
    nWords = corp->nWords;

    votesPerUser = new std::vector<vote*>[nUsers];
    votesPerBeer = new std::vector<vote*>[nBeers];
    trainVotesPerUser = new std::vector<vote*>[nUsers];
    trainVotesPerBeer = new std::vector<vote*>[nBeers];

    for (std::vector<vote*>::iterator it = corp->wholeV->begin(); it != corp->wholeV->end(); it++)
    {
      vote* vi = *it;
      votesPerUser[vi->user].push_back(vi);
    }

    for (int user = 0; user < nUsers; user++)
      for (std::vector<vote*>::iterator it = votesPerUser[user].begin(); it != votesPerUser[user].end(); it++)
      {
        vote* vi = *it;
        votesPerBeer[vi->item].push_back(vi);
      }

    //double testFraction = 0.1;
    //if (corp->V->size() > 2400000)
    //{
    //  double trainFraction = 2000000.0 / corp->V->size();
    // testFraction = (1.0 - trainFraction)/2;
    //}
    
    for (std::vector<vote*>::iterator it = corp->trainV->begin(); it != corp->trainV->end(); it ++)
    {
      trainVotes.push_back(*it);
      trainVotesPerUser[(*it)->user].push_back(*it);
      trainVotesPerBeer[(*it)->item].push_back(*it);
    }
    for (std::vector<vote*>::iterator it = corp->validV->begin(); it != corp->validV->end(); it ++)
    {
      validVotes.push_back(*it);
      //validVotes.insert(*it);
    }
    for (std::vector<vote*>::iterator it = corp->testV->begin(); it != corp->testV->end(); it ++)
    {
      testVotes.insert(*it);
    }


    std::vector<vote*> remove;
    for (std::set<vote*>::iterator it = testVotes.begin(); it != testVotes.end(); it ++)
    {
      if (nTrainingPerUser.find((*it)->user) == nTrainingPerUser.end()) remove.push_back(*it);
      else if (nTrainingPerBeer.find((*it)->item) == nTrainingPerBeer.end()) remove.push_back(*it);
    }
    for (std::vector<vote*>::iterator it = remove.begin(); it != remove.end(); it ++)
    {
      // Uncomment the line below to ignore (at testing time) users/items that don't appear in the training set
      // testVotes.erase(*it);
    }
    


    
    // total number of parameters
    NW = 1 + 1 + (K + 1) * (nUsers + nBeers) + K * nWords;

    // Initialize parameters and latent variables
    // Zero all weights
    W = new double [NW];
    for (int i = 0; i < NW; i++)
      W[i] = 0;
    getG(W, &alpha, &kappa, &beta_user, &beta_beer, &gamma_user, &gamma_beer, &topicWords, true);

    // Set alpha to the average
    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      *alpha += (*vi)->value;
    }
    *alpha /= trainVotes.size();

    double train, valid, test, testSte;
    validTestError(train, valid, test, testSte);
    printf("Error w/ offset term only (train/valid/test) = %f/%f/%f (%f)\n", train, valid, test, testSte);

    // Set beta to user and product offsets
    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      vote* v = *vi;
      beta_user[v->user] += v->value - *alpha;
      beta_beer[v->item] += v->value - *alpha;
    }
    for (int u = 0; u < nUsers; u++)
      beta_user[u] /= votesPerUser[u].size();
    for (int b = 0; b < nBeers; b++)
      beta_beer[b] /= votesPerBeer[b].size();
    validTestError(train, valid, test, testSte);
    printf("Error w/ offset and bias (train/valid/test) = %f/%f/%f (%f)\n", train, valid, test, testSte);

    // Actually the model works better if we initialize none of these terms
    if (lambda > 0)
    {
      *alpha = 0;
      for (int u = 0; u < nUsers; u++)
        beta_user[u] = 0;
      for (int b = 0; b < nBeers; b++)
        beta_beer[b] = 0;
    }

    wordTopicCounts = new int*[nWords];
    for (int w = 0; w < nWords; w++)
    {
      wordTopicCounts[w] = new int[K];
      for (int k = 0; k < K; k++)
        wordTopicCounts[w][k] = 0;
    }

    // Generate random topic assignments
    topicCounts = new long long[K];
    for (int k = 0; k < K; k++)
      topicCounts[k] = 0;
    beerTopicCounts = new int*[nBeers];
    beerWords = new int[nBeers];
    for (int b = 0; b < nBeers; b ++)
    {
      beerTopicCounts[b] = new int[K];
      for (int k = 0; k < K; k ++)
        beerTopicCounts[b][k] = 0;
      beerWords[b] = 0;
    }

    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      vote* v = *vi;
      wordTopics[v] = new int[v->words.size()];
      beerWords[(*vi)->item] += v->words.size();

      for (int wp = 0; wp < (int) v->words.size(); wp++)
      {
        int wi = v->words[wp];
        int t = rand() % K;

        wordTopics[v][wp] = t;
        beerTopicCounts[(*vi)->item][t]++;
        wordTopicCounts[wi][t]++;
        topicCounts[t]++;
      }
    }

    // Initialize the background word frequency
    totalWords = 0;
    backgroundWords = new double[nWords];
    for (int w = 0; w < nWords; w ++)
      backgroundWords[w] = 0;
    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      for (std::vector<int>::iterator it = (*vi)->words.begin(); it != (*vi)->words.end(); it++)
      {
        totalWords++;
        backgroundWords[*it]++;
      }
    }
    for (int w = 0; w < nWords; w++)
      backgroundWords[w] /= totalWords;

    if (lambda == 0)
    {
      for (int u = 0; u < nUsers; u++)
      {
        if (nTrainingPerUser.find(u) == nTrainingPerUser.end()) continue;
        for (int k = 0; k < K; k++)
          gamma_user[u][k] = rand() * 1.0 / RAND_MAX;
      }
      for (int b = 0; b < nBeers; b++)
      {
        if (nTrainingPerBeer.find(b) == nTrainingPerBeer.end()) continue;
        for (int k = 0; k < K; k++)
          gamma_beer[b][k] = rand() * 1.0 / RAND_MAX;
      }
    }
    else
    {
      for (int w = 0; w < nWords; w++)
        for (int k = 0; k < K; k++)
          topicWords[w][k] = 0;
    }

    normalizeWordWeights();
    if (lambda > 0)
      updateTopics(true);

    *kappa = 1.0;
  }

  ~topicCorpus()
  {
    delete[] votesPerBeer;
    delete[] votesPerUser;
    delete[] trainVotesPerBeer;
    delete[] trainVotesPerUser;

    for (int w = 0; w < nWords; w ++)
      delete[] wordTopicCounts[w];
    delete[] wordTopicCounts;

    for (int b = 0; b < nBeers; b ++)
      delete[] beerTopicCounts[b];
    delete[] beerTopicCounts;
    delete[] beerWords;
    delete[] topicCounts;

    delete[] backgroundWords;

    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      delete[] wordTopics[*vi];
    }

    clearG(&alpha, &kappa, &beta_user, &beta_beer, &gamma_user, &gamma_beer, &topicWords);
    delete[] W;
  }

  double prediction(vote* vi);

  void dl(double* grad);
  void train(int emIterations, int gradIterations);
  double lsq(void);
  void validTestError(double& train, double& valid, double& test, double& testSte);
  void normalizeWordWeights(void);
  void save(char* modelPath, char* predictionPath);
  //trainCorp(trainCorp), validCorp(validCorp), testCorp(testCorp), 
  corpus* corp;
  corpus* trainCorp;
  corpus* validCorp;
  corpus* testCorp;
  
  // Votes from the training, validation, and test sets
  std::vector<vote*> trainVotes;
  std::vector<vote*> validVotes;
  std::set<vote*> testVotes;

  std::map<vote*, double> bestValidPredictions;

  std::vector<vote*>* votesPerBeer; // Vector of votes for each item
  std::vector<vote*>* votesPerUser; // Vector of votes for each user
  std::vector<vote*>* trainVotesPerBeer; // Same as above, but only votes from the training set
  std::vector<vote*>* trainVotesPerUser;

  int getG(double* g,
           double** alpha,
           double** kappa,
           double** beta_user,
           double** beta_beer,
           double*** gamma_user,
           double*** gamma_beer,
           double*** topicWords,
           bool init);
  void clearG(double** alpha,
              double** kappa,
              double** beta_user,
              double** beta_beer,
              double*** gamma_user,
              double*** gamma_beer,
              double*** topicWords);

  void wordZ(double* res);
  void topicZ(int beer, double& res);
  void updateTopics(bool sample);
  void topWords();

  // Model parameters
  double* alpha; // Offset parameter
  double* kappa; // "peakiness" parameter
  double* beta_user; // User offset parameters
  double* beta_beer; // Item offset parameters
  double** gamma_user; // User latent factors
  double** gamma_beer; // Item latent factors

  double* W; // Contiguous version of all parameters, i.e., a flat vector containing all parameters in order (useful for lbfgs)

  double** topicWords; // Weights each word in each topic
  double* backgroundWords; // "background" weight, so that each word has average weight zero across all topics
  // Latent variables
  std::map<vote*, int*> wordTopics;

  // Counters
  int** beerTopicCounts; // How many times does each topic occur for each product?
  int* beerWords; // Number of words in each "document"
  long long* topicCounts; // How many times does each topic occur?
  int** wordTopicCounts; // How many times does this topic occur for this word?
  long long totalWords; // How many words are there?

  int NW;
  int K;

  double latentReg;
  double lambda;

  std::map<int,int> nTrainingPerUser; // Number of training items for each user
  std::map<int,int> nTrainingPerBeer; // and item

  int nUsers; // Number of users
  int nBeers; // Number of items
  int nWords; // Number of words
};
