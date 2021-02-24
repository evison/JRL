Overview

This is an implementation of the Joint Representation Learning Model (JRLM) for product
recommendation based on heterogeneous information sources [2].  Please cite the following
paper if you plan to use it for your project：
    
    	Yongfeng Zhang, Qingyao Ai, Xu Chen, W. Bruce Croft.  2017.
	"Joint Representation Learning for Top-N Recommendation with Heterogeneous
	Information Sources".  In Proceedings of CIKM ’17.
    	
The JRL is a deep neural network model that jointly learn latent representations for
products and users based on reviews, images and product ratings.  The model can jointly or
independently latent representations for products and users based on different information.

The probability (which is also the rank score) of an product being purchased by a user can
be computed with their concatenated latent representations from different information
sources.  Please refer to the paper for more details.


Requirements

  o To run the JRL model in ./JRL/ and the python scripts in ./scripts/, python 2.7+ and
    Tensorflow v1.0+ are needed.
    
  o To run the jar package in ./jar/, JDK 1.7 is needed.
  
  o To compile the java code in ./java/, Galago from the Lemur Project is needed.
    (https://sourceforge.net/p/lemur/wiki/Galago%20Installation/)


Data Preparation

  o Note: the already splitted dataset used in this paper can be downloaded from the following link:

	https://www.dropbox.com/s/th672ttebwxhsfx/CIKM2017.zip?dl=0. 
    
    If the above link doesn't work, please click the following one:
    
    	https://drive.google.com/drive/folders/1RG_izwfdXrjTUkIlnRnp4dp1f2Mk33L6?usp=sharing
    
    If you want to process new datasets, please follow the instructions below.

  o Download Amazon review datasets from http://jmcauley.ucsd.edu/data/amazon/.
    In our paper, we used 5-core data.
  
  o Stem and remove stop words from the Amazon review datasets if needed.  In our paper, we
    stem the field of “reviewText” and “summary” without stop word removal.

       java -Xmx4g -jar ./jar/AmazonReviewData_preprocess.jar <jsonConfigFile> <review_file> <output_review_file>

       where

       <jsonConfigFile>       A JSON file that specify the file path of stop words list.
                              An example can be found in the root directory.  Enter “false” if
                              you don’t want to remove stop words. 

       <review_file>          The path for the original Amazon review data.

       <output_review_file>   The output path for processed Amazon review data.
     
    o Index datasets

        python ./scripts/index_and_filter_review_file.py <review_file> <indexed_data_dir> <min_count>

        where
      
        <review_file>       The file path for the Amazon review data.
	
        <indexed_data_dir>  The output directory for indexed data.
	
        <min_count>         The minimum count for terms.  If a term appears less then <min_count>
	                    times in the data, it will be ignored.
      
    o Split train/test
        -- Download the meta data from http://jmcauley.ucsd.edu/data/amazon/ 

        -- Split datasets for training and test
	
             python ./scripts/split_train_test.py <indexed_data_dir> <review_sample_rate>

             where
	     
             <indexed_data_dir>    The directory for indexed data.
             <review_sample_rate>  The proportion of reviews used in test for each user.  In our
	                           paper, we used 0.3.

        --  Match image features
            + Download the image features from http://jmcauley.ucsd.edu/data/amazon/ .
	    
            + Match image features with product ids.

                python ./scripts/match_with_image_features.py <indexed_data_dir> <image_feature_file>

                where
		
		<indexed_data_dir>     The directory for indexed data.
		<image_feature_file>   The file for image features data.

        -- Match rating features
           + Construct latent representations based on rating information with any method you like
	     (e.g. BPR).
	   
           + Format the latent factors of items and users in "item_factors.csv" and "user_factors.csv"
	     such that each row represents one latent vector for the corresponding item/user in the
	     <indexed_data_dir>/product.txt.gz and user.txt.gz.  See example csv files.
	   
           + Put the item_factors.csv and user_factors.csv into <indexed_data_dir>.
		

Model Training/Testing

  python ./JRL/main.py --<parameter_name> <parameter_value> --<parameter_name> <parameter_value> …

  where parameter names and values include:

  learning_rate               The learning rate in training.  Default 0.05.

  learning_rate_decay_factor  Learning rate decays by this much whenever the loss is higher than
                              three previous losses.  Default 0.90.

  max_gradient_norm           Clip gradients to this norm.  Default 5.0.

  subsampling_rate            The rate to subsampling.  Default 1e-4. 

  L2_lambda                   The lambda for L2 regularization.  Default 0.0.

  image_weight                The weight for image feature based training loss.  See the paper for
                              more details.

  batch_size                  Batch size used in training.  Default 64.

  data_dir                    Data directory, which should be the <indexed_data_dir>.

  input_train_dir             The directory of training and testing data, which usually is
                              <data_dir>/query_split/

  train_dir                   Model directory and output directory

  similarity_func             The function to compute the ranking score for an item with the joint model
                              of query and user embeddings.  Default “product”.  

                              Available functions include:
                                “product”       The dot product of two vectors.
                                “cosine”        The cosine similarity of two vectors.
                                “bias_product”  The dot product plus a item-specific bias.

  net_struct                  Network structure parameters.  Different parameters are separated by “_”.
                              Default “simplified_fs”.  Network structure parameters include:

                                “bpr”         Train models in a bpr framework [1].

                                “simplified”  Simplified embedding-based language models without modeling
				              for each review [2].
					      
                                “hdc”         Use regularized embedding-based language models with word
				              context [4].  Otherwise, use the default model, which is the
					      embedding-based language models based on paragraph vector model. [3]

                                “extend”      Use the extendable model structure.  See more details in the paper.

                                “text”        Use review data. 

                                “image”       Use image data.

                                "rate"        Use rating-based latent representations.

                                 Note: If none of "text", "image" and "rate" is specified, the model will use
				       all of them.
		
  embed_size                  Size of each embedding.  Default 100.

  window_size                 Size of context window for hdc model.  Default 5.

  max_train_epoch             Limit on the epochs of training (0 means no limit).  Default 5.

  steps_per_checkpoint        How many training steps to do per checkpoint.  Default 200.

  seconds_per_checkpoint      How many seconds to wait before storing embeddings.  Default 3600.

  negative_sample             How many samples to generate for negative sampling.  Default 5.

  decode                      Set to “False" for training and “True" for testing.  Default “False".

  test_mode                   Test modes.  Default “product_scores".  Test modes include the following:
  
                                “product_scores”    Output ranking results and ranking scores.

                                “output_embedding"  Output embedding representations for users, items and words.
			      
  rank_cutoff                 Rank cutoff for output rank lists.  Default 100.


Evaluation
  o After training with "--decode False”, generate test rank lists with "--decode True”.
  
  o TREC format rank lists for test data will be stored in <train_dir> with name “test.<similarity_func>.ranklist”
  
  o Evaluate test rank lists with ground truth <input_train_dir>/test.qrels.
  
      python recommendation_metric.py <rank_list_file> <test_qrel_file> <rank_cutoff_list>

      where
      
      <rank_list_file>     The result list, e.g. <train_dir>/test.<similarity_func>.ranklist
      
      <test_qrel_file>     The ground truth, e.g. <input_train_dir>/test.qrels
      
      <rank_curoff_list>   The number of top documents to used in evaluation, e.g. NDCG@10 -> rank+cutoff_list=10.
      

References

  [1] Ste en Rendle, C. Freudenthaler, Zeno Gantner and Lars Schmidtieme.  2009.
      "BPR: Bayesian personalized ranking from implicit feedback". In UAI.

  [2] Yongfeng Zhang, Qingyao Ai, Xu Chen, W. Bruce Croft.  2017.
      "Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources".
      In Proceedings of CIKM ’17.

  [3] Quoc V Le and Tomas Mikolov.  2014.
      "Distributed Representations of Sentences and Documents". In ICML.

  [4] Sun, Fei, Jiafeng Guo, Yanyan Lan, Jun Xu, and Xueqi Cheng.  2015.
      "Learning Word Representations by Jointly Modeling Syntagmatic and Paradigmatic Relations".  In ACL.

  [5] Ivan Vulić and Marie-Francine Moens.  2015.
      "Monolingual and cross-lingual information retrieval models based on (bilingual) word embeddings".
      In Proceedings of the 38th ACM SIGIR.
