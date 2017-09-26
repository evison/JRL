Compile using "make".

You will need to export liblbfgs to your LD_LIBRARY_PATH to run the code (export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/liblbfgs-1.10/lib/.libs/).

Run:

./train whole_file tain_file valid_file test_file latentReg lambda latentFactor model_out prediction_out

e.g.

./train /root/share/cx/amazon_baseline_data/Beauty/bpr_whole.txt /root/share/cx/amazon_baseline_data/Beauty/bpr_train.txt /root/share/cx/amazon_baseline_data/Beauty/bpr_test.txt /root/share/cx/amazon_baseline_data/Beauty/bpr_test.txt 0.01 0.1 10 "model.out" "pre.out"


Data format:

user item rating time word_number words

e.g.

A1Z59RFKN0M5QL 7806397051 1.0 1376611200 23 please dont rachett palette size picture colors sheer slides face wax dont expect  makeup stay put spend money good stuff 2 thumbs


