###Usage:
First change the *paths* variables in *calculate_perplexity.sh* to be the paths corresponding to  *test data folder* (*TEST_PATH*) and *topics path* containing the topics pre-compuated (*TOPICS_PATH*)

Change the *ALGORITHM* variable in *calculate_perplexity.sh*
to be one of *dm*, *sdm*, *sddm*.

Compute the perplexity of the specified test dataset using the pre-computed topics:

>
    ./calculate_perplexity.sh

* *top\_words* folder under the *global\_topics* contains 20 words with *the top 20 probability* of
the corresponding topic for each topic computed