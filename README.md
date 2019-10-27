# Code for NeurIPS 2019 paper *Scalable inference of topic evolution via models for latent geometric structures*

*The code has been tested on Ubuntu 14.04, 16.04, 18.04 LTS*



----
## Three Settings
The code repo contains *three different algorithms* for 
*scalable inference of topic evolution* considering 
*three different settings*: 

1. Distributed:
texts *distributed/grouped* in some manner
(e.g., location, categories)
2. Online (Streaming):
texts coming in *online/streaming* fashion

3. Distributed & Online (Streaming):
texts coming in *both distributed and online* fashion

####Algorithms Matching Three Settings (Setting:Algorithm)
1. Distributed: DM
2. Online: SDM
3. Distributed & Online: SDDM


### usage:
Install the required dependencies:

>
    ./conf

To run three different algorithms, go to the corresponding folder
(*distributed* for *DM*, *online* for *SDM*, *distributed_online* for *SDDM*)
and refer the README.md file under each for algorithm & setting specific details.

###Perplexity:
To compute the perplexity on the new dataset with three algorithms mentioned above,
go to the folder *perplexity* and refer the README.md file under the folder for details.

###Data:
Two datasets are used - *EJC* (ejc) and *Wiki* (wiki) (more details regarding the two datasets are given in the paper)

Each dataset folder has *5 subfolders*:

* <dataset name>\_group (dataset partitioned by different groups)

* <dataset name>\_time (dataset partitioned by different timestamp)

* <dataset name>\_time\_group (dataset first partitioned by time and then partitioned by groups)

* <dataset name>\_test (test dataset)

* <dataset name>\_time\_group\_meta_data (contains *vocabulary* and *group name & id mapping*)

EJC_Data ([links] (https://drive.google.com/file/d/1fVQIwmZAdb76pftMKJdT9MsSq9Tlfsk0/view?usp=sharing)):

Wiki_Data ([links]
(https://drive.google.com/file/d/12thNKl_BpF0ozmg_wyLAGHVQKz0lBu7m/view?usp=sharing)):

## Parameter Interpretation

* tau0

Controls how topic changes from time to time:
small value => sharp change of topics between different time points

* tau1

Affects the number of topics:
small tau1 => larger variance between topics => less number of topics:
vectors with small difference are regarded as different topics

* gamma0

Parameter of Beta process
