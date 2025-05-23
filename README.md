These two MATLAB programs implement the two TBQ algorithms presented in the following paper.

Interested readers may run these programs to replicate the calculations for the second numerical example (two-dimensional series system) in the paper. The output results of the algorithm are as follows (generated with TBQ_VI):

Mean estimate of pf（using bridging）： 0.00000734                                   
Mean estimate of pf（without bridging）： 0.00000727                                                                                           
Reference value of pf： 0.00000709                                           
CoV of pf estimate： 0.0494                                             
Number of model calls： 83                                                                           
gamma values:0.0000  1.3663  2.3340  3.9271  Inf                                                          
Accumulated numbers of model calls for each tempering stage:10  51  62  70  83      
Mean estimates of probability ratios:1.0000  0.0054  0.0605  0.1055  0.2122           

 The program will also output a result graph with three columns. Each row corresponds to the results of one tempering stage: the first column shows the mean estimate of the non-normalized AID and the adaptively generated training samples; the second column presents the reference AID; and the third column includes MCMC samples generated to follow this AID. For example, the figure produced with TBQ_VI is given below as an example.
![upload](https://github.com/user-attachments/assets/4b33ac12-c077-47ad-af9e-eefc066df43f)
