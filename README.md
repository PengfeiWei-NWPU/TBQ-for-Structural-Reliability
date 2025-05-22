These two MATLAB programs implement the two TBQ algorithms presented in the following paper.

Interested readers may run these programs to replicate the calculations for the second numerical example (two-dimensional series system) in the paper. The output results of the algorithm are as follows:

Mean estimate of pf（using bridging）： 0.00000734                                   
Mean estimate of pf（without bridging）： 0.00000727                                                                                           
Reference value of pf： 0.00000668                                           
CoV of pf estimate： 0.0494                                             
Number of model calls： 83.0000                                                                       
gamma values:0.0000  1.3663  2.3340  3.9271  Inf                                                          
Accumulated numbers of model calls for each tempering stage:10.0000  51.0000  62.0000  70.0000  83.0000       
Mean estimates of probability ratios:1.0000  0.0054  0.0605  0.1055  0.2122           

 The program will also output a result graph with three columns. Each row corresponds to the results of one tempering stage: the first column shows the mean estimate of the non-normalized AID and the adaptively generated training samples; the second column presents the reference AID; and the third column includes MCMC samples generated to follow this AID.
