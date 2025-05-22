%%this code implement the TBQ-V2 algorithm for the 2d series system
clear
%%%%%%%%%define the LSF of the series system
a=5;b=9;
g1 = @(x)a+(x(:,1)-x(:,2)).^2/10 - (x(:,1)+x(:,2))/sqrt(2);
g2 = @(x)a+(x(:,1)-x(:,2)).^2/10 + (x(:,1)+x(:,2))/sqrt(2);
g3 = @(x)(x(:,1)-x(:,2)) + b/sqrt(2);
g4 = @(x)(x(:,2)-x(:,1)) + b/sqrt(2);
g12 = @(x)min(g1(x),g2(x));
g34 = @(x)min(g3(x),g4(x));
g = @(x)min(g12(x),g34(x));

nx = 2;


%%setting the algorithm parameters
N0 = 10; %%Number of initial sample size
StopThreshold_lim = 0.08;% Stopping threshold for each layer
StopThreshold_Inf = 0.05;% Stopping threshold for last stage
NMC = 1e4; %size of MC/MCMC samples for each layer    
StopLimit = 1;%consecutive times of stopping conditions being satsfied 
Lchain = 50;%length of each chain for avoiding sample replication
Type = 2; %%1 or 2, indicating the type of smooth indicator function
CV = 2;%parameters for controlling the variation of estimator for each layer, suggested to be 1~2
Nrow = 5;%number of rows for plot

%%%%%%%compute reference results with MCS
Ncand = 5e7;
XCandSamp = normrnd(0,1,Ncand,nx);
YcandSamp = g(XCandSamp);
pfRef = mean(YcandSamp<0);


%%%define colors for plotting
C14 = [026 049 139;073 108 206;130 170 231;185 210 243; 230 240 254; 249 219 229;247 166 191;228 107 144;192 063 103;154 019 061]/255;

%%%initialization of the algorithm 
Zratio = [1];
Zratio_bridg = [1];%initialize the 
ZVar = [0];
gamma_fixed = [0]; %initialization of the tempering parameter
j=1;%index of tempering stage
XMC(:,:,1) = mvnrnd(zeros(1,nx),diag(ones(1,nx)),NMC);% samples of first stage, for learning second stage
RandInd = randperm(NMC,NMC);%index of random permutation
IndTrain = randperm(NMC,N0);
X0 = XMC(IndTrain,:,1);
Y0 = g(X0);
Xtrain = X0;
Ytrain = Y0;
Ncall_j = [N0];%record the number of training points for each stage
Ncall = N0;% record the number of training points
PiMean{1} = @(x) ones(size(x,1),1);%%initialize the smooth indictaor function for j=1
PiMean_MC(:,1) = ones(NMC,1);%%initialize the smooth indictaor function value for XMC^(j)

%%used for plot the intermediate smooth indicator function 
Nplot = 100;
Zplot = linspace(-6,6,Nplot);
[Xgrid, Ygrid] = meshgrid(Zplot,Zplot);
figure

NcallAcc = [N0];%accumulate number of function calls
while 1==1
%%%the outer loop 
   j=j+1;
   StopFlag = 0;
   fixGamma(j) = 0;
   Niter(j) = 0;%record the iteration steps of the j-th 
   while 1==1
       %%inner loop for learning Z_j and the corresponding AID
       Niter(j) = Niter(j)+1;
       GPRmodel = fitrgp(Xtrain,Ytrain,'KernelFunction','ardsquaredexponential'...
                   ,'BasisFunction','constant','Sigma', 1e-3, 'ConstantSigma', true,'SigmaLowerBound', eps,'Standardize',false);  
       [MeanPred,SDPRed] = predict(GPRmodel,XMC(:,:,j-1));
        weightLim = normcdf(-MeanPred./SDPRed)./PiMean_MC(:,j-1);
        if std(weightLim)/mean(weightLim) > CV 
            weight= @(gamma)PiMeanFun(GPRmodel,XMC(:,:,j-1), gamma)./PiMean_MC(:,j-1);
            COV_weight = @(gamma) std(weight(gamma))/mean(weight(gamma));
            gamma_Active = fminbnd(@(gamma)abs(COV_weight(gamma)-CV),gamma_fixed(j-1),gamma_fixed(j-1)+5);
            StopThreshold = StopThreshold_lim;
        else
            gamma_Active = Inf;%%if the maximum COV value is lower than CV, set gamma_Active as Infinity
            StopThreshold = StopThreshold_Inf;
        end
       fprintf('Active gamma value： %.4f\n', gamma_Active);
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       Sigma0 = GPRmodel.KernelInformation.KernelParameters(end);
       Sigma1 = GPRmodel.KernelInformation.KernelParameters(1:end-1);
       kfcn = @(ZN,ZM) Sigma0^2*exp(-(pdist2(ZN,ZM,'seuclidean',Sigma1).^2)/2);%%kernel function
       K = kfcn(Xtrain,Xtrain)+GPRmodel.Sigma.^2*eye(Ncall);
       invK = pinv(K);
       CovPredGP= @(z1,z2)kfcn(z1,z2)-kfcn(z1,Xtrain)*invK*kfcn(Xtrain,z2);%%posterior covariance
       CovValue = diag(CovPredGP(XMC(:,:,j-1),XMC(RandInd,:,j-1)));
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


      PiMeanPred = PiMeanFun(GPRmodel,XMC(:,:,j-1), gamma_Active);
      PiVarPred = PiVarFun(GPRmodel,XMC(:,:,j-1), gamma_Active);
      Mean_Zratio = mean(PiMeanPred./PiMean_MC(:,j-1));

       %%use the true COV
       PiCOVPred = PiCOVFun(MeanPred,SDPRed,CovValue,RandInd, gamma_Active);
       Var_Zjratio = mean(PiCOVPred./(PiMean_MC(:,j-1).*PiMean_MC(RandInd,j-1)));
       Cov_Zratio = sqrt(Var_Zjratio)/Mean_Zratio;

       fprintf('COV： %.4f\n', Cov_Zratio);
       fprintf('Ncall： %.1f\n', Ncall);

       if Cov_Zratio<StopThreshold
          StopFlag=StopFlag+1; 
       else
         StopFlag=0;
       end
       if StopFlag==StopLimit
         break;
       else
           AcqMC = sqrt(PiVarPred);
           [~,CandInd] = sort(AcqMC,'descend');
           XtrainCand1 = XMC(CandInd(1:1000),:,j-1);% pre-selection of candidate training point with PUQ
           rowsToRemove = ismember(XtrainCand1, Xtrain, 'rows'); 
           XtrainCand = XtrainCand1(~rowsToRemove,:);% remove the points which are already training points
           PVCAcqCand = PVCFun(GPRmodel,XtrainCand, XMC, CovPredGP,gamma_Active,j,PiMean,PiMean_MC);
           [~,Indmax] = max(PVCAcqCand);%Refine one point with PVC
           Xnew = XtrainCand(Indmax,:);
           Ncall = Ncall+1;
           Xtrain = [Xtrain;Xnew];
           Ytrain = [Ytrain;g(Xnew)];
           clear XtrainCand XtrainCand1 
       end
   end
   NcallAcc(j) = Ncall;
   gamma_fixed(j) = gamma_Active; %% fixed gamma value for the j-th stage
   GPRmodelFixed{j} = GPRmodel;%% fixed GP model for tyhe j-th stage
   PiMean{j} = @(x)PiMeanFun(GPRmodelFixed{j},x, gamma_fixed(j));%%the fixed mean prediction of Pi function for the j-th stage
   Zratio(j) = Mean_Zratio; %mean estimate of the ratio
   ZRef(j) = mean(normcdf(-gamma_fixed(j)*YcandSamp));
   ZAcc(j) = prod(Zratio(1:j));
   CovZ(j) = Cov_Zratio; 
   subplot(Nrow,3,(j-2)*3+1)
   for t = 1:Nplot
       DenGrid(:,t) = PiMean{j}([Xgrid(:,t),Ygrid(:,t)]).*mvnpdf([Xgrid(:,t),Ygrid(:,t)]);
       if Type==1
           RefDenGrid(:,t) = min(exp(-gamma_fixed(j)*g([Xgrid(:,t),Ygrid(:,t)])),1).*mvnpdf([Xgrid(:,t),Ygrid(:,t)]);
       else
           RefDenGrid(:,t) = normcdf(-gamma_fixed(j)*g([Xgrid(:,t),Ygrid(:,t)])).*mvnpdf([Xgrid(:,t),Ygrid(:,t)]);
       end
   end
  pcolor(Xgrid,Ygrid,DenGrid)
  colormap(C14)
  colorbar
  shading interp
  hold on
  if j==2
      plot(Xtrain(1:N0,1),Xtrain(1:N0,2),'Marker','o','MarkerEdgeColor',[231, 31, 24]/255,'LineStyle','none','MarkerFaceColor',[248 207 231]/255)
      hold on
  end
  plot(Xtrain(NcallAcc(j-1)+1:Ncall,1),Xtrain(NcallAcc(j-1)+1:Ncall,2),'Marker','hexagram','MarkerEdgeColor',[231, 31, 24]/255,'LineStyle','none','MarkerFaceColor',[248 207 231]/255)
  xlim([-5,5])
  ylim([-5,5])
  subplot(Nrow,3,(j-2)*3+2)
  pcolor(Xgrid,Ygrid,RefDenGrid)
  colormap(C14)
  colorbar
  shading interp
  xlim([-5,5])
  ylim([-5,5])
        %%Next produce the samples XMC(:,:,j) by pCN MCMC
      UnNormWeight = PiMean{j}(XMC(:,:,j-1))./PiMean_MC(:,j-1);
      SumWeight = sum(UnNormWeight);
      NormWeight = UnNormWeight/SumWeight;%normalize the weights
      SampInd = randsample(1:1:NMC,NMC,true,NormWeight);% resampling with replication
      InitialSamp = XMC(SampInd,:,j-1);%generate initial training samples following h_j
      WeightedMean = sum(repmat(NormWeight,1,nx).* XMC(:,:,j-1),1);%weighted mean
      Deviation = sqrt(repmat(NormWeight,1,nx)).*(XMC(:,:,j-1) - repmat(WeightedMean,NMC,1));%weighted deviation of samples
      WeightedCOV = Deviation'*Deviation;%compute the covariance of samples with weights
      PropSigma = 0.2^2*WeightedCOV;
      clear UnNormWeight SumWeight Weight SampInd_Record SampInd
      for k=1:NMC
          Chain{k}(1,:) = InitialSamp(k,:);%%seed of the k-th chain
      end
      parfor k=1:NMC
          for s = 2:Lchain% grow the s-th chain with seed Chain{k}(1,:), where Lchain can be set as 3~5, the higher the less replications of samples
              Xcand = mvnrnd(Chain{k}(s-1,:),PropSigma);%generate a candidate sample 
              alpha = PiMean{j}(Xcand).*mvnpdf(Xcand)./(PiMean{j}(Chain{k}(s-1,:)).*mvnpdf(Chain{k}(s-1,:))); % compute the acceptance probability of Xcand
              if min(alpha,1)>rand
                  Chain{k}(s,:) = Xcand;
              else
                  Chain{k}(s,:) = Chain{k}(s-1,:);
              end
          end
      end%end parfor for growing chains
   for k=1:NMC%last point of each chain as a sample
       XMC(k,:,j) = Chain{k}(Lchain,:);
   end
   clear Chain InitialSamp
   PiMean_MC(:,j) = PiMean{j}(XMC(:,:,j));
%%%%Next estimation with geometric bridging density%%%%%%%%%%%%%%%%%%%%%%%%%%%
pi_bridg_jminus1 = sqrt(PiMean{j}(XMC(:,:,j-1))./ PiMean_MC(:,j-1));
pi_bridg_j = sqrt(PiMean{j-1}(XMC(:,:,j))./PiMean_MC(:,j));
Zratio_bridg(j) = mean(pi_bridg_jminus1)/mean(pi_bridg_j);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(Nrow,3,(j-2)*3+3)
scatter(XMC(:,1,j),XMC(:,2,j),'Marker','.','MarkerEdgeColor',[0 49 255]/255,'MarkerFaceColor',[0 49 255]/255)
xlim([-5,5])
ylim([-5,5])
if gamma_Active==Inf%if gamma_Active==Inf, break the outer loop
   break;
end
end



fprintf('Mean estimate of pf（using bridging）： %.8f\n', prod(Zratio));
fprintf('Mean estimate of pf（without bridging）： %.8f\n', prod(Zratio_bridg));
fprintf('Reference value of pf： %.8f\n', pfRef);
fprintf('CoV of pf estimate： %.4f\n',  CovZ(end));
fprintf('Number of model calls： %d\n',  Ncall);
fprintf('gamma values:');
fprintf('%.4f  ',  gamma_fixed);
fprintf('\nAccumulated numbers of model calls for each tempering stage:');
fprintf('%d',  NcallAcc);
fprintf('\n Mean estimates of probability ratios:');
fprintf('%.4f  ',  Zratio_bridg);


function PiMeanPred = PiMeanFun(GPRmodel,x, gamma)
%%Define the smooth indicator function smoothed by gamma
if gamma>100
    gamma = Inf;
end
   [MeanPred,SDPRed,~] = predict(GPRmodel,x);
 if gamma == Inf
    PiMeanPred = normcdf(-MeanPred./SDPRed);
 else
    e = sqrt(1+gamma^2*SDPRed.^2);
    PiMeanPred = normcdf(-gamma*MeanPred./e);
 end
end


function PiVarPred = PiVarFun(GPRmodel,x, gamma)
%%Define the smooth indicator function smoothed by gamma
   Nx = size(x,1);
   [MeanPred,SDPRed,~] = predict(GPRmodel,x);
   SDPRed = max(SDPRed,1e-6);
   if gamma == Inf
      PiMeanPred = normcdf(-MeanPred./SDPRed);
      PiVarPred = PiMeanPred.*(1-PiMeanPred);
   else
      e = sqrt(1+gamma^2*SDPRed.^2);
      rho_e = gamma^2*SDPRed.^2./e.^2;
      Ind2 = find(abs(rho_e)>0.01);%index of samples need to be computed via mvncdf
      Ind1 = setdiff((1:Nx)',Ind2);%%can be computed by normcdf
      ZZ  = -gamma*MeanPred./e;
      PiMeanPred = normcdf(ZZ);
      PiVarPred(Ind1,:) = 1e-10;%if rho_e=0, then the posterior variance of pi also equals to zero, given finite value of gamma 
      for i=1:length(Ind2)
          PiVarPred(Ind2(i),:) = mvncdf([ZZ(Ind2(i),:),ZZ(Ind2(i),:)],[0,0],[1,rho_e(Ind2(i),:);rho_e(Ind2(i),:),1]) - PiMeanPred(Ind2(i),:).^2;
      end
   end
   PiVarPred = max(PiVarPred,0);
end

function PiCOVPred = PiCOVFun(MeanPred,SDPRed,CovValue,RandInd, gamma)
%%Covariance of the smooth indicator function smoothed by gamma
% Type = 1 or 2
   Nx = size(MeanPred,1);
   SDPRed2 = SDPRed(RandInd,:);
   rho = CovValue./(SDPRed.*SDPRed2);
   rho = min(rho,0.999);
   rho = max(rho,-0.999);
   if gamma == Inf
     d = -MeanPred./SDPRed;
     d2 = d(RandInd,:);
     PiMeanPred = normcdf(d);
     PiMeanPred2 = PiMeanPred(RandInd,:);
%      Ind2 = find(PiMeanPred.*PiMeanPred2>1e-3);
     Ind2 = find(abs(rho)>0.01);%index of samples need to be computed via mvncdf
     CDF2 = PiMeanPred.*PiMeanPred2;
     for i=1:length(Ind2)
         CDF2(Ind2(i),:) = mvncdf([d(Ind2(i),:),d2(Ind2(i),:)],[0,0],[1,rho(Ind2(i),:);rho(Ind2(i),:),1]);
     end
     PiCOVPred = CDF2 - PiMeanPred.*PiMeanPred2;
   else
      e = sqrt(1+gamma^2*SDPRed.^2);
      e2 = e(RandInd,:);
      rho_ee = -gamma^2*CovValue./(e.*e2);
      rho_ee = min(rho_ee,0.999);
      rho_ee = max(rho_ee,-0.999);
      Ind2 = find(abs(rho_ee)>0.02);%index of samples need to be computed via mvncdf
      Ind1 = setdiff((1:Nx)',Ind2);%%can be computed by normcdf
      PiMeanPred = normcdf(-gamma*MeanPred./e);
      PiMeanPred2 = PiMeanPred(RandInd,:);
      ZZ = -gamma*MeanPred./e;
      ZZ2 = ZZ(RandInd,:); 
      CDFZZ(Ind1,:) = PiMeanPred(Ind1,:).*PiMeanPred2(Ind1,:);
      for k=1:length(Ind2)
          CDFZZ(Ind2(k),:) = mvncdf([ZZ(Ind2(k),:),ZZ2(Ind2(k),:)],[0,0],[1,-rho_ee(Ind2(k));-rho_ee(Ind2(k)),1]);
      end
      PiCOVPred = CDFZZ-PiMeanPred.*PiMeanPred2;
   end
   PiCOVPred(isnan(PiCOVPred)) = 0;
end

function AcqValue = PVCFun(GPRmodel,x, XMC, CovPredGP, gamma,j,PiMean,PiMean_MC)
%%Define the smooth indicator function smoothed by gamma
   Nx = size(x,1);
   NMC = size(XMC(:,:,j-1),1);
   CovValue = CovPredGP(x,XMC(:,:,j-1));
   [MeanPred,SDPRed,~] = predict(GPRmodel,x);
   [MeanPredXMC,SDPRedXMC,~] = predict(GPRmodel,XMC(:,:,j-1));
   if gamma == Inf% in case gamma is infinite
      rho = CovValue./(repmat(SDPRed,1,NMC).*repmat(SDPRedXMC',Nx,1));
      rho_Vec = reshape(rho,Nx*NMC,1);
      ZZ = -MeanPred./SDPRed;
      ZZMC = -MeanPredXMC./SDPRedXMC;
      PiMeanPred = normcdf(-ZZ);
      PiMeanPredXMC = normcdf(-ZZMC);
      PiMeanPred_Mat = repmat(PiMeanPred,1,NMC);
      PiMeanPredXMC_Mat = repmat(PiMeanPredXMC',Nx,1);
      Term2_Vec = reshape(PiMeanPred_Mat.*PiMeanPredXMC_Mat,Nx*NMC,1);
      ZZ_Vec = reshape(repmat(ZZ,1,NMC),Nx*NMC,1);
      ZZMC_Vec = reshape(repmat(ZZMC,Nx,1),Nx*NMC,1);
      Ind2 = find(Term2_Vec>1e-4&Term2_Vec<0.99);%indices of samples need to be computed via 2D CDF 
      CDF2_Vec = Term2_Vec;
      for i=1:length(Ind2)
          CDF2_Vec(Ind2(i),:) = mvnpdf([ZZ_Vec(Ind2(i),:),ZZMC_Vec(Ind2(i),:)],[0,0],[1,rho_Vec(Ind2(i),:);rho_Vec(Ind2(i),:),1]);
      end
      CDF2 = reshape(CDF2_Vec,Nx,NMC);
      CovPi = CDF2-PiMeanPred_Mat.*PiMeanPredXMC_Mat;
   else
      e = sqrt(1+gamma^2*SDPRed.^2);
      eMC = sqrt(1+gamma^2*SDPRedXMC.^2);
      rho_ee = -gamma^2*CovValue./(repmat(e,1,NMC).*repmat(eMC',Nx,1));
      rho_ee_Vec = reshape(rho_ee,Nx*NMC,1);
      Ind2 = find(abs(rho_ee_Vec)>0.02);%index of samples need to be computed via mvncdf
      Ind1 = setdiff((1:Nx*NMC)',Ind2);%%can be computed by normcdf

      PiMeanPred = normcdf(-gamma*MeanPred./e);
      PiMeanPredXMC = normcdf(-gamma*MeanPredXMC./eMC);
      ZZ = -gamma*MeanPred./e;
      ZZMC = -gamma*MeanPredXMC./eMC; 
      ZZ_Vec = reshape(repmat(ZZ,1,NMC),Nx*NMC,1);
      ZZMC_Vec = reshape(repmat(ZZMC,Nx,1),Nx*NMC,1);
      CDF2_Vec(Ind1,:) = normpdf(ZZ_Vec(Ind1,:)).*normpdf(ZZMC_Vec(Ind1,:));
      for i=1:length(Ind2)
          CDF2_Vec(Ind2(i),:) = mvnpdf([ZZ_Vec(Ind2(i),:),ZZMC_Vec(Ind2(i),:)],[0,0],[1,-rho_ee_Vec(Ind2(i),:);-rho_ee_Vec(Ind2(i),:),1]);
      end
      CDF2 = reshape(CDF2_Vec,Nx,NMC);
         CovPi = CDF2-repmat(PiMeanPred,1,NMC).*repmat(PiMeanPredXMC',Nx,1);
   end
     AcqValue =  ((mean(CovPi./repmat(PiMean_MC(:,j-1)',Nx,1),2))./PiMean{j-1}(x));
end