clear all;close all;clc;
n = 2;
N_train = [10,100,1000];
N_val = 10000;
p = [0.65, 0.35];
mu0 = [3 0;0 3]';

mu1 = [2 2]';
w = [0.5,0.5];

sigma0(:,:,1)=[2 0;0 1];sigma0(:,:,2)=[1 0;0 2];
sigma1=[1 0;0 1];

%generate true class labels and draw samples 
label_val = (rand(1,N_val)>=p(1));
Nc_val = [length(find(label_val==0)),length(find(label_val==1))];
gmmParameters.priors = [.65,.35];
gmmParameters.meanVectors = mu;
gmmParameters.covMatrices = sigma;
[x,componentLabels] = generateDataFromGMM(N_val, gmmParameters);

%calculate discriminant score and tau
discriminantScore = log(evalGaussian(x,mu(:,3),sigma(:,:,3)))-log(0.5*evalGaussian(x,mu(:,1),sigma(:,:,1))+0.5*evalGaussian(x,mu(:,2),sigma(:,:,2)));
tau = log(sort(discriminantScore(discriminantScore >= 0)));

%find midpoints of tau
mid_tau = [tau(1)-1 tau(1:end-1)+diff(tau)./2 tau(length(tau))+1];

%make decision for every threshold and calculate error values
for i = 1:length(mid_tau)
    decision = (discriminantScore>=mid_tau(i));
    pFA(i) = sum(decision == 1 & label_val == 0)/Nc_val(1);
    pCD(i) = sum(decision == 1 & label_val == 1)/Nc_val(2);
    pE(i) = pFA(i)*p(1)+(1-pCD(i))*p(2);
end

%find minimum error
[min_error,min_index] = min(pE);
min_decision = (discriminantScore>=mid_tau(min_index));
min_FA = pFA(min_index);min_CD=pCD(min_index);

%plot ROC curve
figure(1);
plot(pFA,pCD,'-',min_FA,min_CD,'o');
title('MInimum Expected Risk ROC Curve');
legend('ROC Curve','Calculated Min ERROR');
xlabel('P_{False Alarm}');ylabel('P_{Correct Decision}');

