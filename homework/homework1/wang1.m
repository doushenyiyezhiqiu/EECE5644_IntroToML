
clear all; close all;

N = 10000; %Number of data points
n=2; %Dimensions of data
p0 = 0.7; %Prior for label 0
p1 = 0.3; %Prior for label 1
u = rand(1,N)>=p0; %Determine posteriors
%Create appropriate number of data points from each distribution
N0 = length(find(u==0));
N1 = length(find(u==1));
N=N0+N1;
label=[zeros(1,N0) ones(1,N1)];
%Parameters for two classes
mu01 = [3;0];
Sigma01 = [2 ,0 ;
0,1];
mu02 = [0;3];
Sigma02 = [1 ,0 ;
0,2];
mu1 = [2;2];
Sigma1 = [1,0;
0,1];
%Generate data as prescribed in assignment descriptio
r01 = mvnrnd(mu01, Sigma01, N0);
r02 = mvnrnd(mu02, Sigma02, N0);
r0=0.5*r01+0.5*r02;
r1 = mvnrnd(mu1, Sigma1, N1);

%Combine data from each distribution into a single dataset
x=zeros(N,n);
x(label==0,:)=r0;
x(label==1,:)=r1;

discScore=log(evalGaussian(x' ,mu1,Sigma1)./(0.5.*(evalGaussian(x' ,mu01,Sigma01)+evalGaussian(x' ,mu02,Sigma02))));
sortDS=sort(discScore);
%Generate vector of gammas for parametric sweep

%Calculate Theoretical Minimum Error
logGamma_ideal=log(p0/p1);
decision_ideal=discScore>logGamma_ideal;
pFP_ideal=sum(decision_ideal==1 & label==0)/N0;
pTP_ideal=sum(decision_ideal==1 & label==1)/N1;
pFE_ideal=(pFP_ideal*N0+(1-pTP_ideal)*N1)/(N0+N1);
%Estimate Minimum Error

fprintf('Theoretical: Gamma=%1.2f, Error=%1.2f%%\n',...
exp(logGamma_ideal),100*pFE_ideal);

function g = evalGaussian(x ,mu,Sigma)
%Evaluates the Gaussian pdf N(mu, Sigma ) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2); %coefficient
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);%exponent
g = C*exp(E); %finalgaussianevaluation
end