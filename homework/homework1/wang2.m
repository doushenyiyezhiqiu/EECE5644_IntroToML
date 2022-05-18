clear all, close all,
C = 4;
N = 10000;
n = 3;
symbols='ox+*';
%Parameters for different classes
params.mean_scaling=4*(0:C-1);
params.Cov_scaling=3*(1:C) ;
gmmParameters.priors = [0.3 0.3 0.4];
gmmParameters.meanVectors(1,:) = params.mean_scaling;
gmmParameters.meanVectors(2,:) = params.mean_scaling;
gmmParameters.meanVectors(3,:) = params.mean_scaling;
%Define loss matrices
lossMatrix={'minErr' 'deltaU10' 'deltaU100'};
lossMatrixA.minErr = ones(C-1,C-1)-eye(C-1);
lossMatrixA.deltaU10= [0 1 10;
1 0 10;
1 1 0];
lossMatrixA.deltaU100= [0 1 100;
1 0 100;
1 1 0];
for ind = 1:C
gmmParameters.covMatrices(:,:,ind) =params.Cov_scaling(ind)*eye(n);
%A = params.Cov_scaling(ind)*eye(n);
%gmmParameters.covMatrices(:,:,ind) = A'*A; % arbitrary covariance matrices
end
% Generate data from specified pdf
[x,labels] = generateDataFromGMM(N,gmmParameters);
for ind = 1:3
Nclass(ind,1) = length(find(labels==ind));
end
C=C-1;
% Shared computation for both parts
for ind = 1:3
pxgivenl(ind,:) =evalGaussianPDF(x,gmmParameters.meanVectors(:,ind),gmmParameters.covMatrices(:,:,ind));
end
px = gmmParameters.priors*pxgivenl;
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,C,1);
%Plot Data with True Labels
figure;
for ind=1:C
plot3(x(1,labels==ind),x(2,labels==ind),x(3,labels==ind),symbols(ind),'DisplayName',['Class ' num2str(ind)]);
hold on;
end
xlabel('x1');
ylabel('x2');
grid on;
title('X Vector with True Data(True Classes)');
legend 'show';
%Classify Data based on loss values
for ind3=1:length(lossMatrix)
expectedRisksA.(lossMatrix{ind3}) =lossMatrixA.(lossMatrix{ind3})*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
[~,decisionsA.(lossMatrix{ind3})] =min(expectedRisksA.(lossMatrix{ind3}),[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP
fDecision_ind.(lossMatrix{ind3})=(decisionsA.(lossMatrix{ind3})~=labels);%Incorrect classificiation vector
%Confusion Matrix
confMatrix.(lossMatrix{ind3})=zeros(C,C);
for ind=1:C
for ind2=1:C
confMatrix.(lossMatrix{ind3})(ind,ind2)=sum(decisionsA.(lossMatrix{ind3})==ind& labels==ind2)/Nclass(ind2);
end
end

%Expected Risk
ExpRisk.(lossMatrix{ind3})=...
gmmParameters.priors*diag(expectedRisksA.(lossMatrix{ind3})'...
*confMatrix.(lossMatrix{ind3}));
fprintf('Expected Risk for %s=%1.2f\n',...
lossMatrix{ind3},ExpRisk.(lossMatrix{ind3}));
%Plot Decisions
figure;
for ind=1:C
class_ind=decisionsA.(lossMatrix{ind3})==ind;
plot3(x(1,class_ind),x(2,class_ind),x(3,class_ind),symbols(ind),...
'DisplayName',['Class ' num2str(ind)]);
hold on;
end

xlabel('x1');
ylabel('x2');
grid on;
title('X Vector with Classified (ERM) Values');
legend 'show';
%Plot Decisions with Incorrect Results as specified in assignment
figure;
for ind=1:C
class_ind=decisionsA.(lossMatrix{ind3})==ind;
plot3(x(1,class_ind & ~fDecision_ind.(lossMatrix{ind3})),...
x(2,class_ind & ~fDecision_ind.(lossMatrix{ind3})),...
x(3,class_ind & ~fDecision_ind.(lossMatrix{ind3})),...
symbols(ind),'Color',[0.39 0.83 0.07],'DisplayName',...
['Class ' num2str(ind) ' Correct Classification']);
hold on;
plot3(x(1,class_ind & fDecision_ind.(lossMatrix{ind3})),...
x(2,class_ind & fDecision_ind.(lossMatrix{ind3})),...
x(3,class_ind & fDecision_ind.(lossMatrix{ind3})),...
['r' symbols(ind)],'DisplayName',...
['Class ' num2str(ind) ' Incorrect Classification']);
hold on;
end
xlabel('x1');
ylabel('x2');
grid on;
title('X Vector with Correct and Incorrect Classifications');
legend 'show';
%Plot Decisions with Incorrect Decisions
figure;
for ind2=1:C
subplot(4,1,ind2);
for ind=1:C
class_ind=decisionsA.(lossMatrix{ind3})==ind;
plot3(x(1,class_ind),x(2,class_ind),x(3,class_ind),symbols(ind),'DisplayName',...
['Class ' num2str(ind)]);
hold on;
end

plot3(x(1,fDecision_ind.(lossMatrix{ind3}) & labels==ind2),...
x(2,fDecision_ind.(lossMatrix{ind3}) & labels==ind2),...
x(3,fDecision_ind.(lossMatrix{ind3}) & labels==ind2),...
'g.','DisplayName','Incorrectly Classified');
ylabel('x2');
grid on;
title(['X Vector with Incorrect Decisions for Class ' num2str(ind2)]);
if ind2==1
legend 'show';
elseif ind2==4
xlabel('x1');
end
end
end




function g = evalGaussian(x ,mu,Sigma)
%Evaluates the Gaussian pdf N(mu, Sigma ) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2); %coefficient
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);%exponent
g = C*exp(E); %finalgaussianevaluation
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%From g/Code/generateDataFromGMM.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N);
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
indl = find(u <= thresholds(l)); Nl = length(indl);
labels(1,indl) = l*ones(1,Nl);
u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%From g/Code/evalGaussianPDF.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function px = evalGaussianPDF(x,mu,Sigma)
% x should have n-dimensional N vectors in columns
n = size(x,1); % data vectors have n-dimensions
N = size(x,2); % there are N vector-valued samples
C = (2*pi)^(-n/2)*det(Sigma)^(-1/2); % normalization constant
a = x-repmat(mu,1,N); b = inv(Sigma)*a;
% a,b are preparatory random variables, in an attempt to avoid a for loop
px = C*exp(-0.5*sum(a.*b,1)); % px is a row vector that contains p(x_i) values
end
