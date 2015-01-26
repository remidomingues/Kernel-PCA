clear all

%% Load the data

load('usps_resampled.mat');

Ntrain      = 3000;
Ntest       = 2000;
Xtrain      = train_patterns(:,1:Ntrain);
Xtest       = test_patterns(:,1:Ntest);
train_index = train_index(1:Ntrain);
test_index  = test_index(1:Ntest);

clear train_patterns test_patterns train_labels test_labels

%% Setting some parameters
d = 2;   % Order of polynomial for polynomial kernel
n = 128; % Number of Principal components required

%% Running the kernel PCA
%Training kernel
%[Y, eVtr, ~] = train_kpca(Xtrain,n,d);
[alpha,Y] = kernelPCAtrain(Xtrain,d,n);

%%
%Testing kernel matrix
%Z = test_kpca(Xtest,Xtrain,eVtr,d);
Z = kernelPCAtest(Xtest,Xtrain,alpha,d);


%% Classification

% Only works with LIBSVM: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
err = classUsingSVM(Y,Z,train_index,test_index); 



%class = classify(Z',Y',train_index);
%error_testing=sum(class'~=test_index)/length(class);



