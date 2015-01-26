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
D = 1:6;   % Order of polynomial for polynomial kernel
N = 2.^(6:11); % Number of Principal components required

%% Running the kernel PCA

ERR = nan(length(D),length(N));
%for ii = 1:2
%    d = D(ii);
%    for jj = 1:3
%        n = N(jj);
%        [alpha,Y] = kernelPCAtrain(Xtrain,d,n);
%        Z = kernelPCAtest(Xtest,Xtrain,alpha,d);
%        ERR(ii,jj)=classUsingSVM(Y,Z,train_index,test_index);
%    end
%end

for ii = 1:length(D)
    d = D(ii);
    for jj = 1:length(N)
        n = N(jj);
        [alpha,Y] = kernelPCAtrain(Xtrain,d,n);
        Z = kernelPCAtest(Xtest,Xtrain,alpha,d);
        ERR(ii,jj)=classUsingSVM(Y,Z,train_index,test_index);
    end
end

%% Plotting the errors

col = ['r','b','g','c','m','k'];
for jj = 1:length(N)
    plot(D,ERR(:,jj),['-',col(jj),'*'])
    hold on
end
set(gca,'fontsize',16)
xlabel('Polynomial degree','fontsize',16)
ylabel('Relative error','fontsize',16)
legend('64 PC','128 PC','256 PC','512 PC','1024 PC','2048 PC','location','SouthWest')
