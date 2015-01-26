function Z = kernelPCAtest(Xtest,Xtrain,alpha,d)
% Project the test data onto the kernel pricipal components using a
% polynomial kernel of degree d.
%
% Input: 
%       Xtest   Test data set where each column is a data point
%       Xtrain  Training data set where each column is a data point
%       alpha   Matrix containing the eigenvectors alpha_i
%       d       Degree of polynomial kernel
%
% Output:
%       Z       The projection onto the eigenvectors
%

% Compute the dotproduct matrix (Thaya's way)
K = Xtrain'*Xtest;
K = K.^d;
% Project onto the eigenvectors
Z = alpha*K;
end