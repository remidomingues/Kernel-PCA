function [alpha,Y] = kernelPCAtrain(X,d,n)
% Compute the kernel pricipal components using a polynomial kernel of 
% degree d.
%
% Input: 
%       X       Data set where each column is a data point
%       d       Degree of polynomial kernel
%
% Output:
%       alpha   Matrix containing the eigenvectors alpha_i
%       Y       Project of training data on to alpha
%

[~,N] = size(X);

% Compute dot product matrix
K = X'*X;
K = K.^d;

% Eigenvalues and vector of K (8)
[V,D] = eig(K);

% Normalizing expansion coefficients (9)
nV = sign(diag(D)').*(sqrt(abs(diag(D)')/N).*sum(V.^2));
V=V./repmat(nV,N,1);

% Take out the n first eigenvectors
[~,temp]=sort(diag(D),'descend');
V2=V(:,temp);
alpha = V2(1:n,:);

Y = alpha*K;

end
