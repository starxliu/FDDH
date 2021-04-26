
%If you use our codes, we are appreciated if you appropriately cite our work.
%	X. Liu*, X.Z. Wang, Y.M. Cheung, FDDH: Fast Discriminative  Discrete  Hashing for Large-Scale Cross-Modal Retrieval. IEEE Transactions on Neural Networks and Learning Systems, in press.


function [ B,L, iter, currentF ] = solveFDDH( L ,X1, X2, mu, theta, gamma1 ,bits )

% SOLVE 
% minimize lamda1 * ||B - X1 * U1||_{F}^{2} + lamda2 * ||B - X2 * U2||_{F}^{2})
%          +  ||C * L - B ||_{F}^{2} + mu * ||X1 - R1 * C * L||_{F}^2 +
%          theta * ||X2 - R2 * C * L||_{F}^2 +
%          + gamma * (||U1||_{F}^{2} + ||U2||_{F}^{2} + ||S||_{F}^{2})
% input£º
%   L : label matrix£¬ c * n
% 	X1,X2£ºtraining data of two modalities£¬ d1*n ºÍ d2*n
%	lamda12£ºparameter
%   gamma : regularization parameters often set at 0.1 or 0.01
% 	bits£º hash length
% output£º
%	B£ºcommon hash codes , {-1,1}
%   iter£ºiteration number
%   currentF£ºfinal loss

%% initializaiton
[d1, col] = size(X1);
[d2,~] = size(X2);

B = sign(normrnd(0, 1, bits, col));
R1 = rand(d1);
R2 = rand(d2);
R1 = R1(:,1:bits);
R2 = R2(:,1:bits);

threshold = 1e-4;
lastF = inf;
iter = 1;
maxiter = 20;
L0 = L;
ind1 = L0>0;
ind2 = L0 == 0;
k = inf;
param1 = 1/((mu + theta + gamma1));
param2 = mu/(mu + theta + gamma1);
param3= theta/(mu + theta + gamma1);

while (iter<maxiter)
    % update C
    [U ,~ ,V] = svd((B + mu * R1' * X1 + theta * R2' * X2) * L',0);
    C= U  * V';
    [U ,~ ,V] = svd(X1 * L' * C',0);
    R1 = U * V';
    [U ,~ ,V] = svd(X2 * L' * C',0);
    R2 = U * V';    
    
    B0 = B;
    
    % update B
    B = sign(C * L);
        
    % update L
    L = param1 * C' * B + param2 * C' * R1' * X1 + param3 * C' * R2' * X2;
    L(ind1) = e_dragging(L(ind1),1,k);
    L(ind2) = e_dragging(L(ind2),-k,0);
    
    % compute objective function
    norm1 = norm(B - C * L, 'fro') ^ 2;
    norm2 = norm(X1 - R1 * C * L,'fro') ^ 2;
    norm3 = norm(X2 - R2 * C * L,'fro') ^ 2;
    norm4 = gamma1 * norm(L,'fro') ^ 2;
    currentF= norm1 + mu * norm2 + theta * norm3 + norm4;
    if (lastF - currentF) < threshold * currentF
        if((lastF - currentF) <=  0)
            B = B0;
            currentF = lastF;
        end
        return;
    end
    iter = iter + 1;
    lastF = currentF;
end
end