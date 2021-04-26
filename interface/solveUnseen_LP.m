function [ F1, F2 ] = solveUnseen_LP( S, X1, X2, gamma, H)

% LP 求解未知样本的映射,采用线性映射方法。
% 输入：
%	S：学习到的 hash 矩阵 N*bits
%	X1,X2：两个模态的训练数据，维度为 d1*n 和 d2*n
%	gamma : 正则化参数，防止矩阵不可逆 ，通常取0.1或0.01
%   H：可选是否输入。已学习到的训练集 hash 矩阵  N*bits
% 输出：
%	F1：X1 到 S 的映射，维度为 bits*d1
%	F2：X2 到 S 的映射，维度为 bits*d2
S = sign(S);
flag = false;
if exist('H', 'var')
    flag = true;
end

[row, ~] = size(X1);
[rowt, ~] = size(X2);
F1 = S * X1' / (X1 * X1' + gamma * eye(row));
if flag
    F2 = H * X2' / (X2 * X2' + gamma * eye(rowt));
else
    F2 = S * X2' / (X2 * X2' + gamma * eye(rowt));
end

end