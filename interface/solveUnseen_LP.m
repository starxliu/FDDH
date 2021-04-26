function [ F1, F2 ] = solveUnseen_LP( S, X1, X2, gamma, H)

% LP ���δ֪������ӳ��,��������ӳ�䷽����
% ���룺
%	S��ѧϰ���� hash ���� N*bits
%	X1,X2������ģ̬��ѵ�����ݣ�ά��Ϊ d1*n �� d2*n
%	gamma : ���򻯲�������ֹ���󲻿��� ��ͨ��ȡ0.1��0.01
%   H����ѡ�Ƿ����롣��ѧϰ����ѵ���� hash ����  N*bits
% �����
%	F1��X1 �� S ��ӳ�䣬ά��Ϊ bits*d1
%	F2��X2 �� S ��ӳ�䣬ά��Ϊ bits*d2
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