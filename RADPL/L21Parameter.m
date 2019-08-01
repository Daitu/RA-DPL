function [Matout] = L21Parameter(Mat,tau)
%% L21Parameter 该函数用来计算L21范数的添加参数
%   input
% cellMat:输入为一个矩阵
% tau: 一个很小的参数
% output :
% Matout : cellMatout_ii =1/(2*||cellMat^i||+tao) ,为对角矩阵
%% 
num = size(Mat,1);
Matout = Mat;
M  = 1:num;
for k  = 1:num
    M(k) = 1 / (norm(Matout(k,:))*2 + tau);
end
Matout = diag(M);
end

