function [ P_Mat ] = UpdateP(  S_Mat, W_Mat, P_Mat, Data, DataInvMat, tau, alpha, beta,gama)
%% Update P
% input:
% Dict: Dictionary D
% W_Mat : Adaptive Representations 
% Data: The original data array, each classify is an array matrix
% P_Mat: Dictionary array P, each classify is an array matrix
% DataInvMat :  (~X_i)*(~X_i)^T
% tau : Prevent matrix singular additions
% alpha : Robust projective parameter
% beta : discriminative adaptive representation parameter

%------------------------------------------------
% output:
% P_Mat : Dictionary array P, each classify is an array matrix
%============================================================
%% cumpater

ClassNum = size(S_Mat,2);

for i=1:ClassNum
    Temp_W = W_Mat{i};
    Temp_P = P_Mat{i};
    Temp_S = S_Mat{i};
    Temp_Data = Data{i};
    % (~X_l)*(~X_l)^T
    Temp_Da_i = DataInvMat{i};
    %======================================================
    % Q = (I-W)(I-W)^T    
    I = eye(size(Temp_W));
    Q = (I - Temp_W)*(I -Temp_W)';
    % cumputer M ; H
    % cumputer M
    Temp_M = Temp_S' - Temp_Data'*Temp_P';
    M = L21Parameter(Temp_M,tau);

    % cumputer H
    H = L21Parameter(Temp_P',tau);
    H = ones(size(H));

    % ===============================================
    Temp_A = 4*gama*Temp_Data*M*Temp_Data'+2*alpha*Temp_Da_i+4*alpha*H+beta*Temp_Data*Q*Temp_Data'+beta*Temp_Data*Q'*Temp_Data';
    Temp_A = Temp_A + tau*eye(size(Temp_A));
    P_Mat{i} = (4*gama*Temp_S*M*Temp_Data')/Temp_A;

end

