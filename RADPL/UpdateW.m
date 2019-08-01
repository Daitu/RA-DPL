function [ W_Mat ] = UpdateW( Data, P_Mat, W_Mat,tau)
%% Update W , This function undata discriminative adaptive representation
% input:
% Data: The original data array, each classify is an array matrix
% P_Mat:Dictionary cell P, each classify is an array matrix
% tau : Prevent matrix singular 
% DictSize : Dictionary D automs number
% W_Mat:Adaptive Representations W_(t-1)
%------------------------------------------------
% output:
% W_Mat : Adaptive Representations W_t
%============================================================
%% class number
ClassNum = size(Data,2);
% I_Mat    = eye(DictSize,DictSize);
for i=1:ClassNum
    %TempD       = Dict{i};
    TempData       = Data{i};
    Temp_P = P_Mat{i};
    % cumputer W
    Temp_A = TempData'*TempData+TempData'*Temp_P'*Temp_P*TempData;
    Temp_A = Temp_A +eye(size(Temp_A)) + eye(size(Temp_A))*tau;
    Temp_W = Temp_A\(TempData'*TempData+TempData'*Temp_P'*Temp_P*TempData);

    W_Mat{i} = Temp_W;

end

