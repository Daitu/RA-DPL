 function [ DictMat,P_Mat,W_Mat,S_Mat, EncoderMat,ofv ] = TrainRADPL( Data, Label,...
    DictSize, tau, alpha, beta,gama, Iter )
%% This is the RADPL training function
% Robust projective Dictionary pair learning by discriminative adapive 
% representations training code
% Input arguments:
%  Data : Train Data,every class is one of cell;Data{i} class i train data
%  Label : Train Label;example:[1,1,1,1,........,i,i,i,......,k,k]
%  DictSize : Dictionary size,The number of atoms per sub Dictionaries
%  alpha : Robust projective parameter
%  beta : discriminative adaptive representation parameter
%  Iter : max Iter times
%  tau : a small number,avoid the singularity issue
%  gama: parameter
% output:
% Dict_D : Dictionary D
% Dict_P : Dictionary P
% W_Mat : class reconfiguration matrix W
% S_Mat : sparse code S
% EncoderDP : D*P
% ofv : object function value
%%
% Initilize D and P , precompute the update W for one time 
[ DataMat, DictMat, P_Mat, DataInvMat, W_Mat,S_Mat ] = Initilization( Data , Label, DictSize, tau);
% Alternatively update P, D and S
ofv = 1:Iter;
for i=1:Iter
    [ P_Mat ]   = UpdateP(  S_Mat,  W_Mat, P_Mat, DataMat,DataInvMat, tau ,alpha,beta,gama);
    [ S_Mat ]   = UpdateS(  S_Mat,  DictMat, P_Mat, DataMat, tau ,gama);
    [ DictMat] = UpdateD(  DictMat, S_Mat,DataMat,tau);
    W_Mat = UpdateW( DataMat, P_Mat,  W_Mat,tau);
    ofv(i) = objectfunvalue(Data, Label,DictMat,P_Mat,W_Mat,alpha,beta);
end

% Reorganize the D * P  matrix to make the classification fast
EncoderMat = cell(size(P_Mat));
for ii = 1:size(EncoderMat,2)
    EncoderMat{ii} = DictMat{ii}*P_Mat{ii};
end


    
