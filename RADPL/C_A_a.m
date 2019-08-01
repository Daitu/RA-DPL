%% A.	Convergence Analysis
clear;
close all;
clc;
warning off



%% b£º  For the random  AR  database, 
% we select 20 face image from each person class for training,
% and fix the number of dictionary atoms corresponding to an average of 5 items per person.
% Load training and testing data
DataPath   = 'Random_face_features_AR';
load(DataPath);
% splist data
A = normcol_equal(A);
[TrData,TtData,TrLabel,TtLabel]=ExtractData(A,20,labels,1210101);

% Parameter setting  5e-05       0.005     0.001  
DictSize = 5;
tau    = 0.00001;
alpha = 0.00005;
beta  = 0.00001;
gama = 500;
Iter = 50;
[ DictMat,P_Mat,W_Mat,S_Mat, EncoderMat,Ofv] = TrainRADPL(  TrData, TrLabel,...
        DictSize, tau,alpha, beta,gama,Iter);
[Error,PredictLabel] = ClassificationRADPL( TtData, EncoderMat);
Acc = sum(TtLabel==PredictLabel)/size(TtLabel,2);
disp(['  dictsize:',num2str(DictSize),'   tao:',num2str(tau)])
disp(['×î´óÖµAcc£º',num2str(Acc),'  alph:',num2str(alpha),'  lambda:',num2str(gama),'   beta:',num2str(beta)])      
figure
plot(1:Iter,Ofv(1:end),'rs-','Linewidth',1.5)
grid on
xlabel('Iteration Number')
ylabel('object function values')
title('AR face database')


