%% =========================================================================
% DL AR  Random_face_features_AR
% 2017.12.21
% Daitu
% 进行一次测试，查看函数是否正确运行
%% =========================================================================
clear;
close all;
clc;
%% Load training and testing data
DataPath   = 'Random_face_features_AR';
load(DataPath);
% splist data
% [TrData,TtData,TrLabel,TtLabel]=ExtractData(A,20,labels,12345);
[TrData,TtData,TrLabel,TtLabel]=ExtractData(A,20,labels,1210101);
%% 添加噪声
TrData = normcol_equal(TrData);
TtData = normcol_equal(TtData);

%Parameter setting
DictSize = 5;
tau    = 0.00001;
alpha = 0.000052;
beta  = 0;
Iter = 20;
gama = 0.0125;

%% DPL trainig
tic
% [ DictMat,P_Mat,W_Mat, EncoderMat ] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,Iter);
[ DictMat,P_Mat,W_Mat,S_Mat, EncoderMat] = TrainRADPL(  TrData, TrLabel, DictSize, tau, alpha, beta,gama,Iter);
TrTime = toc;

%
[~,PredictLabel] = ClassificationRADPL( TrData, EncoderMat);
Acc = sum(TrLabel==PredictLabel)/size(TrLabel,2);
disp(['训练集最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])
%% DPL testing
tic
[Error,PredictLabel] = ClassificationRADPL( TtData, EncoderMat);
Error;
% [Error,PredictLabel] = ClassificationDPL2( TtData, DictMat ,W_Mat, EncoderMat);
TtTime = toc;
%% confusion matrix
confmat = confusionmat(TtLabel,PredictLabel);
% heatmap
h = heatmap(confmat);
h.Title = 'Dictionart Learning';
h.XLabel = 'Test Label';
h.YLabel = 'Predict Label';
h.ColorScaling = 'scaledcolumns';
%% Show accuracy and time
Acc = sum(TtLabel==PredictLabel)/size(TtLabel,2);
disp(['最大值Acc：',num2str(Acc),'   alpha:',num2str(alpha),'   beta:',num2str(beta)])