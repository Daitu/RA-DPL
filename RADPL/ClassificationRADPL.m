function [Error,PredictLabel] = ClassificationRADPL( TestData , EncoderMat)

%Projective representation coefficients estimation
ClassNum = size(EncoderMat,2);
% PredictCoef = EncoderMat*TestData;
Error = ones(size(EncoderMat,2),size(TestData,2));
% Class-specific reconstruction error caculation
for i = 1:ClassNum
    Error(i,:)=sqrt(sum((TestData - EncoderMat{i}*TestData).^2));
end
% [Distance PredictLabel] = min(Error);
[Error,PredictLabel] = min(Error);



