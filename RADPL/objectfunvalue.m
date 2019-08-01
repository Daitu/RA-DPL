function [ofv] = objectfunvalue(TrData, TrLabel,DictMat,P_Mat,W_Mat,alpha,beta)
%%
ClassNum = length(unique(TrLabel));
ofv = 1:ClassNum;
for i=1:ClassNum
    % 第一项的值
    TempData      = TrData(:,TrLabel==i);  % classify i Data
    temp_one = TempData' - TempData'*P_Mat{i}'*DictMat{i}';
    temp_one_O = L21Parameter(temp_one,0);
    one = 2*trace(temp_one'*temp_one_O*temp_one);
    % 第2项的值
    Temp_P = P_Mat{i};
    % Calculate the not class i  data
    TempDataC     = TrData(:,TrLabel~=i); % not i classifi Data
    PX_norF = norm(Temp_P*TempDataC,'fro');
    P21 = 2*trace(Temp_P* L21Parameter(Temp_P',0)*Temp_P');
    two = alpha*(PX_norF+P21);
    % 第3项的值
    Temp_W = W_Mat{i};
    three = beta*(norm((TempData-TempData*Temp_W),'fro')+...
        norm((Temp_P*TempData-Temp_P*TempData*Temp_W),'fro')+...
        norm(Temp_W,'fro'));
    
    ofv(i) = one+two+three;
end
ofv = mean(ofv);
