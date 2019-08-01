function [ S ] = compute_S(X,knn,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��Ȩ�ؾ���S
% knn =2;
N = size(X,2); 

%%% weight construction, optimal
% tol = 0; 

X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;
[sorted,index] = sort(distance); %[B1,INDEX] = sort(A)��A�������򣬿���ȡ������INDEX���������Բ�ѯB1Ԫ����C����һ����Ӧ��
neighborhood = index(2:(1+knn),:);
S = zeros(knn,N);
for i = 1:N
   z = X(:,neighborhood(:,i))-repmat(X(:,i),1,knn);        % shift ith pt to origin
   C = z'*z;                                             % local covariance
   C = C + eye(knn,knn)*tol*trace(C);                        % regularlization (K>D)
   S(:,i) = C\ones(knn,1);                                 % solve Cw=1
   S(:,i) =  max(S(:,i),0);  %% non-negative constraint
   S(:,i) = S(:,i)/sum(S(:,i));                          % enforce sum(w)=1
end; 

   S_temp = zeros(N,N);          %����һ��N*N�������
   for i = 1:N
        S_temp(neighborhood(:,i),i) = S(:,i);
   end
   clear S;
   S = S_temp; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% weight construction, optimal

S = (S+S')/2; 

sum_S = sum(S'); %a=sum(x);�����,��sum_SΪһλ����

S = (diag(sum_S))^(-1/2)*S*(diag(sum_S))^(-1/2);

%%% end weight construction, optimal
%S = S./repmat(sum(S')',1,size(X,2));  %% stochastic matrix P = inv(D)*W; ./ͬ�׶�ӦԪ�����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end weight construction, optimal


