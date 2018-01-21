function obj = boolLassoObj(R,lambda,Z,c, beta)

%Square Loss
%D = R-M;
%loss = sum(sum(D.*D))/2;

Zc = Z * diag(sqrt(c));
[n,k] = size(Z);

%Poisson Loss
ind = find(R);
[I,J,V] = find(R);
loss = sum( -R(ind).*log(1-exp( -beta*sum(Zc(I,:).*Zc(J,:),2) )+1e-10) ...
				-R(ind).*beta.*sum(Zc(I,:).*Zc(J,:),2) ) + beta.*(ones(1,n)*Zc)*(Zc'*ones(n,1));

%objective
%obj = loss + lambda*sum(c);
obj = loss ;
