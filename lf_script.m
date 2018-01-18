
n=1000
k=10
unbalanced = 0;
rand('seed',10);
randn('seed',10);
sample_rate = 0.1;

Z_true = zeros(n,k);
for i=1:n
	temp = ceil(rand*k);
	Z_true(i,temp)=1;
end

W_true = randn(k,n)*10;
if unbalanced == 1
	W_true = W_true-10;
end
R = 1 ./ (1+exp(-Z_true*W_true));
Rb = double(rand(n,n) < R);

ratio = 1-(sum(sum(Rb)))/(n*n);



Sr = double(rand(n,n) > 1-sample_rate); 

E = Z_true*W_true;




R1 = sparse(Rb.*Sr);
R0 = sparse((1-Rb).*Sr);

lambda = 0.1;
T = 300;
threshold = [0.5];
tic;
[c, Z_guess,W_guess] = LFlasso(Rb,1-Sr,R0,R1,lambda,Z_true,W_true,T,threshold,unbalanced);
toc;
%[c, Z_guess,W_guess] = LFlasso_over(R0,R1,lambda,Z_true,W_true,T,threshold);
%R_guess = double(1./(1+exp(-Z_guess*W_guess))>0.5);