mex  w_solver.cpp -largeArrayDims -I../locallib/include -L../locallib/lib ../locallib/lib/liblbfgs-1.10.so  CXXFLAGS='$CXXFLAGS -fpermissive -O3 -lm'
mex  all_statistic.cpp
mex  MixMaxCutSparseAAT.cpp -largeArrayDims -I./MixingSDPSolve
n=100
k=3

rand('seed',10);
sample_rate = 1.0;

Z_true = zeros(n,k);
for i=1:n
	temp = ceil(rand*k);
	Z_true(i,temp)=1;
end

W_true = randn(k,n)*10;
R = 1 ./ (1+exp(-Z_true*W_true));
Rb = double(rand(n,n) < R);


Sr = double(rand(n,n) > 1-sample_rate);

E = Z_true*W_true;

R1 = sparse(Rb.*Sr);
R0 = sparse((1-Rb).*Sr);

lambda = 1;
T = 500;
threshold = [0.5];
%[c, Z_guess,W_guess] = LFlasso(Rb,Sr,R0,R1,lambda,Z_true,W_true,T,threshold);
[c, Z_guess,W_guess] = LFlasso_over(R0,R1,lambda,Z_true,W_true,T,threshold);
R_guess = double(1./(1+exp(-Z_guess*W_guess))>0.5);