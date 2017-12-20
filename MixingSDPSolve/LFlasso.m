%% solving the following problem:
%%
%%     min \frac{1}{2}\|R-M\|^2_F  + lambda*\|M\|_S
%% 
%% where \|.\|_S is atomic norm based on the atomic set:
%%
%%     S = { zz' | z_i\in {0,1} }.
%%
%% Our solver reterns c and Z=[z_1,...,z_k] s.t.  M=\sum_j c_j* z_jz_j'.

function  [c, Z,W] = LFlasso(R_true,R_test,R0_sp,R1_sp,lambda,Z0,W0,T,threshold)

TOL = 1e-5;
T2 = 100;
SDP_rank = 10;
SDP_iter = 100;
mu = 10000
f = @(Z1,z0,thd) sum( Z1~=(z0*ones(1,size(Z1,2))) ) <= thd;
n = size(R0_sp,1);

Z = [];
c = [];
W =[];
tol_rate = 0.01;
best_auc= 0;

for t = 1:T


	t
	% compute A and use it to solve maxcut for Z empty and not empty
	if length(c)==0
		A = setA_dense(zeros(n,1),zeros(1,n),R0_sp,R1_sp);
		z = MixMaxCutSparseAAT(A,SDP_rank,SDP_iter);
	else
		Zc = Z * diag(sqrt(c));
		[W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),n),lambda);
		A = setA_dense(Zc,W_o,R0_sp,R1_sp);
		z = MixMaxCutSparseAAT(A,SDP_rank,SDP_iter);
	end

	Z = [Z z];
	c = [c;0];
	'maxcut done'




	
	%fully corrective by prox-GD


	k = length(c);
	h = diag_hessian(Z);

	eta = 1e-3/(max(h)*k); %step size

	for t2 = 1:T2

		grad_c = gradient_c_fast(R0_sp,R1_sp,Z,c,mu);

		c = prox( c - eta*grad_c, eta*mu);

	end
	'prox done'


	tic;
	%shrink c and Z for j:cj=0
	Z = Z(:,c'>TOL);
	c = c(c>TOL);
	
	%dump info
	if mod(t,5)==0
			%M = compute_M(c,Z);
			%obj = boolLassoObj(R,lambda,M,c, beta);

			match = zeros(1,length(c));
			for k = 1:size(Z0,2)
					match_k = f(Z,Z0(:,k),n*tol_rate);
					match(match_k>0)=k;
			end
			P = [match;c'];
			P


			[obj,R_guess,W] = LFLassoObj(R0_sp,R1_sp,lambda,Z,c);
			Rt_guess = 1./(1+exp(-Z0*W0));
			[acc,rec,prec,F1,F1_2,auc] = all_statistic(R_true,R_test,R_guess,Z,1,threshold);
			[~,~,~,~,~,g_auc] = all_statistic(R_true,R_test,Rt_guess,Z,1,threshold);
			fprintf('t=%f auc=%f g_auc=%f obj=%f\n',t,auc,g_auc,obj);
			for u=1:length(threshold)
				fprintf('threshold=%f K=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));					
			end	
			if  auc > best_auc 
				best_auc = auc;
			end
	end

end
end

function M = compute_M(c,Z)
	
	[n,k] = size(Z);
	M = zeros(n,n);
	for j = 1:k
		M = M + c(j)*Z(:,j)*Z(:,j)';
	end
end


function grad_M = gradient_M(R,M, beta)
	
	grad_M = -R .* beta .* exp(-beta*M) ./ (1-exp(-beta*M)+1e-2) + (1-R)*beta;
end


function grad_c = gradient_c(R,Z,c, beta)
	
	k = length(c);
	grad_c = zeros(k,1);
	%for j=1:k
	%	ZTzj = Z'*Z(:,j); %k by 1
	%	grad_c(j) = -Z(:,j)'*R*Z(:,j) + 0.5*c'*(ZTzj.^2) + 0.5*c(j)*ZTzj(j).^2;
	%end
	grad_M = gradient_M(R,compute_M(c,Z), beta);
	for j=1:k
		grad_c(j) = Z(:,j)'*grad_M*Z(:,j);
	end
end

function grad_c = gradient_c_fast(R0_sp,R1_sp,Z,c,lambda)

	k = length(c);
	grad_c = zeros(k,1);
	Zc = Z * diag(sqrt(c));
	[W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),size(R0_sp,1)),lambda);
	A = setA_dense(Zc,W_o,R0_sp,R1_sp);

	for j=1:k
		grad_c(j) = -Z(:,j)'*A*A'*Z(:,j);
	end
end

function h = diag_hessian(Z)
	k = size(Z,2);
	h = zeros(k,1);
	for i = 1:k
		h(i) = (Z(:,i)'*Z(:,i)).^2;
	end
end

function c2 = prox( c, lambda )
	
	c2 = c;
	c2(c<=lambda)=0;
	c2(c>lambda) = c(c>lambda)-lambda;
end
function A = setA_dense(Z,W,R0_sp,R1_sp)
	% -grad_L(ZW)

	[I1,J1,V1] = find(R0_sp);
	ind1 = find(R0_sp);
	E_1 = sum(Z(I1,:).*(W(:,J1)'),2);
	temp1 = 1./(1+exp(-E_1));
	[I2,J2,V2] = find(R1_sp);
	ind2 = find(R1_sp);
	E_2 = sum(Z(I2,:).*(W(:,J2)'),2);
	temp2 = -exp(-E_2)./(1+exp(-E_2));
	n = size(Z,1);
	S1 = sparse(I1,J1,temp1,n,n);
	S2 = sparse(I2,J2,temp2,n,n);
	A = -S1-S2;
	% E = Z*W;
	% R = full(R1_sp);
	% A = -(-R.*exp(-E)+1-R)./(1+exp(-E));
end
function [obj,R_guess,W_o] = LFLassoObj(R0_sp,R1_sp,lambda,Z,c)
	Zc = Z * diag(sqrt(c));
	[W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Z,2),size(R0_sp,1)),lambda);
	E = Zc*W_o;
	R_guess = 1 ./ (1+exp(-Zc*W_o));
	R = full(R1_sp);
	obj = sum(sum(-R.*log(1./(1+exp(-E)))-(1-R).*log(1-1./(1+exp(-E)))))+0.5*lambda*norm(W_o,'fro')^2;
end
