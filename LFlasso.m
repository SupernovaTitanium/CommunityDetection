%% solving the following problem:
%%
%%     min \frac{1}{2}\|R-M\|^2_F  + lambda*\|M\|_S
%% 
%% where \|.\|_S is atomic norm based on the atomic set:
%%
%%     S = { zz' | z_i\in {0,1} }.
%%
%% Our solver reterns c and Z=[z_1,...,z_k] s.t.  M=\sum_j c_j* z_jz_j'.

function  [nmi_g,em_g,Z,W] = LFlasso(R_true,R_test,R0_sp,R1_sp,parameter,Z0,W0,datatype,out)

%setting parameter
ground_truth_existence = parameter(1,1);
noisy = parameter(1,2);
threshold = parameter(1,5);
T=parameter(1,6);
lambda =parameter(1,7);
TOL = parameter(1,8);
tol_rate = parameter(1,9);
T2 =parameter(1,10);
SDP_rank = parameter(1,11);
SDP_iter = parameter(1,12);
mu = parameter(1,13);
stepsize = parameter(1,14);
mod_num = parameter(1,15);

nmi_g = zeros(1,3);
em_g = zeros(1,4);

f = @(Z1,z0,thd) sum( Z1~=(z0*ones(1,size(Z1,2))) ) <= thd;
n = size(R0_sp,1);

Z = [];
c = [];
W =[];
best_auc= 0;





%first calculate modularity matrix B
B = full(R1_sp); 
m = sum(sum(B));
degree = sum(B,2);
BB = (degree*degree')/(2*m);
B = B-BB;

Z0n = sum(Z0,2);
Z0n(Z0n < 1)=1;
Z0n = repmat(Z0n,1,size(Z0,2));
Z4 = Z0./(Z0n);
em_g(4) = 1/(2*m)*(sum(sum(B.*(Z4*Z4'))));
gem = em_g(4);
% pending for boosting 
%can actually calculate groundtruth guess
Rt_guess=0;
g_auc=0;
if(strcmp(datatype,'Synthetic'))
	Rt_guess = 1./(1+exp(-Z0*W0));
	[~,~,~,~,~,g_auc] = all_statistic(R_true,R_test,Rt_guess,Z0,1,threshold);
	%Z0 is actually dummy variable in all_statistics
end	
for t = 1:T


	t
	% compute A and use it to solve maxcut for Z empty and not empty
	if length(c)==0
		A = setA_dense(zeros(n,1),zeros(1,n),R0_sp,R1_sp);
		z = MixMaxCutSparseAAT(A,SDP_rank,SDP_iter);
		% Z = [Z z];
		% c = [c;0];
		% Zc = Z * diag(sqrt(c));
		% [W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),n),lambda);
		% 'maxcut done'
	else
		Zc = Z * diag(sqrt(c));
		[W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),n),lambda);
		A = setA_dense(Zc,W_o,R0_sp,R1_sp);
		z = MixMaxCutSparseAAT(A,SDP_rank,SDP_iter);
		% Z = [Z z];
		% c = [c;0];
		% Zc = Z * diag(sqrt(c));
		% [W_o,~] = w_solver(R0_sp,R1_sp,Zc,Zc',zeros(size(Zc,2),n),lambda);
		% 'maxcut done'
	end

	Z = [Z z];
	c = [c;0];
	% 'maxcut done'





	
	%fully corrective by prox-GD


	k = length(c);
	h = diag_hessian(Z);

	eta = stepsize/(max(h)*k); %step size

	for t2 = 1:T2

		grad_c = gradient_c_fast(R0_sp,R1_sp,Z,c,mu);

		c = prox( c - eta*grad_c, eta*mu);
		

	end
	'prox done'


	tic;
	%shrink c and Z for j:cj=0
	Z = Z(:,c'>TOL);
	c = c(c>TOL);
	[obj,~,~] = LFLassoObj(R0_sp,R1_sp,lambda,Z,c);
	obj
	%dump info

	if mod(t,mod_num)==0
			%M = compute_M(c,Z);
			%obj = boolLassoObj(R,lambda,M,c, beta);

			match = zeros(1,length(c));
			for k = 1:size(Z0,2)
					match_k = f(Z,Z0(:,k),n*tol_rate);
					match(match_k>0)=k;
			end
			P = [match;c'];
			P
			[~,ind] = sort(c,'descend');
			if(noisy == 1)
				for K0 = min(2,length(c)):1:length(c)
					Z2 = Z(:,ind(1:min(end,K0)));
					c2 = c(ind(1:min(end,K0))); 
					[obj,R_guess,W] = LFLassoObj(R0_sp,R1_sp,lambda,Z2,c2);
					[acc,rec,prec,F1,F1_2,auc] = all_statistic(R_true,R_test,R_guess,Z2,1,threshold);
					if(strcmp(datatype,'Synthetic'))				
						% compute mutual information
						nmi = normalized_mutual_information(Z0,Z2,1);
						% compute extended modularity 
						Z3n = sum(Z2,2);
						Z3n(Z3n < 1)=1;
						Z3n = repmat(Z3n,1,size(Z2,2));
						% Z0n = sum(Z0,2);
						% Z0n = repmat(Z0n,1,size(Z0,2));
						Z3 = Z2./(Z3n);
						% Z4 = Z0./(Z0n);
		             	em = 1/(2*m)*(sum(sum(B.*(Z3*Z3'))));
		             	% gem = 1/(2*m)*(sum(sum(B.*(Z4*Z4'))));
		             	fprintf('t=%f nmi=%f em=%f groundtruth_em=%f\n',t,nmi,em,gem);
						fprintf(out,'t=%f nmi=%f em=%f\n groundtruth_em=%f\n',t,nmi,em,gem);
						if(nmi>nmi_g(1))
							nmi_g(1) = nmi;
							nmi_g(2) = t;
							nmi_g(3) = K0;
						end
						if(em > em_g(1))
							em_g(1) = em;
							em_g(2) = t;
							em_g(3) = K0;
						end
						fprintf('t=%f auc=%f g_auc=%f obj=%f\n',t,auc,g_auc,obj);
						fprintf(out,'t=%f auc=%f g_auc=%f obj=%f\n',t,auc,g_auc,obj);
					elseif(ground_truth_existence == 1 )
						% compute mutual information
						nmi = normalized_mutual_information(Z0,Z2,1);
						% compute extended modularity 
						Z3n = sum(Z2,2);
						Z3n(Z3n < 1) = 1;
						Z3n = repmat(Z3n,1,size(Z2,2));
						% Z0n = sum(Z0,2);
						% Z0n = repmat(Z0n,1,size(Z0,2));
						Z3 = Z2./(Z3n);
						% Z4 = Z0./(Z0n);
		             	em = 1/(2*m)*(sum(sum(B.*(Z3*Z3'))));
		             	% gem = 1/(2*m)*(sum(sum(B.*(Z4*Z4'))));
		             	fprintf('t=%f nmi=%f em=%f groundtruth_em=%f\n',t,nmi,em,gem);
						fprintf(out,'t=%f nmi=%f em=%f\n groundtruth_em=%f\n',t,nmi,em,gem);
						if(nmi>nmi_g(1))
							nmi_g(1) = nmi;
							nmi_g(2) = t;
							nmi_g(3) = K0;
						end
						if(em > em_g(1))
							em_g(1) = em;
							em_g(2) = t;
							em_g(3) = K0;
						end
					else
						% compute extended modularity 
						Z3n = sum(Z2,2);
						Z3n(Z3n < 1) = 1;
						Z3n = repmat(Z3n,1,size(Z2,2));
						Z3 = Z2./(Z3n);
					   	em = 1/(2*m)*(sum(sum(B.*(Z3*Z3'))));
		              	fprintf('t=%f em=%f\n',t,em);
						fprintf(out,'t=%f em=%f\n',t,em);
						if(em > em_g(1))
							em_g(1) = em;
							em_g(2) = t;
							em_g(3) = K0;
						end
					end		
					%print other statistical information
					for u=1:length(threshold)
						fprintf('threshold=%f K0=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z2,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));
						% fprintf('threshold=%f K=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z2,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));
						fprintf(out,'threshold=%f K0=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z2,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));
					end	
					% if  auc > best_auc 
					% 	best_auc = auc;
					% end
				end
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

function nmi = normalized_mutual_information(Z1, Z2, way)
% Compute NMI as described in:
% Lancichinetti, Fortunato, and Kertesz. Detecting the overlapping and 
% hierarchical community structure of complex networks. 2009.
% Input: two sparse binary matrices with size n x k1 and n x k2.
% Output: NMI score of two covers. 

if nnz(Z1) ~= nnz(Z1==1)
  error('Z1 should be a binary matrix!');
end

if nnz(Z2) ~= nnz(Z2==1)
  error('Z2 should be a binary matrix!');
end

H_x = marginal_entropy(Z1);
H_y = marginal_entropy(Z2);
	if  way~=3
		H_x_y = conditional_entropy(Z1,Z2,H_x,way);
		H_y_x = conditional_entropy(Z2,Z1,H_y,way);
		nmi = 1 - 0.5 * (H_x_y + H_y_x);
	else
		H_x_s = sum(H_x);
		H_y_s = sum(H_y);
		H_x_y = conditional_entropy(Z1,Z2,H_x,way);
		H_y_x = conditional_entropy(Z2,Z1,H_y,way);
		I_x_y = 0.5*(H_x_s-H_x_y+H_y_s-H_y_x);
		nmi = (I_x_y)/max(H_x_s,H_y_s);
		
	end

end


function H = marginal_entropy(Z)
% Compute marginal entropy H(X_k) for all k. 
% Return value H is a column vector of length k.
n = size(Z,1);
prob = sum(Z,1)' / n; % nnz of each column divided by n
prob = max(prob, 1e-9);
H = - prob .* log2(prob) - (1 - prob) .* log2(1 - prob);

end
% function H_joint = joint_entropy(Z1, Z2)

% end
function H_cond = conditional_entropy(Z1,Z2,H_x,way)

% Compute joint entropy H(X_k, Y_l) for all k and l
% Return value H is a k1 x k2 matrix
n = size(Z1,1);
k1 = size(Z1,2);
k2 = size(Z2,2);
H_cond_pre = zeros(k1,k2);
for c1 = 1:k1
  col1 = Z1(:,c1);
  for c2 = 1:k2
    col2 = Z2(:,c2);
    intersect_size = nnz(col1 .* col2);
    union_size = nnz(col1 + col2);
    p00 = 1 - union_size / n;
    p01 = (nnz(col2) - intersect_size) / n;
    p10 = (nnz(col1) - intersect_size) / n;
    p11 = intersect_size / n;
    prob = [p00; p01; p10; p11];
    assert(sum(prob) - 1 < 1e-9);
    prob = max(prob, 1e-9); % clip to avoid 0 * log(0)
    H_prob = -prob.*log2(prob); 
    if(way ~= 1 & (H_prob(1)+H_prob(4))<(H_prob(2)+H_prob(3)))
    	prob2 =[p00+p01;p10+p11];
      	H_cond_pre(c1,c2) =  - prob2' * log2(prob2);
    else   
    	prob2 =[p00+p10;p11+p01];	
       	H_cond_pre(c1,c2) = - prob' * log2(prob) + prob2' * log2(prob2);
   	end  
  end
end	
% Compute conditional entropy H(X|Y), a scalar.
% H_x and H_y are column vectors of length k1 and k2.
% H_cond_pre is a k1 x k2 matrix.
% Subtract H_y' from every row of H_xy, get a k1 x k2 matrix
% Find min for every row, get a column vector of length k1
H_cond = min(H_cond_pre, [], 2); 

% Normalize, average over k1
if way ~= 3
	H_cond = mean(H_cond ./ H_x);
else
	H_cond = sum(H_cond);
end

end