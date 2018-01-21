%% solving the following problem:
%%
%%     min \frac{1}{2}\|R-M\|^2_F  + lambda*\|M\|_S
%% 
%% where \|.\|_S is atomic norm based on the atomic set:
%%
%%     S = { zz' | z_i\in {0,1} }.
%%
%% Our solver reterns c and Z=[z_1,...,z_k] s.t.  M=\sum_j c_j* z_jz_j'.

function  [nmi_g,em_g,Z,W,snap_tcom] = LFlasso_ZW(R_true,R_test,R0_sp,R1_sp,parameter,Z0,W0,datatype,out,snap_load)

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
mu = parameter(1,13)
stepsize = parameter(1,14);
mod_num = parameter(1,15);

nmi_g = zeros(1,4);
% nmi_g4 : sanp nmi
em_g = zeros(1,5);
% em_g4 :snap modularity em_g5 :ground_truth modularity
f = @(Z1,z0,thd) sum( Z1~=(z0*ones(1,size(Z1,2))) ) <= thd;
n = size(R0_sp,1);

Z = [];
c = [];
W =[];
best_auc= 0;





%first calculate modularity matrix B
B = full(R1_sp); 
m = sum(sum(B))/2;
degree = sum(B,2);
BB = (degree*degree')/(2*m);
B = B-BB;
Z6 = load(snap_load);
Z6 = Z6';
snap_tcom = size(Z6,2);
if(ground_truth_existence > 0.5)
	em_g(5) = extended_modularity(B,Z0,m); 
	nmi_g(4) = normalized_mutual_information(Z0,Z6,1);
end
em_g(4) = extended_modularity(B,Z6,m);
%can actually calculate groundtruth guess
Rt_guess=0;
g_auc=0;
if(strcmp(datatype,'Synthetic'))
	Rt_guess = 1./(1+exp(-Z0*W0));
	[~,~,~,~,~,g_auc] = all_statistic(R_true,R_test,Rt_guess,Z0,1,threshold);
	%Z0 is actually dummy variable in all_statistics
end	
snap_loss = snap_obj(R_true,Z6);
true_snap_loss = snap_obj(R_true,Z0);
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
			if(noisy > 0.5)
				for K0 = min(2,length(c)):1:length(c)
					Z2 = Z(:,ind(1:min(end,K0)));
					c2 = c(ind(1:min(end,K0))); 
					[obj,R_guess,W] = LFLassoObj(R0_sp,R1_sp,lambda,Z2,c2);
					fprintf('here1\n');
					[acc,rec,prec,F1,F1_2,auc] = all_statistic(R_true,R_test,R_guess,Z2,1,threshold);
					fprintf('here2\n');
					if(strcmp(datatype,'Synthetic'))				
						% compute mutual information
						nmi = normalized_mutual_information(Z0,Z2,1);
						% compute extended modularity 
			           	em = extended_modularity(B,Z2,m);
		             	fprintf(2,'t=%f nmi=%f snap_nmi=%f em=%f snap_em=%f groundtruth_em=%f\n',t,nmi,nmi_g(4),em,em_g(4),em_g(5));
						fprintf(out,'t=%f nmi=%f snap_nmi=%f em=%f snap_em=%f groundtruth_em=%f\n',t,nmi,nmi_g(4),em,em_g(4),em_g(5));
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
						%save('tmp');
						%exit(0);
					elseif(ground_truth_existence > 0.5 )
						% compute mutual information
						nmi = normalized_mutual_information(Z0,Z2,1);
						% compute extended modularity 
						em = extended_modularity(B,Z2,m);
		             	fprintf(2,'t=%f nmi=%f snap_nmi=%f em=%f snap_em=%f groundtruth_em=%f\n',t,nmi,nmi_g(4),em,em_g(4),em_g(5));
						fprintf(out,'t=%f nmi=%f snap_nmi=%f em=%f snap_em=%f groundtruth_em=%f\n',t,nmi,nmi_g(4),em,em_g(4),em_g(5));
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
						em = extended_modularity(B,Z2,m);
		              	fprintf('t=%f em=%f snap_em=%f\n',t,em,em_g(4));
						fprintf(out,'t=%f em=%f\n snap_em=%f\n',t,em,em_g(4));
						if(em > em_g(1))
							em_g(1) = em;
							em_g(2) = t;
							em_g(3) = K0;
						end
					end		
					%print other statistical information
					lfsnaploss = snap_obj(R_true,Z2);
					fprintf(2,'t=%f lf_snap_loss=%f snap_loss=%f  ground_truth_snap_loss=%f\n',t,lfsnaploss,snap_loss,true_snap_loss);
					fprintf(out,'t=%f lf_snap_loss=%f snap_loss=%f  ground_truth_snap_loss=%f\n',t,lfsnaploss,snap_loss,true_snap_loss);
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

function [sobj] = snap_obj(Z,R_true)
	sobj=0;
	[I1,J1,V1] = find(R_true);
	E_1 = sum(Z(I1,:).*Z(J1,:),2);
	E_1 = 1-exp(-E_1);
	E_1 = max(E_1, 1e-10);
	E_1 = sum(log(E_1));
	[I1,J1,V1] = find(1-R_true);
	E_2 = sum(sum(Z(I1,:).*Z(J1,:),2));
	sobj = E_1-E_2;
end
