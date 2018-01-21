%% solving the following problem:
%%
%%     min \frac{1}{2}\|R-M\|^2_F  + lambda*\|M\|_S
%% 
%% where \|.\|_S is atomic norm based on the atomic set:
%%
%%     S = { zz' | z_i\in {0,1} }.
%%
%% Our solver reterns c and Z=[z_1,...,z_k] s.t.  M=\sum_j c_j* z_jz_j'.

function  [nmi_g,em_g,Z,snap_tcom] = LFlasso_ZZt(R_true,R,R_test,parameter,z_beta,Z_true,data_type,reduce_info,out,snap_load)
%R = R_train
n = size(R,1);
Z = [];
c = [];
%setting parameter
ground_truth_existence = parameter(1,1);
noisy = parameter(1,2);
threshold = parameter(1,5);
T = parameter(1,6);
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
%first calculate modularity matrix B
B = full(R);
m = sum(sum(B))/2;
degree = sum(B,2);
BB = (degree*degree')/(2*m);
B = B-BB;
Z6 = load(snap_load);
Z6 = Z6';
snap_tcom = size(Z6,2);
if(ground_truth_existence > 0.5)
	em_g(5) = extended_modularity(B,Z_true,m); 
	nmi_g(4) = normalized_mutual_information(Z_true,Z6,1);
end
em_g(4) = extended_modularity(B,Z6,m);
%can actually calculate groundtruth guess
Rt_guess=0;
g_auc=0;
if(strcmp(data_type,'Synthetic'))
	Rt_guess = 1./(1+exp(-Z_true*Z_true'));
	[~,~,~,~,~,g_auc] = all_statistic(R_true,R_test,Rt_guess,Z_true,0.15,threshold);
	%Z0 is actually dummy variable in all_statistics
end	
snap_loss = snap_obj(R_true,Z6);
true_snap_loss = snap_obj(R_true,Z_true);
f = @(Z1,z0,thd) sum( Z1~=(z0*ones(1,size(Z1,2))) ) <= thd;

M = zeros(n);
prev_obj = 1e300;
for t = 1:T
	t
	%find greedy direction & add to active set
	%grad_M = gradient_M( R, M, beta );
	%[z, obj] = boolIQP(-grad_M);
	
	%grad = S - beta*11^T


	tic;

	% S = sparse(n,n);
	[I,J,V] = find(R);
	ind = find(R);
	 if length(c)>0
	  	Zc = Z * diag(sqrt(c));
	 	M_select = sum(Zc(I,:).*Zc(J,:),2);
	 	if reduce_info < 1
	 		temp = R(ind) .* (z_beta*exp(-z_beta*M_select)) ./ (1-exp(-z_beta*M_select)+lambda) + z_beta*R(ind);
	 	else
	 		temp = (z_beta*exp(-z_beta*M_select)) ./ (1-exp(-z_beta*M_select)+lambda) + z_beta;
	 	end
	 	S = sparse(I,J,temp,n,n);
	 else
	 	if reduce_info <1
	 		temp = R(ind) .* (z_beta/lambda) + z_beta*R(ind);
	 	else
	 		temp = (z_beta/lambda)*ones(size(V,1),1) + z_beta;
	 	end
	 	S = sparse(I,J,temp,n,n);
	 end
%	if length(c)>0
%	 	Zc = Z * diag(sqrt(c));
%		M_select = sum(Zc(I,:).*Zc(J,:),2);
%		S(ind)= R(ind) .* (z_beta*exp(-z_beta*M_select)) ./ (1-exp(-z_beta*M_select)+1e-2) + z_beta*R(ind);
%	else
%		S(ind) = R(ind) .* (z_beta/1e-2) + z_beta*R(ind);
%	end

	% time_collect(t,1)=toc;
	% tic;
	z = MixMaxCutComposite( S, -z_beta, ones(n,1), SDP_rank, SDP_iter );
	%z = MixMaxCutSparse( S, SDP_rank, SDP_iter );
	Z = [Z z];
	c = [c;0];
	'maxcut done'

	% time_collect(t,2)= toc; 
	%grad_M = -S + z_beta*ones(n,1)*ones(1,n);
	%[z2,obj2] = boolIQP(-grad_M);
	%diff = abs(z'*(-grad_M)*z - z2'*(-grad_M)*z2);
	%if( diff > 1e-2 )
	%			[z'*(-grad_M)*z z2'*(-grad_M)*z2]
	%			return
	%end
	
	%fully corrective by prox-GD

	tic;
	k = length(c);
	h = diag_hessian(Z);
	% time_collect(t,3)= toc; 
	eta = stepsize/(max(h)*k); %step size

	for t2 = 1:T2
		% tic;
		grad_c = gradient_c_fast(R,Z,c, z_beta,reduce_info);
		% fasttime = toc;
		% tic;
		c = prox( c - eta*grad_c, eta*mu);
		% proxtime = toc;
		% time_collect(t,4)=time_collect(t,4)+fasttime;
		% time_collect(t,5)=time_collect(t,5)+proxtime;
	end
	'prox done'


	tic;
	%shrink c and Z for j:cj=0
	Z = Z(:,c'>TOL);
	c = c(c>TOL);
	
	%dump info
	if mod(t,mod_num)==0
			%M = compute_M(c,Z);
			%obj = boolLassoObj(R,lambda,M,c, z_beta);
			obj = boolLassoObjSparse(R,mu,Z,c, z_beta);
			['t=' num2str(t) ', obj=' num2str(obj) ', stepsize=' num2str(stepsize)]
			if obj > prev_obj
					'obj incrased'
					return;
			end
			prev_obj = obj;

			%match = zeros(1,length(c));
			%for k = 1:size(Z_true,2)
			%		match_k = f(Z,Z_true(:,k),n*tol_rate);
			%		match(match_k>0)=k;
			%end
			%P = [match;c'];
			%P
			
			[~,ind] = sort(c,'descend');
			if(noisy > 0.5)
				for K0 = min(2,length(c)):2:length(c)
				%for K0 = length(c):length(c)
					Z2 = Z(:,ind(1:min(end,K0)));
					c2 = c(ind(1:min(end,K0))); 
					Zc = Z2 * diag(sqrt(c2));
					R_guess = 1 ./ (1+exp(-z_beta*Zc*Zc'));
					fprintf('here1\n');
					[acc,rec,prec,F1,F1_2,auc] = all_statistic(R_true,R_test,R_guess,Z2,z_beta,threshold);
					fprintf('here2\n');
					if(strcmp(data_type,'Synthetic'))				
						% compute mutual information
						nmi = normalized_mutual_information(Z_true,Z2,1);
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
						nmi = normalized_mutual_information(Z_true,Z2,1);
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
						em = extended_modularity(B,Zc,m);
		        fprintf(2,'t=%f em=%f snap_em=%f\n',t,em,em_g(4));
						fprintf(out,'t=%f em=%f\n snap_em=%f\n',t,em,em_g(4));
						if(em > em_g(1))
							em_g(1) = em;
							em_g(2) = t;
							em_g(3) = K0;
						end
					end		
					%print other statistical information
					lfsnaploss = snap_obj(R, sqrt(z_beta)*Zc);
					fprintf(2,'t=%f lf_snap_loss=%f snap_loss=%f \n',t,lfsnaploss,snap_loss);
					fprintf(out,'t=%f lf_snap_loss=%f snap_loss=%f \n',t,lfsnaploss,snap_loss);
					for u=1:length(threshold)
						fprintf('threshold=%f K0=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z2,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));
						% fprintf('threshold=%f K=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z2,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));
						fprintf(out,'threshold=%f K0=%d acc=%f prec=%f rec=%f F1=%f F1_2=%f\n',threshold(u),size(Z2,2),acc(u),prec(u),rec(u),F1(u),F1_2(u));
					end	
				end
				fprintf(out,'Objective=%f\n',obj);	
			end

	end
	% time_collect(t,6)= toc; 
	% fprintf(out,'Iter %d ,premax=%f maxcut=%f diag_hessian=%f gradient_c_fast=%f prox=%f dumpinfo=%f \n',t,time_collect(t,1),time_collect(t,2),time_collect(t,3),time_collect(t,4),time_collect(t,5),time_collect(t,6));
	% fprintf('Best_auc %f\n',best_auc);
	
end

end

function M = compute_M(c,Z)
	
	[n,k] = size(Z);
	M = zeros(n,n);
	for j = 1:k
		M = M + c(j)*Z(:,j)*Z(:,j)';
	end
end


function grad_M = gradient_M(R,M, z_beta)
	
	grad_M = -R .* z_beta .* exp(-z_beta*M) ./ (1-exp(-z_beta*M)+1e-2) + (1-R)*z_beta;
end


function grad_c = gradient_c(R,Z,c, z_beta)
	
	k = length(c);
	grad_c = zeros(k,1);
	%for j=1:k
	%	ZTzj = Z'*Z(:,j); %k by 1
	%	grad_c(j) = -Z(:,j)'*R*Z(:,j) + 0.5*c'*(ZTzj.^2) + 0.5*c(j)*ZTzj(j).^2;
	%end
	grad_M = gradient_M(R,compute_M(c,Z), z_beta);
	for j=1:k
		grad_c(j) = Z(:,j)'*grad_M*Z(:,j);
	end
end

function grad_c = gradient_c_fast(R,Z,c, z_beta,reduce_info)

	k = length(c);
	[n,n] = size(R);
	grad_c = zeros(k,1);
	
	%grad_M = -S + z_beta*11^T
	
	%S=sparse(n,n);
	[I,J,V] = find(R);
	ind = find(R);
	Zc = Z * diag(sqrt(c));
	M_select = sum(Zc(I,:).*Zc(J,:),2);
	 if reduce_info < 1
     	temp= R(ind) .* (z_beta*exp(-z_beta*M_select)) ./ (1-exp(-z_beta*M_select)+1e-2) + z_beta*R(ind);
    else
     	temp= (z_beta*exp(-z_beta*M_select)) ./ (1-exp(-z_beta*M_select)+1e-2) + z_beta;
     end
	S = sparse(I,J,temp,n,n);
  %S(ind)= R(ind) .* (z_beta*exp(-z_beta*M_select)) ./ (1-exp(-z_beta*M_select)+1e-2) + z_beta*R(ind);
	for j=1:k
		grad_c(j) = -Z(:,j)'*S*Z(:,j) + z_beta * (Z(:,j)'*ones(n,1))^2;
	end
end

function h = diag_hessian(Z)
	k = size(Z,2);
	h = zeros(k,1);
	for i = 1:k
		h(i) = (Z(:,i)'*Z(:,i)).^2;
	end
end

function c2 = prox( c, mu )
	
	c2 = c;
	c2(c<=mu)=0;
	c2(c>mu) = c(c>mu)-mu;
end
function [sobj] = snap_obj(R_true, Z)
	sobj=0;
	[I1,J1,V1] = find(R_true);
	E_1 = sum(Z(I1,:).*Z(J1,:),2);
	E_1 = 1-exp(-E_1);
	E_1 = E_1 + 1e-10;
	E_1 = sum(log(E_1));
	[I1,J1,V1] = find(1-R_true);
	E_2 = sum(sum(Z(I1,:).*Z(J1,:),2));
	sobj = E_2-E_1;
end

