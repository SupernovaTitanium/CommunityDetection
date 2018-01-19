%  use it as a general interface to run communitiy detection test data
%  usage  general_script path_to_setting_file
function [] = general_script(set_file_path)
	fprintf(2,'Reading setting file from %s\n',set_file_path);	
	fid = fopen(set_file_path,'r');
	assert(fid > 0,'Open file error\n');
	parameter = zeros(1,15);
	syn_parameter = zeros(1,6);
	tline = fgetl(fid);
	syn = 0;
	data_path ='/'; 
	data_type ='Synthetic';
	cluster_path = 'none';
	if(strcmp(tline,'Synthetic'));
		syn= 1;
	elseif(strcmp(tline,'Real'));
		syn= 0;
	else
		assert(1 < 0,'Unknown data type\n');
	end
	if(syn == 1)
		for i = 1:6
			tline = fgetl(fid);
			fprintf(2,'Try to set %s\n',tline);
			C = strsplit(tline,' ');
			syn_parameter(1,i) = str2num(C{3});
		end
	end
	if(syn == 0)
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		data_path = C{3};
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		cluster_path = C{3};
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		data_type = C{3};	
	end
	tline = fgetl(fid);
	i=1;
	while ischar(tline)
		fprintf(2,'Try to set %s\n',tline);
		C = strsplit(tline,' ');
		parameter(1,i) = str2num(C{3}); 
		i=i+1;
		tline = fgetl(fid);	
	end
	% fix random number
	rand('seed',10);
	randn('seed',10);
	fprintf(2,'Converting data to adjancy matrix')
	if(syn == 1)
		n = syn_parameter(1,4)
		k = syn_parameter(1,5)
		Z_true = zeros(n,k);
		% use Z_sparsity to set binary matrix Z_true
		nnz_row = ceil(syn_parameter(1,6)*k);
		for i=1:n
			rand1 = randperm(k);
			Z_true(i,rand1(1:nnz_row)) = 1;
		end
		I = eye(k);
		M = normrnd(syn_parameter(1,1),syn_parameter(1,2),k,k);
		V = syn_parameter(1,3)*I + (M+M')/2;
		W_true = V*Z_true';
		S = Z_true*W_true;
		R = 1 ./ (1+exp(-S));
		%mean(mean(R))
		M2 = rand(n,n);
		Rb = double( (M2+M2')/2 < R);
		%nnz(Rb)
		positive_edge = sum(sum(Rb))
		negative_edge = n*n-sum(sum(Rb))
		sample_positive_edge = positive_edge*parameter(1,3)
		sample_negative_edge = positive_edge*parameter(1,4)
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = min(sample_negative_edge/negative_edge,1);
		Sp = double(rand(n,n) > 1-positive_ratio);
		Sn = double(rand(n,n) > 1-negative_ratio);
		%deal with sample rate
		R1 = sparse(Rb.*Sp);
		R0 = sparse((1-Rb).*Sn);
		Sr = ones(n,n);
		%save('tmp1');
		%exit(0);
		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd ..;
		cur_time = datestr(clock);
		out = fopen(strcat('Result/',data_type,'/',cur_time),'w+');
		assert(out > 0,'Open file error\n');
		outresultname = strcat('Result/',data_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Open file error\n');
		fprintf(2,'Start running algorithm\n');
		[nmi_g,em_g,Z_guess,W_guess] = LFlasso(Rb,Sr,R0,R1,parameter,Z_true,W_true,data_type,out);
		fprintf(2,'Results will be written to %s\n',outresultname);
		write_to_result(outresult,parameter,syn_parameter,syn,nmi_g,em_g);
	end
	if(strcmp(data_type,'email_EU'))
		%notice: index start from zero
		node = 1005
		edge_num = 25571
		com_num = 42;
		data_tmp = load(data_path);
		data_tmp = data_tmp + 1;
		v1 = ones(edge_num,1);
		Rb = sparse(data_tmp(:,1),data_tmp(:,2),v1,node,node);
		Z_true = zeros(node,42);
		cluster_tmp = load(cluster_path);
		cluster_tmp = cluster_tmp + 1;
		for i=1:node
			Z_true(cluster_tmp(i,1),cluster_tmp(i,2)) = 1;
		end
		positive_edge = sum(sum(Rb))
		negative_edge = node*node-sum(sum(Rb))
		sample_positive_edge = positive_edge*parameter(1,3)
		sample_negative_edge = positive_edge*parameter(1,4) 
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = sample_negative_edge/negative_edge;
		Sp = double(rand(node,node) > 1-positive_ratio);
		Sn = double(rand(node,node) > 1-negative_ratio);
		%deal with sample rate
		R1 = sparse(Rb.*Sp);
		R0 = sparse((1-Rb).*Sn);
		Sr = ones(node,node);
		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd ..;
		cur_time = datestr(clock);
		out = fopen(strcat('Result/',data_type,'/',cur_time),'w+');
		assert(out > 0,'Open file error\n');
		outresultname = strcat('Result/',data_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Open file error\n');
		fprintf(2,'Start running algorithm\n');
		W_true =randn(0,1,com_num,node);
		[nmi_g,em_g,Z_guess,W_guess] = LFlasso(Rb,Sr,R0,R1,parameter,Z_true,W_true,data_type,out);
		fprintf(2,'Results will be written to %s\n',outresultname);
		write_to_result(outresult,parameter,syn_parameter,syn,nmi_g,em_g);
	end
	if(strcmp(data_type,'email'))
		%notice: index start from 1 undirected (1133,1133)?
		node = 1133
		edge_num = 10903
		data_tmp = load(data_path);
		Rb = sparse(data_tmp(:,1),data_tmp(:,2),data_tmp(:,3),node,node);
		positive_edge = sum(sum(Rb))
		negative_edge = node*node-sum(sum(Rb))
		sample_positive_edge = positive_edge*parameter(1,3) 
		sample_negative_edge = positive_edge*parameter(1,4) 
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = sample_negative_edge/negative_edge;
		Sp = double(rand(node,node) > 1-positive_ratio);
		Sn = double(rand(node,node) > 1-negative_ratio);
		%deal with sample rate
		R1 = sparse(Rb.*Sp);
		R0 = sparse((1-Rb).*Sn);
		Sr = ones(node,node);
		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd ..;
		cur_time = datestr(clock);
		out = fopen(strcat('Result/',data_type,'/',cur_time),'w+');
		assert(out > 0,'Open file error\n');
		outresultname = strcat('Result/',data_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Open file error\n');
		fprintf(2,'Start running algorithm\n');
		Z_true =double(rand(node,10)>0.8);
		W_true =randn(0,1,10,node);
		[nmi_g,em_g,Z_guess,W_guess] = LFlasso(Rb,Sr,R0,R1,parameter,Z_true,W_true,data_type,out);
		fprintf(2,'Results will be written to %s\n',outresultname);
		write_to_result(outresult,parameter,syn_parameter,syn,nmi_g,em_g);
	end
end
function []= write_to_result(fid,parameter,syn_parameter,syn,nmi_g,em_g)
	if(syn == 1)
		fprintf(fid,'mean = %f\n',syn_parameter(1,1));
		fprintf(fid,'var = %f\n',syn_parameter(1,2));
		fprintf(fid,'tunningI = %f\n',syn_parameter(1,3));
		fprintf(fid,'n = %f\n',syn_parameter(1,4));
		fprintf(fid,'k = %f\n',syn_parameter(1,5));
		fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,6));
	end
	fprintf(fid,'ground_truth_existence = %f\n',parameter(1,1));
	fprintf(fid,'noisy = %f\n',parameter(1,2));
	fprintf(fid,'positive_sample_rate = %f\n',parameter(1,3));
	fprintf(fid,'negative_sample_rate = %f\n',parameter(1,4));
	fprintf(fid,'threshold_prob_for_link = %f\n',parameter(1,5));
	fprintf(fid,'iteration_num = %f\n',parameter(1,6));
	fprintf(fid,'w_solver_lambda = %f\n',parameter(1,7));
	fprintf(fid,'c_TOL= %f\n',parameter(1,8));
	fprintf(fid,'match_tol_rate = %f\n',parameter(1,9));
	fprintf(fid,'inner_iteration_num = %f\n',parameter(1,10));
	fprintf(fid,'SDP_rank = %f\n',parameter(1,11));
	fprintf(fid,'SDP_iter = %f\n',parameter(1,12));
	fprintf(fid,'mu  = %f\n',parameter(1,13));
	fprintf(fid,'stepsize = %f\n',parameter(1,14));
	fprintf(fid,'mod = %f\n',parameter(1,15));	
	if(parameter(1,1)==1)
		fprintf(fid,'Best nmi occurs at t=%f k=%f value=%f\n',nmi_g(2),nmi_g(3),nmi_g(1));
		fprintf(fid,'Best em occurs at t=%f k=%f value=%f and ground_truth_em=%f\n',em_g(2),em_g(3),em_g(1),em_g(4));
	else
		fprintf(fid,'Best nmi occurs at t=%f k=%f value=%f\n',nmi_g(2),nmi_g(3),nmi_g(1));
		fprintf(fid,'Best em occurs at t=%f k=%f value=%f\n',em_g(2),em_g(3),em_g(1));
	end	
end

