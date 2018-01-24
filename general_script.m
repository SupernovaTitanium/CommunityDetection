%  use it as a general interface to run communitiy detection test data
%  usage  general_script path_to_setting_file
function [] = general_script(set_file_path)
	if(nargin < 2)
			fprintf(2,'Usage:general_script [path_to_setting_file]\n');	
	end
	fprintf(2,'Reading setting file from %s\n',set_file_path);	
	fid = fopen(set_file_path,'r');
	assert(fid > 0,'Open setting file error\n');
	parameter = zeros(1,19);
	syn_parameter = zeros(1,6);
	tline = fgetl(fid);
	syn = 0;
	if(strcmp(tline,'Synthetic'));
		syn = 1;
	elseif(strcmp(tline,'Real'));
		syn = 0;
	else
		assert(1 < 0,'Unknown data type\n');
	end
	data_path ='/'; 
	data_type ='Synthetic';
	data_format = 'none';
	cluster_path = 'none';
	cluster_format = 'none';
	model_type = 'none';
	snap_path = 'none';
	snap_out = 'none';
	snap_community = 0;
	z_beta = 0;
	tline = fgetl(fid);
	fprintf(2,'%s\n',tline);
	C = strsplit(tline,' ');
	model_type = C{3};
	if(strcmp(model_type,'ZZ^t') & (syn > 0))
		syn = 2;
	end
	if(strcmp(model_type,'ZZ^t'))
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		z_beta = str2num(C{3});
	end
	tline = fgetl(fid);
	fprintf(2,'%s\n',tline);
	C = strsplit(tline,' ');
	snap_path = C{3};
	tline = fgetl(fid);
	fprintf(2,'%s\n',tline);
	C = strsplit(tline,' ');
	snap_out = C{3};
	tline = fgetl(fid);
	fprintf(2,'%s\n',tline);
	C = strsplit(tline,' ');
	snap_community = str2num(C{3});
	snap_cell = {snap_path,snap_out,snap_community};
	if(syn > 0.5 & syn < 1.5)
		for i = 1:6
			tline = fgetl(fid);
			fprintf(2,'Try to set %s\n',tline);
			C = strsplit(tline,' ');
			syn_parameter(1,i) = str2num(C{3});
		end
	end
	if(syn > 1.5)
		for i = 1:3
			tline = fgetl(fid);
			fprintf(2,'Try to set %s\n',tline);
			C = strsplit(tline,' ');
			syn_parameter(1,i) = str2num(C{3});
		end
	end
	if(syn < 0.5 )
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		data_path = C{3};
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		data_format = C{3};
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		cluster_path = C{3};
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		cluster_format = C{3};
		tline = fgetl(fid);
		fprintf(2,'%s\n',tline);
		C = strsplit(tline,' ');
		data_type = C{3};		
	end
	data_cell = {data_path,data_format,cluster_path,cluster_format,data_type,model_type};
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
	rand('seed',1);
	randn('seed',1);
	% perform unit_test
	%unittest(syn,model_type,z_beta,snap_cell,data_cell,syn_parameter,parameter);
	fprintf(2,'Converting data to adjancy matrix\n')
	if(syn < 1.5 & syn > 0.5)
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
		parameter(1,12) = positive_edge*parameter(1,12); 
		negative_edge = n*n-sum(sum(Rb))
		sample_positive_edge = positive_edge*parameter(1,3)
		sample_negative_edge = positive_edge*parameter(1,4)
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = min(sample_negative_edge/negative_edge,1);
		Sp = double(rand(n,n) > 1-positive_ratio);
		Sn = double(rand(n,n) > 1-negative_ratio);
		Strain = double(rand(n,n) > 1-parameter(1,15));
		Stest = 1-Strain;
		%deal with sample rate
		if(parameter(1,15) == 1)
			fprintf(2,'Sampling on positive/negative edge\n');
			R1 = sparse(Rb.*Sp);
			R0 = sparse((1-Rb).*Sn);
			Stest = Strain;
		else 
			fprintf(2,'Dividing into training and testing test\n');
			R1 = sparse(Rb.*Strain);
			R0 = sparse((1-Rb).*Strain);
		end		
		R2 = full(R1);
		snap_data ='none';
		if (parameter(1,16)>0.5)
			fprintf(2,'write the data to Syn_data for snap usage\n');
			C = strsplit(set_file_path,'/');
			if(7 ~= exist('Syn_data','dir'))
				mkdir('Syn_data');
			end
			snap_data = strcat('Syn_data/',C{2},'_snap.txt');
			fid5 = fopen(snap_data,'w+');
			assert(fid5>0,'Open file error\n');
			for i=1:n
				for j=1:n
					if(R2(i,j) > 0.9 )
						fprintf(fid5,'%d\t%d\n',i,j);
					end
				end
			end
			fclose(fid5);
		end	
		% whos
		% exit(0);
		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd(data_type);
		if(7 ~= exist(model_type,'dir'))
			mkdir(model_type);
		end
		cd ..;
		cd ..;
		cur_time = datestr(clock);
		outname = strcat('Result/',data_type,'/',model_type,'/',cur_time);
		out = fopen(outname,'w+');
		assert(out > 0,'Open out file error\n');
		outresultname = strcat('Result/',data_type,'/',model_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Append result file error\n');
		snapload = 'none';
		if(parameter(1,16) > 0.5)
			fprintf(2,'Running bigclam first\n');
			snapload = run_snap(snap_data,snap_path,snap_out,snap_community,n);
		end
		fprintf(2,'Start running algorithm LFlasso_ZW\n');
		[nmi_b,em_b,nmi_s,em_s,Z_guess,W_guess,snap_tcom,obj_suc] = LFlasso_ZW(Rb,Stest,R2,R0,R1,snap_cell,syn_parameter,parameter,Z_true,W_true,data_cell,out,snapload,outname,outresult);
		fclose(out);
		fclose(outresult);
	end
	if(syn > 1.5)
		n = syn_parameter(1,1)
		k = syn_parameter(1,2)
		Z_true = zeros(n,k);
		% use Z_sparsity to set binary matrix Z_true
		nnz_row = ceil(syn_parameter(1,3)*k);
		for i=1:n
			rand1 = randperm(k);
			Z_true(i,rand1(1:nnz_row)) = 1;
		end
		M_true = Z_true*Z_true';
		L = 1-exp(-z_beta*M_true);
       	t1 = triu(rand(n),1);
       	t2 = triu(rand(n),0);
       	rand_mat = t1+t1'; 
       	Rb = double(rand_mat < L);	       	
		%nnz(Rb) It actually suggest self links
		positive_edge = sum(sum(Rb))
		parameter(1,12) = positive_edge*parameter(1,12); 
		negative_edge = n*n-sum(sum(Rb))
		sample_positive_edge = positive_edge*parameter(1,3)
		sample_negative_edge = positive_edge*parameter(1,4)
		positive_ratio = sample_positive_edge/positive_edge;  
		negative_ratio = min(sample_negative_edge/negative_edge,1);
		Sp = double(rand(n,n) > 1-positive_ratio);
		Sn = double(rand(n,n) > 1-negative_ratio);
		Strain = double(rand(n,n) > 1-parameter(1,15));
		Stest = 1-Strain;
		%deal with sample rate
		if(parameter(1,15) == 1)
			fprintf(2,'Sampling on positive/negative edge\n');
			R1 = sparse(Rb.*Sp);
			R0 = sparse((1-Rb).*Sn);
			Stest = Strain;
		else 
			fprintf(2,'Dividing into training and testing test\n');
			R1 = sparse(Rb.*Strain);
			R0 = sparse((1-Rb).*Strain);
		end		
		R2 = full(R1);
		snap_data ='none';
		if (parameter(1,16)>0.5)
			fprintf(2,'write the data to Syn_data for snap usage');
			C = strsplit(set_file_path,'/');
			if(7 ~= exist('Syn_data','dir'))
				mkdir('Syn_data');
			end
			snap_data = strcat('Syn_data/',C{2},'_snap.txt');
			fid5 = fopen(snap_data,'w+');
			assert(fid5>0,'Open file error\n');
			for i=1:n
				for j=1:n
					if(R2(i,j) > 0.9)
						fprintf(fid5,'%d\t%d\n',i,j);
					end
				end
			end
			fclose(fid5);
		end	
		if(7 ~= exist('Result','dir'))
			mkdir('Result');
		end
		cd Result;
		if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
		end
		cd(data_type);
		if(7 ~= exist(model_type,'dir'))
			mkdir(model_type);
		end
		cd ..;
		cd ..;
		cur_time = datestr(clock);
		outname = strcat('Result/',data_type,'/',model_type,'/',cur_time);
		out = fopen(outname,'w+');
		assert(out > 0,'Open out file error\n');
		outresultname = strcat('Result/',data_type,'/',model_type,'/Result');
		outresult = fopen(outresultname,'a');
		assert(outresult > 0,'Append result file error\n');
		snapload = 'none';
		if(parameter(1,16) > 0.5)
			fprintf(2,'Running bigclam first\n');
			snapload = run_snap(snap_data,snap_path,snap_out,snap_community,n);
		end
		fprintf(2,'Start running algorithm LFlasso_ZZ^t\n');
		[nmi_b,em_b,nmi_s,em_s,Z_guess,snap_tcom,obj_suc] = LFlasso_ZZt(Rb,R1,Stest,snap_cell,syn_parameter,parameter,Z_true,z_beta,data_cell,out,snapload,outname,outresult);
		fclose(out);
		fclose(outresult);
	end
	node = 0;
	edge_num = 0;
	cluster_num = 1;
	Rb = [];
	Z_true = [];
	if(strcmp(data_format,'nn'))
		data_tmp = load(data_path);
		node = data_tmp(1,1)
		edge_num =data_tmp(1,2)
		v1 = ones(edge_num,1);
		Rb = sparse(data_tmp(2:end,1),data_tmp(2:end,2),v1,node,node);	
	elseif (strcmp(data_format,'nnv'))
		data_tmp = load(data_path);
		node = data_tmp(1,1)
		edge_num = data_tmp(1,2)
		Rb = sparse(data_tmp(2:end,1),data_tmp(2:end,2),data_tmp(2:end,3),node,node);
	else 
		assert(1 < 0,'Non-readible data_format\n');
	end
	if(strcmp(cluster_format,'nc'))
		cluster_tmp = load(cluster_path);
	    cluster_num = cluster_tmp(1,1);
	    Z_true = zeros(node,cluster_num);
		for i=2:size(cluster_tmp,1)
			Z_true(cluster_tmp(i,1),cluster_tmp(i,2)) = 1;
		end
	elseif (strcmp(cluster_format,'none'))
		Z_true = double(rand(node,1)>0.8);
	else 
		assert(1 < 0,'Non-readible cluster_format\n');
	end
	positive_edge = sum(sum(Rb))
	parameter(1,12) = positive_edge*parameter(1,12); 
	negative_edge = node*node-sum(sum(Rb))
	sample_positive_edge = positive_edge*parameter(1,3)
	sample_negative_edge = positive_edge*parameter(1,4)
	positive_ratio = sample_positive_edge/positive_edge;  
	negative_ratio = min(sample_negative_edge/negative_edge,1);
	Sp = double(rand(node,node) > 1-positive_ratio);
	Sn = double(rand(node,node) > 1-negative_ratio);
	Strain = double(rand(node,node) > 1-parameter(1,15));
	Stest = 1-Strain;
	%deal with sample rate
	if(parameter(1,15) == 1)
		fprintf(2,'Sampling on positive/negative edge\n');
		R1 = sparse(Rb.*Sp);
		R0 = sparse((1-Rb).*Sn);
		Stest = Strain;
	else 
		fprintf(2,'Dividing into training and testing test\n');
		R1 = sparse(Rb.*Strain);
		R0 = sparse((1-Rb).*Strain);
	end		
    Rb1 = full(R1);
	if (parameter(1,16)>0.5)
		C = strsplit(data_path,'.');
		snap_data = strcat(C{2},'_snap.txt');
		snap_data = snap_data(2:end);
		if exist(snap_data, 'file') ~= 2
			fprintf(2,'write the data to %s for snap usage\n',snap_data);
			fid5 = fopen(snap_data,'w+');
			assert(fid5>0,'Open file error\n');
			for i=1:node
				for j=1:node
					if(Rb1(i,j) > 0.9)
						fprintf(fid5,'%d\t%d\n',i,j);
					end
				end
			end
			fclose(fid5);
		end	
	end
	if(7 ~= exist('Result','dir'))
			mkdir('Result');
	end
	cd Result;
	if(7 ~= exist(data_type,'dir'))
			mkdir(data_type);
	end
	cd(data_type);
	if(7 ~= exist(model_type,'dir'))
			mkdir(model_type);
	end
	cd ..;
	cd ..;
	cur_time = datestr(clock);
	outname = strcat('Result/',data_type,'/',model_type,'/',cur_time);
	out = fopen(outname,'w+');
	assert(out > 0,'Open out file error\n');
	outresultname = strcat('Result/',data_type,'/',model_type,'/Result');
	outresult = fopen(outresultname,'a');
	assert(outresult > 0,'Append result file error\n');
	snapload = 'none';
	if(parameter(1,16) > 0.5)
		fprintf(2,'Running bigclam first\n');
		snapload = run_snap(snap_data,snap_path,snap_out,snap_community,node);
	end
	if(strcmp(model_type,'ZW'))
		W_true = normrnd(0,1,1,node);
		fprintf(2,'Start running algorithm LFlasso_ZW\n');
		[nmi_b,em_b,nmi_s,em_s,Z_guess,W_guess,snap_tcom,obj_suc] = LFlasso_ZW(Rb,Stest,Rb1,R0,R1,snap_cell,syn_parameter,parameter,Z_true,W_true,data_cell,out,snapload,outname,outresult);
	else
		fprintf(2,'Start running algorithm LFlasso_ZZt\n');
	   [nmi_b,em_b,nmi_s,em_s,Z_guess,snap_tcom,obj_suc] = LFlasso_ZZt(Rb,R1,Stest,snap_cell,syn_parameter,parameter,Z_true,z_beta,data_cell,out,snapload,outname,outresult);
	end			
	fclose(out);
	fclose(outresult);
end
function []= unittest(syn,model_type,z_beta,snap_cell,data_cell,syn_parameter,parameter)
	fid=2;
	fprintf(fid,'data_type = %s\n',data_cell{5});
	fprintf(fid,'model_type = %s\n',model_type);
	if(strcmp(model_type,'ZZ^t'))
		fprintf(fid,'beta = %f\n',z_beta);
	end
	fprintf(fid,'snap_path = %s\n',snap_cell{1});
	fprintf(fid,'snap_out = %s\n',snap_cell{2});
	fprintf(fid,'snap_community = %f\n',snap_cell{3});
	if(syn < 1.5 & syn > 0.5)
		fprintf(fid,'mean = %f\n',syn_parameter(1,1));
		fprintf(fid,'var = %f\n',syn_parameter(1,2));
		fprintf(fid,'tunningI = %f\n',syn_parameter(1,3));
		fprintf(fid,'Z_dim_1 = %f\n',syn_parameter(1,4));
		fprintf(fid,'Z_dim_2 = %f\n',syn_parameter(1,5));
		fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,6));
	end
	if(syn > 1.5)
		fprintf(fid,'Z_dim_1 = %f\n',syn_parameter(1,1));
		fprintf(fid,'Z_dim_2 = %f\n',syn_parameter(1,2));
		fprintf(fid,'Z_sparsity = %f\n',syn_parameter(1,3));
	end
	if(syn < 0.5)
		fprintf(fid,'data_path = %s\n',data_cell{1});
		fprintf(fid,'data_format = %s\n',data_cell{2});
		fprintf(fid,'cluster_path = %s\n',data_cell{3});
		fprintf(fid,'cluster_format = %s\n',data_cell{4});
	end	
	fprintf(fid,'ground_truth_existence = %f\n',parameter(1,1));
	fprintf(fid,'noisy = %f\n',parameter(1,2));
	fprintf(fid,'positive_sample_rate = %f\n',parameter(1,3));
	fprintf(fid,'negative_sample_rate = %f\n',parameter(1,4));
	fprintf(fid,'iteration_num = %f\n',parameter(1,5));
	if(strcmp(model_type,'ZZ^t'))
		fprintf(fid,'linear_regularization_lambda = %f\n',parameter(1,6));
	else
		fprintf(fid,'w_solver_lambda = %f\n',parameter(1,6));
	end	
	fprintf(fid,'c_TOL= %f\n',parameter(1,7));
	fprintf(fid,'match_tol_rate = %f\n',parameter(1,8));
	fprintf(fid,'inner_iteration_num = %f\n',parameter(1,9));
	fprintf(fid,'SDP_rank = %f\n',parameter(1,10));
	fprintf(fid,'SDP_iter = %f\n',parameter(1,11));
	fprintf(fid,'mu_multiple_edge = %f\n',parameter(1,12));
	fprintf(fid,'stepsize = %f\n',parameter(1,13));
	fprintf(fid,'mod = %f\n',parameter(1,14));	
	fprintf(fid,'train = %f\n',parameter(1,15));
	fprintf(fid,'snap_run = %f\n',parameter(1,16));	
	fprintf(fid,'threshold_prob_for_link_start = %f\n',parameter(1,17));	
	fprintf(fid,'threshold_prob_for_link_space = %f\n',parameter(1,18));
	fprintf(fid,'threshold_prob_for_link_end = %f\n',parameter(1,19));	
end
function [snapload] = run_snap(snap_data,snap_path,snap_out,snap_community,node)
	cur_work_path = pwd;
	snap_data = strcat(cur_work_path,'/',snap_data);
	fprintf(2,'Move to snap directory %s\n',snap_path);
	cd(snap_path);
	if (snap_community > 0.5 )
		operation = strcat('./bigclam -o:',snap_out,' -i:',snap_data,' -c:',num2str(snap_community));
		fprintf(2,'Running operation %s\n',operation);
		system(operation);
	else
		operation = strcat('./bigclam -o:',snap_out,' -i:',snap_data);
		fprintf(2,'Running operation %s\n',operation);
		system(operation);
	end
	fprintf('Convert snap result\n');
	% convert snapresult to directly readible data by python
	snap_out = strcat(snap_out,'cmtyvv.txt');
	operation = strcat('python cmt_to_Z.py',{' '},snap_out,{' '},num2str(node));
	operation = operation{1};
	fprintf(2,'Running operation %s\n',operation);
	system(operation);
	snapload = strcat(snap_path,'/',snap_out,'.Z');
	cd(cur_work_path);
end
