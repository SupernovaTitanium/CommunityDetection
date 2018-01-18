Usage:

      1:make

      2:open matlab and execute lf_script

Requirement:

	  1:the data is simulated while running lf_script.m

	  2:Be careful to check all the paths in the makefile are correct(especially matlab root)

	  3:directly tune parameters by editing lf_script.m and LFlasso

	  4:You can test different parallel programming settings in w_solver.cpp

Info:

      1:Another MKL version is still on working

      2:LFlasso_over use dense A maxcut solver 
	
      3:If you find "libstdc++.so.6: version `GLIBCXX_3.4.20' not found" during make, try to add
		
	"export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6" to the shell.
 

