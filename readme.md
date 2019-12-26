The python is mainly used to clean the data set to run any simple ML algorithm.
The code makes use of the libraries present in scikit learn.
The following are the main goals of the script:
	1. To read a csv file.
	2. Store the data into 2 variables based on dependent and independent vars (Here, it is assumed that the data has independent vars in all but the last column, with the data itself having 4 columns).
	3. Replace all the nan datavalues by the mean of that column.
	4. Convert categorical data into onhotencoded data.
	5. Create train and test datasets with testset having 20% of the entire data.
	6. To perform feature scaling on the independent vars.

Note: Include print statements to check the flow of the code.