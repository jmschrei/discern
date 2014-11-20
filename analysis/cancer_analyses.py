from discern import *
import time
import sys

def survival( expression, survival, outfile='CoxPH_p_vals.csv' ):
	'''
	Take in a filename for expression data, and a filename for survival data,
	and perform a univariate Cox Proportional Hazard model to identify which
	genes are associated with survivor time. 
	'''

	data = load_data( expression, gene_delimiter="|" ).T

	# This is what Maxim does with his data 
	#data[ data == 0 ] = 1e-5
	#data = preprocess_data( data, mean_threshold=9.64, log=True, winsorize=2.5, merge_duplicates=True )
	#data['ID'] = map( lambda s: s.split('-')[2], data.index )

	# Preprocess the data by adding a pseudocount, filtering out low expression
	# levels, merging duplicate genes, and log-transforming the data 
	data = preprocess_data( data, pseudocount=1, mean_threshold=10, 
		merge_duplicates=True, log=True )

	# Split the data into those with cancer and those without cancer 
	null = data.ix[[ sample for sample in data.index if sample.split('-')[3][0] == '1' ]]
	cancer = data.ix[[ sample for sample in data.index if sample.split('-')[3][0] != '1' ]]

	# Make a column for ID based on the barcode of the patient
	data['ID'] = map( lambda s: '-'.join( s.split('-')[:3] ), data.index )
	data = data.ix[[ sample for sample in data.index if sample.split('-')[3][0] != '1' ]]
	data = data.drop( ['?'], axis=1 )

	# Load the clinical data
	clinical = pd.read_table( "LUAD\\LUAD.clin.merged.txt", index_col=0 ).T

	# Make an ID column based on the patient barcode 
	clinical['ID'] = map( str.upper, clinical['patient.bcrpatientbarcode'] )
	clinical['patient.survival_time'] = clinical[['patient.daystodeath', 'patient.daystolastfollowup']].max( axis=1 )
	clinical = clinical[ clinical['patient.survival_time'] > 0 ]
	clinical = clinical[['ID', 'patient.survival_time', 'patient.vitalstatus']]

	# Do an outer join on the clinical data and expression data, using the ID
	# column as the pivot 
	data = data.merge( clinical, how='outer' )

	# Remove any rows with null values, and remove the ID column
	data = data.dropna( how='any' ).drop('ID', axis=1)

	# Cast survival time and vital status as integers
	data['patient.survival_time'] = map( int, data['patient.survival_time'] )
	data['patient.vitalstatus'] = map( int, data['patient.vitalstatus'] == 'dead' )

	# Pull out the expression matrix, survival time, and censoring information
	survival_time = np.array( data['patient.survival_time'] )
	alive = np.array( data['patient.vitalstatus'] )
	gene_exp = data[[ gene for gene in data.columns if 'patient' not in gene ]]

	# Pass the info into the coxph regression function, which returns a
	# pandas dataframe, and save that to a csv. 
	info = coxph_regression( gene_exp, survival_time, alive, n_cores=-1 )
	info.to_csv( outfile )

def aml_analysis():
	'''
	Load up the AML data and pass it to the DISCERN methods.
	'''

	# Load up the data
	null = pd.read_csv( 'AML\\AML1_normal.csv', index_col=0 ).T
	cancer = pd.read_csv( 'AML\\AML1_cancer.csv', index_col=0 ).T

	# Pull the gene names from the columns 
	gene_names = null.columns

	# Run the DISCERN analysis
	run_discern( null, cancer, gene_names, 0.05, "DISCERN_LUAD.csv" )

def brca_analysis():
	'''
	Load up the BRCA data and pass it to the DISCERN methods.
	'''

	# Load up the data
	null = pd.read_csv( 'BRCA\BRCA_data_normal.csv', index_col=0 ).T
	cancer = pd.read_csv( 'BRCA\BRCA_data_cancer_RESTRICTED.csv', index_col=0 ).T

	# Pull the gene names from the columns 
	gene_names = null.columns

	# Run the DISCERN analysis
	run_discern( null, cancer, gene_names, 0.1, "DISCERN_BRCA.csv" )

def luad_analysis():
	'''
	Load up the LUAD data and pass it to the DISCERN methods.
	'''

	# Load up the data
	null = pd.read_csv( 'LUAD\luad_data_normal.csv', index_col=0 ).T
	cancer = pd.read_csv( 'LUAD\luad_data_cancer.csv', index_col=0 ).T

	# Pull the gene names from the columns 
	gene_names = null.columns

	# Run the DISCERN analysis
	run_discern( null, cancer, gene_names, 0.1, "DISCERN_AML.csv" )

def run_discern( null, cancer, gene_names, lambda_opt, outfile ):
	'''
	Take in a null preprocessed dataset and a cancer preprocessed dataset and
	a lambda_opt and run DISCERN, saving to an appropriate CSV file.
	'''

	null_training = null[::2]
	null_testing = null[1::2]
	cancer_training = cancer[::2]
	cancer_testing = cancer[1::2]

	discern = DISCERN()
	scores = discern.fit_score( null_training, null_testing, cancer_training, 
		cancer_testing, gene_names, n_cores=6, l=lambda_opt )
	scores.to_csv( outfile )

tic = time.time()
aml_analysis()
brca_analysis()
luad_analysis()
print "Took {} seconds".format( time.time() - tic)
