import matplotlib
matplotlib.use('pdf')

from discern import *
import time
import sys
from LNS import *
from scipy.stats import fisher_exact, f_oneway as anova

# I personally find these tests to be extremely poor indicators of LNS or DISCERN
# effectiveness. DISCERN and LNS try to identify perturbations in the conditional
# dependence structure, and identify 'driver' genes. These genes do not change
# their expression levels when a cell becomes cancerous. This means that they shouldn't
# be correlated with survival time significantly. It thus makes little sense to look
# at the overlap between univariate survival time p-values. I also find the idea of
# a univariate survival time model to be somewhat shaky. Using these analyses indicates
# that you are OK with these concerns.

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
	run_discern( null, cancer, gene_names, 0.1, "DISCERN_AML.csv" )
	run_anova( null, cancer, gene_names, "ANOVA_AML.csv" )
	run_lns( null, cancer, gene_names, "LNS_AML.csv" )

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
	run_discern( null, cancer, gene_names, 0.05, "DISCERN_BRCA.csv" )
	run_anova( null, cancer, gene_names, "ANOVA_BRCA.csv" )
	run_lns( null, cancer, gene_names, "LNS_BRCA.csv" )

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
	run_discern( null, cancer, gene_names, 0.05, "DISCERN_LUAD.csv" )
	run_anova( null, cancer, gene_names, "ANOVA_LUAD.csv" )
	run_lns( null, cancer, gene_names, "LNS_LUAD.csv" )

def run_anova( null, cancer, gene_names, outfile ):
	'''
	Take in a null preprocessed dataset and a cancer preprocessed dataset and
	runs ANOVA.
	'''

	n, d = null.shape
	anova_scores = np.zeros( d )

	for i in xrange( d ): 
		anova_scores[i] = anova( null.values[:,i], cancer.values[:,i] )[0] 

	data = pd.DataFrame( {'ANOVA': anova_scores } )
	data.index = gene_names
	data.to_csv( outfile )

def run_lns( null, cancer, gene_names, outfile ):
	'''
	Take in a null preprocessed dataset and a cancer preprocessed dataset and
	run local network similarity.
	'''

	n, d = null.shape
	lns = LNS()
	scores = lns.fit_score( null, cancer, gene_names )
	scores.to_csv( outfile )

def run_discern( null, cancer, gene_names, lambda_opt, outfile ):
	'''
	Take in a null preprocessed dataset and a cancer preprocessed dataset and
	a lambda_opt and run DISCERN using a 50-50 train-test split, saving the
	resulting DISCERN scores to an appropriate CSV file.
	'''

	null_training = null[::2]
	null_testing = null[1::2]
	cancer_training = cancer[::2]
	cancer_testing = cancer[1::2]

	discern = DISCERN()
	scores = discern.fit_score( null_training, null_testing, cancer_training, 
		cancer_testing, gene_names, n_cores=8, l=lambda_opt )
	scores.to_csv( outfile )

def survival_expression_comparison( discern=None, anova=None, lns=None, survival=None, name=None ):
	'''
	Compare the DISCERN scores for genes to the p-values obtained by running
	Cox Proportional Hazards using survival time. Look at the overlap between
	the identified genes using p-value and enrichment. Also compare LNS and
	ANOVA scores in the same way, to allow for a proper comparison.
	'''

	import seaborn as sns

	if name:
		survival = pd.read_table( "{}\\{}_all_genes_survival.txt".format( 
			name.upper(), name.lower() ), sep=' ', names=['gene', 'p-value'] )
		discern = pd.read_csv( "{}\\DISCERN_{}.csv".format( name.upper(), name.upper() ), index_col=0 )
		anova = pd.read_csv( "{}\\ANOVA_{}.csv".format( name.upper(), name.upper() ), index_col=0 )
		lns = pd.read_csv( "{}\\LNS_{}.csv".format( name.upper(), name.upper() ), index_col=0 )

	survival_top = set( survival[ survival['p-value'] < 0.05 ].gene ).intersection( set( discern.index ) )
	discern = discern.sort( 'T2' )
	anova = anova.sort( 'ANOVA' )
	lns = lns.sort( 'p' )
	n = len( discern.values )
	cn = len(survival_top)

	discern_p_vals, anova_p_vals, lns_p_vals = [], [], []
	discern_enrichment, anova_enrichment, lns_enrichment = [], [], []

	discern_overlap = 1 if discern.index[0] in survival_top else 0
	anova_overlap = 1 if anova.index[0] in survival_top else 0
	lns_overlap = 1 if lns.index[0] in survival_top else 0

	for i in xrange( 1, n ):
		discern_overlap += 1 if discern.index[i] in survival_top else 0
		anova_overlap += 1 if anova.index[i] in survival_top else 0
		lns_overlap += 1 if lns.index[i] in survival_top else 0
		
		table = [[discern_overlap, cn-discern_overlap], [i-discern_overlap, n-i-cn+discern_overlap]]
		discern_p_vals.append( -np.log10( fisher_exact( table )[1] ) )
		discern_enrichment.append( discern_overlap / (1.*cn*i/n) )

		table = [[anova_overlap, cn-anova_overlap], [i-anova_overlap, n-i-cn+anova_overlap]]
		anova_p_vals.append( -np.log10( fisher_exact( table )[1] ) )
		anova_enrichment.append( anova_overlap / (1.*cn*i/n) )

		table = [[lns_overlap, cn-lns_overlap], [i-lns_overlap, n-i-cn+lns_overlap]]
		lns_p_vals.append( -np.log10( fisher_exact( table )[1] ) )
		lns_enrichment.append( lns_overlap / (1.*cn*i/n) )

	plt.title( "Overlap P-Value Using Top N Genes" )
	plt.xlabel( "N" )
	plt.ylabel( "-log10( p-value )" )
	plt.plot( discern_p_vals, alpha=0.2, color='r', label='DISCERN' )
	plt.plot( anova_p_vals, alpha=0.2, color='g', label='ANOVA' )
	plt.plot( lns_p_vals, alpha=0.2, color='b', label='LNS' )
	plt.legend()
	plt.savefig( name+"_p_value_plot.pdf" )
	plt.clf()
	
	plt.title( "Overlap Enrichment Using Top N Genes" )
	plt.xlabel( "N" )
	plt.ylabel( "Enrichment" )
	plt.plot( discern_enrichment[:500], alpha=0.2, color='r', label='DISCERN' )
	plt.plot( anova_enrichment[:500], alpha=0.2, color='g', label='ANOVA' )
	plt.plot( lns_enrichment[:500], alpha=0.2, color='b', label='LNS' )
	plt.legend()
	plt.savefig( name+"_enrichment_plot.pdf" )
	plt.clf()

def plot_discern_distributions():
	'''
	Plot some useful visualizations of the DISCERN scores as a scatter matrix, where
	the diagonal is the kernel density of the scores, and the off-diagonals are
	scatter plots comparing two conditions.
	'''

	from pandas.tools.plotting import scatter_matrix
	import seaborn as sns

	AML = pd.read_csv( "AML\\DISCERN_AML.csv", index_col=0 )
	BRCA = pd.read_csv( "BRCA\\DISCERN_BRCA.csv", index_col=0 )
	LUAD = pd.read_csv( "LUAD\\DISCERN_LUAD.csv", index_col=0 )
	AML['Gene'], BRCA['Gene'], LUAD['Gene'] = AML.index, BRCA.index, LUAD.index
	AML['AML'], BRCA['BRCA'], LUAD['LUAD'] = np.log10(AML['T2']), np.log10(BRCA['T2']), np.log10(LUAD['T2'])
	AML, BRCA, LUAD = AML[['Gene', 'AML']], BRCA[['Gene', 'BRCA']], LUAD[['Gene', 'LUAD']]

	data = pd.merge( AML, BRCA, on='Gene' )
	data = pd.merge( data, LUAD, on='Gene' )
	
	with sns.axes_style( "whitegrid" ):
		scatter_matrix( data, alpha=0.2, figsize=(6,6), diagonal='kde', color='c', density_kwds={'c': 'r', 'lw':1}, lw=0, grid=False ) 

	plt.savefig( 'DISCERN_Scores.pdf' )
	plt.clf()

	print "AML"
	print data.sort( 'AML', ascending=False )[['Gene', 'AML']][:10]
	print
	print "BRCA"
	print data.sort( 'BRCA', ascending=False )[['Gene', 'BRCA']][:10]
	print
	print "LUAD"
	print data.sort( 'LUAD', ascending=False )[['Gene', 'LUAD']][:10]