# discern.py
# Contact: Jacob Schreiber
#          jmschreiber91@gmail.com

'''
Run specific analyses for a cancer dataset using the general functions defined
in ogimos. Each function should be a specific cancer type. These will usually
run analyses in batches and output RData files.
'''

import pandas as pd
import numpy as np
import multiprocessing
import rpy2.robjects as ro
import matplotlib.pyplot as plt

from rpy2.robjects.numpy2ri import numpy2ri
ro.conversion.py2ri = numpy2ri

def scale( data ):
	'''
	Scale a matrix in a columnwise manner.
	'''

	return ( data - data.mean( axis=0 ) ) / data.std( axis=0 )

class DISCERN( object ):
	'''
	DISCERN is an unsupervised feature selection algorithm which uses
	differential correlation in order to identify perturbed features given
	data from two conditions. A Gaussian Graphical Model (GGM) is constructed
	for each condition, with nodes being features and edges being weights
	calculated by an Elastic Net Regressor. The DISCERN score is then calculated
	by looking at which edges differ significantly between the two networks, and
	identifying which genes are perturbed between the features, i.e. have
	different neighbors in the GGM.
	'''

	def __init__( self, l=0.3, alpha=0.95 ):
		pass

	def fit_score( self, null_training, null_testing, alternate_training,
		alternate_testing, names, mask=None, l=0, alpha=1.00, n_cores=None ):
		'''
		Build a GGM for the null dataset and the alternate dataset separately
		using an elastic net regressor. This is done through the R package
		glmnet. The DISCERN score will then be calculated and saved in
		self._scores.
		'''

		# Determine the number of cores to use. If -1 is passed in, use all the
		# cores, if nothing is passed in use 1 core, else use the specified
		# number of cores.
		if n_cores == -1:
			n_cores = multiprocessing.cpu_count()
		else:
			n_cores = n_cores or 1

		# Assign a uniform true mask on the covariates by default
		mask = mask or np.ones( null_training.shape[1] )

		# First we need to import the R libraries we want to work with
		ro.r( "library(glmnet)" )
		ro.r( "library(foreach)" )
		ro.r( "library(doParallel)" )

		# Set up the cluster using the given size
		ro.r( "cl <- makeCluster({})".format( n_cores ) )
		ro.r( "registerDoParallel(cl)" )

		# Scale all four data sets used independently of each other.
		null_training = scale( null_training )
		null_testing = scale( null_testing )
		alternate_training = scale( alternate_training )
		alternate_testing = scale( alternate_testing )

		# Now we need to push the data we're working with to the R environment
		ro.r.assign( "null_training", null_training )
		ro.r.assign( "alternate_training", alternate_training )
		ro.r.assign( "null_testing", null_testing )
		ro.r.assign( "alternate_testing", alternate_testing )
		ro.r.assign( "names", names )
		ro.r.assign( "mask", mask )	

		# Now push the glmnet usage and scoring function
		# y_n and y_a refer to a specific column (gene) from those two matrices
		# X_n and X_a refer to all the covariates, excluding the one chosen for y
		# fit_n and fit_a are fitten glmnet objects for the null or cancer set
		# y_pred_ab are the predicted values using fit_a on X_b
		# error_ab is the error calculated using fit_a on X_b

		ro.r( r"""
		discern <- function( start, finish, lambda, alpha ) {
			clusterExport( cl, c("null_testing", "null_training", "alternate_testing", 
				"alternate_training", "names", "mask" ))
			results <- foreach( i=start:finish, .packages='glmnet' ) %dopar% {
				covariates <- mask
				covariates[i] = F
				name = names[[i]]

				y_n <- null_training[, i]
				y_a <- alternate_training[, i]

				X_n <- null_training[, as.logical( covariates ) ]
				X_a <- alternate_training[, as.logical( covariates ) ]

				fit_n <- glmnet( X_n, y_n, standardize=FALSE, alpha=alpha, lambda=lambda )
				fit_a <- glmnet( X_a, y_a, standardize=FALSE, alpha=alpha, lambda=lambda )

				y_n <- null_testing[, i]
				y_a <- alternate_testing[, i]

				X_n <- null_testing[, as.logical( covariates ) ]
				X_a <- alternate_testing[, as.logical( covariates ) ]

				r <- tryCatch(
				{
					y_pred_nn <- predict( fit_n, X_n, s=lambda )
					y_pred_na <- predict( fit_n, X_a, s=lambda )
					y_pred_an <- predict( fit_a, X_n, s=lambda )
					y_pred_aa <- predict( fit_a, X_a, s=lambda )

					error_nn = sum( (y_pred_nn-y_n)^2 )
					error_na = sum( (y_pred_na-y_a)^2 )
					error_an = sum( (y_pred_an-y_n)^2 )
					error_aa = sum( (y_pred_aa-y_a)^2 )

					T4 = ( error_na + error_an ) / ( error_nn + error_aa )
					T2 = error_na + error_an - error_nn - error_aa
					c( T2, T4 )

				}, error = function(err) { 
					c( NaN, NaN )
				} ) 

				result = c( name, r[1], r[2] )
				return( result )
			}

			stopCluster(cl)
			gc()
			return( results ) 
		}""" )


		print null_training.shape[1] 
		scores = np.array( ro.r['discern']( 1, null_training.shape[1], l, alpha  ) )
		self._scores = pd.DataFrame( scores, columns=['Feature', 'T2', 'T4'] )
		self._scores = self._scores.convert_objects(convert_numeric=True)
		self._scores.index = self._scores['Feature']
		return self._scores

	def fit_score_transform( self, null, alternate, names, mask=None, threshold=3., 
		l=0., alpha=1.00, n_cores=None ):
		'''
		Return the original matrices having done feature selection, returning
		only features which scored above the threshold. 
		'''

		self.fit_score( null, alternate, names, mask, n_cores )
		features = self._scores[ self._scores['T4'] > threshold ]['Feature']
		return null[ features ], alternate[ features ]

	def lambda_opt( self, data, names, mask=None, n_cores=None,
		nfolds=5, alpha=1.00, plot=False ):
		'''
		Determine lambda_opt for a given set of data. Make sure that the data
		used here is the training data for the 
		'''

		# Determine the number of cores to use. If -1 is passed in, use all the
		# cores, if nothing is passed in use 1 core, else use the specified
		# number of cores.
		if n_cores == -1:
			n_cores = multiprocessing.cpu_count()
		else:
			n_cores = n_cores or 1

		# Assign a uniform true mask on the covariates by default
		mask = mask or np.ones( data.shape[1] )
		lambdas = 10**np.arange( 1, -3.5, -.05 )

		# First we need to import the R libraries we want to work with
		ro.r( "library(glmnet)" )
		ro.r( "library(foreach)" )
		ro.r( "library(doParallel)" )
		ro.r( "library(matrixStats)")

		# Set up the cluster using the given size
		ro.r( "cl <- makeCluster({})".format( n_cores ) )
		ro.r( "registerDoParallel(cl)" )

		# Pass these variables into R
		ro.r.assign( "data", data )
		ro.r.assign( "names", names )
		ro.r.assign( "mask", mask )	
		ro.r.assign( "alpha", alpha )
		ro.r.assign( "lambdas", lambdas )
		ro.r.assign( "nfolds", nfolds )

		lambda_cv = ro.r("""
		lambda_cv <- function() {
			clusterExport( cl, c("data", "names", "mask", "alpha", "lambdas", "nfolds") )

			#SSE <- foreach( i=1:dim(data)[2], .packages=c('glmnet', 'foreach', 'matrixStats'), .combine='+' ) %do% {
			SSE <- foreach( i=1:1, .packages=c('glmnet', 'foreach', 'matrixStats'), .combine='+' ) %do% {
				covariates <- mask
				covariates[i] = F
				name = names[[i]]

				errors <- foreach( j=1:nfolds, .combine='+' ) %do% {
					fold = seq( j, dim(data)[1], nfolds )

					y <- data[ -fold, i]
					X <- data[ -fold, as.logical( covariates ) ]

					x_mu_fit <- colMeans( X )
					x_sigma_fit <- colSds( X )

					y_mu_fit <- mean(y)
					y_sigma_fit <- sd(y)

					y <- ( y - y_mu_fit ) / y_sigma_fit
					X <- t( ( t(X) - x_mu_fit ) / x_sigma_fit )

					fit <- glmnet( X, y, standardize=FALSE, alpha=alpha, lambda=lambdas )

					y <- data[ fold, i ]
					X <- data[ fold, as.logical( covariates ) ]

					y <- ( y - y_mu_fit ) / y_sigma_fit 
					X <- t( ( t(X) - x_mu_fit ) / x_sigma_fit )

					y_pred = predict( fit, X )

					e <- foreach( k=1:dim(y_pred)[2] ) %do% {
						sum( ( y_pred[, k] - y )^2 )
					}

					e = unlist(e)
					return( e )
				}

				errors = unlist(errors)
				return( errors )
 			}

 			print( c(SSE, i) )
			stopCluster(cl)
			return( SSE )
		}
		""")

		l = np.array(ro.r['lambda_cv']())

		if plot:
			plt.plot( lambdas, l, c='c', alpha=0.5 )
			plt.xscale('log')
			plt.xlabel('$\lambda$')
			plt.ylabel('SSE')
			plt.title('Cross Validation Selection of $\lambda$')
			plt.savefig('lambda_opt.png')

		return lambdas[ np.argmin(l) ], l.min()

	def sparsity( self, data, names, mask=None, n_cores=None,
		nfolds=5, alpha=1.00 ):
		'''
		Create a graph of the sparsity of the matrix.
		'''

		# Determine the number of cores to use. If -1 is passed in, use all the
		# cores, if nothing is passed in use 1 core, else use the specified
		# number of cores.
		if n_cores == -1:
			n_cores = multiprocessing.cpu_count()
		else:
			n_cores = n_cores or 1

		# Assign a uniform true mask on the covariates by default
		mask = mask or np.ones( data.shape[1] )
		lambdas = 10**np.arange( 1, -3.5, -.05 )

		# First we need to import the R libraries we want to work with
		ro.r( "library(glmnet)" )
		ro.r( "library(foreach)" )
		ro.r( "library(doParallel)" )
		ro.r( "library(matrixStats)")

		# Set up the cluster using the given size
		ro.r( "cl <- makeCluster({})".format( n_cores ) )
		ro.r( "registerDoParallel(cl)" )

		# Pass these variables into R
		ro.r.assign( "data", data )
		ro.r.assign( "names", names )
		ro.r.assign( "mask", mask )	
		ro.r.assign( "alpha", alpha )
		ro.r.assign( "lambdas", lambdas )
		ro.r.assign( "nfolds", nfolds )

		sparsity_cv = ro.r("""
		sparsity_cv <- function() {
			clusterExport( cl, c("data", "names", "mask", "alpha", "lambdas", "nfolds") )

			#sparsity <- foreach( i=1:dim(data)[2], .packages=c('glmnet', 'foreach', 'matrixStats'), .combine='+' ) %dopar% {
			sparsity <- foreach( i=1:dim(data)[2], .packages=c('glmnet', 'foreach', 'matrixStats'), .combine='+' ) %dopar% {
				covariates <- mask
				covariates[i] = F
				name = names[[i]]

				edges <- foreach( j=1:nfolds, .combine='+' ) %do% {
					fold = seq( j, dim(data)[1], nfolds )

					y <- data[ -fold, i]
					X <- data[ -fold, as.logical( covariates ) ]

					x_mu_fit <- colMeans( X )
					x_sigma_fit <- colSds( X )

					y_mu_fit <- mean(y)
					y_sigma_fit <- sd(y)

					y <- ( y - y_mu_fit ) / y_sigma_fit
					X <- t( ( t(X) - x_mu_fit ) / x_sigma_fit )

					fit <- glmnet( X, y, standardize=FALSE, alpha=alpha, lambda=lambdas )

					n_edges = fit['df']$df / nfolds
					return(n_edges)
				}

				edges = unlist(edges)
				return( edges )
 			}

			stopCluster(cl)
			return( sparsity )
		}
		""")

		s = np.array(ro.r['sparsity_cv']())
		s /= data.shape[1] ** 2

		plt.plot( lambdas, s, c='c', alpha=0.5, linewidth=2.5 )
		plt.xscale('log')
		plt.xlabel('$\lambda$')
		plt.ylabel('Percent Edges')
		plt.savefig('sparsity.png')
