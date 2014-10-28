# LNS.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

'''
This is an implementation of Local Network Similarity (LNS), a metric
introduced by Yuanfang Guan et al in their 2013 paper "Comparative
gene expression between two yeast species". 
'''

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def scale( data ):
	'''
	Scale a matrix in a columnwise manner.
	'''

	return ( data - data.mean( axis=0 ) ) / data.std( axis=0 )

class LNS( object ):
	'''
	This is a Local Similarity Network score calculator. It calculates a
	standardized correlation coefficient for each pairwise interaction in
	order to build a pairwise-interaction network. Data must be input as a
	pandas dataframe where each column is a feature. 
	'''

	def __init__( self ):
		pass

	def fit_score( self, null, alternate, node_names ):
		'''
		Take in a matrix of values, and compute the standardized correlation
		between each of them in order to produce a standardized  
		'''

		# Convert from columns being featurs to rows being features
		null = null.values
		alternate = alternate.values

		# Get the number of nodes, and initialize the edge matrix as zeros.
		n, d = null.shape
		null_edges = np.zeros((n,n))
		alternate_edges = np.zeros((n,n))

		for i in xrange( d ):
			for j in xrange( i+1 ):
				null_edges[i, j] = pearsonr( null[:,i], null[:,j] )[0]
				null_edges[j, i] = null_edges[i, j]

				alternate_edges[i, j] = pearsonr( alternate[:,i], alternate[:,j] )[0]
				alternate_edges[j, i] = alternate_edges[i, j]

		# Perform Fisher's Z Transform, which is just the arctan
		null_edges = np.arctan( null_edges )
		alternate_edges = np.arctan( alternate_edges )

		# Normalize the edges so that they follow the normal distribution
		null_edges = scale( null_edges.flatten() ).reshape( (n, n) )
		alternate_edges = scale( alternate_edges.flatten() ).reshape( (n, n) )

		# Calculate the score for each node
		scores = np.zeros((d,2))

		for i in xrange( d ):
			r, p = pearsonr( null_edges[:,i], alternate_edges[:,i] )
			scores[i,0], scores[i,1] = r, p

		self._scores = pd.DataFrame( scores, columns=['r', 'p'] )
		self._scores.index = node_names
		return self._scores 
