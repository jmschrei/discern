# synthetic_analyses.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

'''
These tests will show the difference between DISCERN, ANOVA, and LNS
on pairs of synthetic Bayesian networks. You can build your own Bayesian
networks by hand (three examples shown below) and then use the barchart
and score_network_pair functions to handle the scoring of these
networks using DISCERN, ANOVA, and LNS in a standardized manner.
'''

import matplotlib
matplotlib.use('pdf')

import numpy
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from yabn import *
from discern import *
from LNS import *
from scipy.stats import f_oneway

random.seed(0)
numpy.random.seed(0)

def barchart( scores, method_names, node_names, title, normalize=True ):
	'''
	Take in the scores from two different feature selectors and plot them.
	'''

	sns.set( style='white', context='talk' )
	plt.figure( figsize=(12, 6) )
	n = len( scores )
	
	items = zip( xrange(n), scores, method_names, sns.color_palette('husl', 3) )
	for i, score, name, color in items:
		if normalize:
			score /= score.sum()

		x = np.arange( 0.5, score.shape[0]+0.5 )
		plt.bar( x+i*(0.8/n), score, width=0.8/n, alpha=0.5, edgecolor='w', label=name, facecolor=color )

	plt.legend()
	plt.title( title )
	plt.xticks( x+(1.0/n), node_names )
	plt.savefig( title + '.pdf' )

def score_network_pair( networka, networkb, node_names, i=100, j=100 ):
	'''
	This will take in a network and produce DISCERN and ANOVA scores for
	each node in the network. The user may set the number of samples
	generated for each network through adjusting i and j. Pass in the
	order of the node names to get the scores in the proper order.
	'''

	node_names_a = [ node.name for node in networka.nodes ]
	node_names_b = [ node.name for node in networkb.nodes ]
	
	# Get the data from sampling the two networks
	a_data = numpy.array([ networka.sample() for n in xrange( i ) ])
	b_data = numpy.array([ networkb.sample() for n in xrange( j ) ])

	# Convert this data into a dataframe for DISCERN
	a_data = pd.DataFrame( a_data, columns=node_names_a )
	b_data = pd.DataFrame( b_data, columns=node_names_b )

	# Initialize DISCERN and use it on the data
	discern = DISCERN()
	#l, sse = discern.lambda_opt( a_data[::2], node_names_a, n_cores=6 )
	discern.fit_score( a_data[::2], a_data[1::2], b_data[::2], b_data[1::2], 
		node_names_a, l=0.4, n_cores=8 )

	# Get the LNS scores
	lns = LNS()
	lns.fit_score( a_data, b_data, node_names_a )

	# Unpack the two score vectors into a numpy array
	discern_scores = numpy.array(discern._scores.ix[ node_names ]['T2'])
	anova_scores = numpy.array([ f_oneway( a_data[name], b_data[name] )[0] for name in node_names ])
	lns_scores = numpy.array( lns._scores.ix[ node_names ]['r'] )

	return discern_scores, anova_scores, lns_scores


def seven_star_tests():
	'''
	These tests work on a star network, where one node influences a second node,
	which then influences three nodes, and there are two independent nods, which
	switch identities in the graph. Basically, an influencer no longer influences
	and an independent node takes its place.
	'''

	# Define the two networks we will use
	networka = Network( "A" )
	networkb = Network( "B" )

	# Define all seven nodes, which are the same between the two networks
	n1 = Node( NormalDistribution( 12, 0.7 ), name="n1" )
	n2 = Node( NormalDistribution( 5, 0.3 ), name="n2" )
	n3 = Node( NormalDistribution( 17, 0.9 ), name="n3" )
	n4 = Node( NormalDistribution( 22, 1.2 ), name="n4" )
	n5 = Node( NormalDistribution( 12, 0.3 ), name="n5" )
	n6 = Node( NormalDistribution( 27, 3.2 ), name="n6" )
	n7 = Node( NormalDistribution( 88, 1.2 ), name="n7" )

	# We'll use a single edge of unit variance for this simple test
	e = 1.0

	# Add all the nodes to the networks
	networka.add_nodes( [n1, n2, n3, n4, n5, n6, n7] )
	networkb.add_nodes( [n1, n2, n3, n4, n5, n6, n7] )

	# Add all the edges to network A
	networka.add_edge( n1, n3, e )
	networka.add_edge( n3, n5, e )
	networka.add_edge( n3, n6, e )
	networka.add_edge( n3, n7, e )

	# Add all the edges to network B
	networkb.add_edge( n4, n3, e )
	networkb.add_edge( n3, n5, e )
	networkb.add_edge( n3, n6, e )
	networkb.add_edge( n3, n7, e )

	# Finalize the internals of the models
	networka.bake()
	networkb.bake()

	# Define the ordered names
	node_names = [ "n1", "n2", "n3", "n4", "n5", "n6", "n7" ]

	# Score the network
	discern, anova, lns = score_network_pair( networka, networkb, node_names )

	# Plot the scores
	barchart( [discern, anova, lns], ['DISCERN', 'ANOVA', 'LNS'], node_names, "n4-n3+ n1-n3-" )


	# Time for a second test, involving a network where only an edge between
	# n4 and n1 is added and nothing is removed.
	networkb = Network( 'b' )

	# Add the nodes in
	networkb.add_nodes( [n1, n2, n3, n4, n5, n6, n7] )

	# Add the edges in
	networkb.add_edge( n1, n3, e )
	networkb.add_edge( n3, n5, e )
	networkb.add_edge( n3, n6, e )
	networkb.add_edge( n3, n7, e )
	networkb.add_edge( n4, n1, e )

	# Finalize the model

	networkb.bake()

	# Score the nodes
	discern, anova, lns = score_network_pair( networka, networkb, node_names )

	# Plot the scores
	barchart( [discern, anova, lns], ['DISCERN', 'ANOVA', 'LNS'], node_names, "n4-n1+" )

def independent_no_perturbation_test( name="independent" ):
	''' 
	This will test a network which has no edges, and no perturbation, to see
	that the prediction power is not random.
	'''

	network = Network( 'independent' )

	# Create 12 distributions of random size 
	e = NormalDistribution( 50, 1.2 )
	n1 = Node( e, name="n1" )
	n2 = Node( e, name="n2" )
	n3 = Node( e, name="n3" )
	n4 = Node( e, name="n4" )
	n5 = Node( e, name="n5" )
	n6 = Node( e, name="n6" )
	n7 = Node( e, name="n7" )
	n8 = Node( e, name="n8" )
	n9 = Node( e, name="n9" )
	n10 = Node( e, name="n10" )
	n11 = Node( e, name="n11" )
	n12 = Node( e, name="n12" )

	# Add the nodes and finalize the structure of the data
	network.add_nodes( [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12] )
	network.bake()

	node_names = [ 'n{}'.format( i ) for i in xrange( 1, 13 ) ]

	# Get the scores
	discern, anova, lns = score_network_pair( network, network, node_names )

	# Plot it
	barchart( [discern, anova, lns], ['DISCERN', 'ANOVA', 'LNS'], node_names, name, normalize=False )

def three_component_test( name="three_component"):
	'''
	This will test a network which has thirteen nodes and several perturbations.
	'''

	networka = Network( 'a' )
	networkb = Network( 'b' )

	# Create some nodes
	emission = NormalDistribution( 10, 1 )
	n1 = Node( emission, name="n1" )
	n2 = Node( emission, name="n2" )
	n3 = Node( emission, name="n3" )
	n4 = Node( emission, name="n4" )
	n5 = Node( emission, name="n5" )
	n6 = Node( emission, name="n6" )
	n7 = Node( emission, name="n7" )
	n8 = Node( emission, name="n8" )
	n9 = Node( emission, name="n9" )
	n10 = Node( emission, name="n10" )
	n11 = Node( emission, name="n11" )
	n12 = Node( emission, name="n12" )
	n13 = Node( emission, name="n13" )


	# Unpack nodes
	node_names = [ 'n{}'.format( i ) for i in xrange( 1, 14 ) ]

	# Add the nodes to the module
	networka.add_nodes( [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13] )
	networkb.add_nodes( [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13] )

	# Define a uniform edge for simplicity
	e = 1.0

	# Add edges to the models
	networka.add_edge( n1, n2, e )
	networka.add_edge( n2, n3, e )
	networka.add_edge( n4, n2, e )
	networka.add_edge( n5, n6, e )
	networka.add_edge( n6, n7, e )
	networka.add_edge( n7, n9, e )
	networka.add_edge( n7, n10, e )
	networka.add_edge( n12, n11, e )
	networka.add_edge( n13, n12, e )

	networkb.add_edge( n1, n2, e )
	networkb.add_edge( n4, n2, e )
	networkb.add_edge( n5, n6, e )
	networkb.add_edge( n6, n7, e )
	networkb.add_edge( n7, n9, e )
	networkb.add_edge( n7, n10, e )
	networkb.add_edge( n12, n11, e )
	networkb.add_edge( n13, n12, e )
	networkb.add_edge( n4, n11, e )
	networkb.add_edge( n5, n8, e )
	networkb.add_edge( n8, n7, e )

	# Finalize the models
	networka.bake()
	networkb.bake()

	discern, anova, lns = score_network_pair( networka, networkb, node_names )

	barchart( [discern, anova, lns], ['DISCERN', 'ANOVA', 'LNS'], node_names, name )

def DCG( relevance ):
	'''
	Calculates the Discounted Cumulative Gain by comparing a 'true' ranking
	to a predicted ranking.
	'''
	
	n = len( relevance )
	return sum( (2.**relevance[i]-1.) / (i+1) for i in xrange( n ) )


def large_sparse_network( n=5000, m=50, low=1, high=10, name="large_sparse" ):
	'''
	Create a synthetic large, n nodes, where m of them get perturbed between
	the two graphs by changing between ~low~ and ~high~ edges. 
	'''

	# Randomly generate normal distributions for the node emissions
	# Means based on a gamma distribution, stds based on a lognormal
	# so that they are both bounded by 1
	
	means = [50]*n
	stds = [0.5]*n
	#means = numpy.random.gamma( 50, 3.0, n )
	#stds = numpy.random.lognormal( 0.5, 0.1, n )

	# Randomly choose M genes to perturb, and then for each perturbed gene
	# randomly choose the number of edges to perturb
	perturbed = numpy.random.choice( np.arange( n ), size=m, replace=False )
	n_perturbed_edges = numpy.random.randint( low, high, m )

	# Randomly generate the graph structure from beta distributions. All
	# weights are rounded to 1, instead of being variable.
	null_edges = numpy.tril( numpy.around( numpy.random.beta( 1, 3, (n,n) ) ) )
	numpy.fill_diagonal( null_edges, 0 )
	alternate_edges = null_edges.copy()

	perturb_count = { i:0 for i in xrange(n) }
	to_perturb_count = { i:0 for i in xrange(n) }

	# For each perturbed edge, randomly select between `low` and `high` number
	# of edges to perturb, and perturb them--in this case just a binary flip.
	for i, k in it.izip( perturbed, n_perturbed_edges ):
		perturbed_id = numpy.random.choice( numpy.arange( i ), size=min(k, i), replace=False )
		alternate_edges[i, perturbed_id] = numpy.abs( alternate_edges[i, perturbed_id] - 1 )

		perturb_count[i] += perturbed_id.shape[0]
		for index in perturbed_id:
			to_perturb_count[index] += 1 
 	total_perturb = { i: perturb_count[i]+to_perturb_count[i] for i in xrange(n) }

	if numpy.triu( alternate_edges ).sum() > 0:
		raise SyntaxError( "Matrix is not a DAG.")

	# Initiate the network objects
	null = Network( "Null" )
	alternate = Network( "Alternate" ) 
	
	# Create all the nodes 
	nodes = [ Node( NormalDistribution( mu, sigma ), name="n{}".format( i ) ) for i, mu, sigma in it.izip( xrange(n), means, stds ) ]
	node_names = [ node.name for node in nodes ]

	# Add them to the model
	null.add_nodes( nodes )
	alternate.add_nodes( nodes )

	# Create all the edges, one at a time
	for i in xrange( n ):
		for j in xrange( n ):
			p = null_edges[i, j]
			if p > 0:
				null.add_edge( nodes[i], nodes[j], p )

			p = alternate_edges[i, j]
			if p > 0:
				alternate.add_edge( nodes[i], nodes[j], p )

	# Finalize the internal structure of the network
	null.bake()
	alternate.bake()

	# Score the network pair according to the metrics
	discern, anova, lns = score_network_pair( null, alternate, node_names, i=100, j=300 )

	# Make a plot of the scores acorss the nodes
	#barchart( [discern, anova, lns], ['DISCERN', 'ANOVA', 'LNS'], node_names, name )

	scores = pd.DataFrame({ 'DISCERN': discern, 'ANOVA': anova, 
		'LNS': lns, 'FROM': perturb_count.values(), 'TO': to_perturb_count.values(),
		'TOTAL': total_perturb.values() })

	# Calculate the Discounted Cumulative Gain matrix. DCG is a way of measuring
	# a ranking of items if you know their true ordering. In this case, genes
	# should be ordered by the true number of perturbations to them, and we
	# compare the ordering we get from DISCERN, ANOVA, and LNS to that. DCG is
	# implemented in the DCG function above. In this case we divide nodes into
	# FROM nodes, which are the ranking of nodes according to perturbation in
	# number of edges LEAVING that nodes, TO nodes, which is perturbation in number
	# of edges going TO that node, and TOTAL which includes both. DISCERN is
	# predicted to identify FROM nodes better than other techniques, as those
	# should be similiar to driver mutations.
	DCG_Matrix = pd.DataFrame( { 'FROM': [ DCG( scores.sort( 'DISCERN', ascending=False )['FROM'].values ),
							               DCG( scores.sort( 'ANOVA', ascending=False )['FROM'].values ),
							               DCG( scores.sort( 'LNS', ascending=False )['FROM'].values ) ],
					'TO': [ DCG( scores.sort( 'DISCERN', ascending=False )['TO'].values ), 
							DCG( scores.sort( 'ANOVA', ascending=False )['TO'].values ),
							DCG( scores.sort( 'LNS', ascending=False )['TO'].values ) ],
					'TOTAL': [ DCG( scores.sort( 'DISCERN', ascending=False )['TOTAL'].values ),
							   DCG( scores.sort( 'ANOVA', ascending=False )['TOTAL'].values ),
							   DCG( scores.sort( 'LNS', ascending=False )['TOTAL'].values ) ] } )
	DCG_Matrix.index = [ 'DISCERN', 'ANOVA', 'LNS' ]

	print DCG_Matrix

	return scores, DCG_Matrix

if __name__ == '__main__':
	# Run the three smaller tests. Graphs will be output automatically.
	independent_no_perturbation_test()
	three_component_test()
	seven_star_tests()

	# Run the large sparse network. This example has 1000 nodes, of which
	# 25 are perturbed. You can play with these parameters as much as you
	# want, and the Discounted Cumulative Gain matrix will be returned.
	large_sparse_network( 1000, 25 )

