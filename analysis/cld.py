# discern
# Contact: Jacob Schreiber (jmschr@cs.washington.edu)

from cancer_analyses import *
import pandas as pd
import argparse
import sys


def parse_command_line():
	'''
	Parse the command line and return the parser.
	'''

	parser = argparse.ArgumentParser( description="Read in Data" )
	parser.add_argument( '-n', action='store', type=file, 
		help='A CSV file of gene expression data for healthy patients.' )
	parser.add_argument( '-c', action='store', type=file, 
		help='A CSV file of gene expression data for cancerous patients.' )
	parser.add_argument( '-l', type=float, action='store', default=0.5,
		help='A value of lambda to run DISCERN at.')
	args = parser.parse_args()

	return args

def run_analyses():
	'''
	Run the analyses indicated by the number of arguments passed in.
	'''

	parser = parse_command_line()
	cancer = pd.read_csv( parser.c, index_col=0 ).T
	normal = pd.read_csv( parser.n, index_col=0 ).T

	discern_scores = run_discern( normal, cancer, cancer.columns, parser.l, sys.stdout )

def main():
	print "a"
	run_analyses()

if __name__ == '__main__':
	main()