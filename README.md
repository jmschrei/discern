DISCERN
=======

This module implements DISCERN, a method of identifying features whose 
conditional dependancy structure differs between two networks. An example
is in the field of biology, where features may be genes, and the two networks
may be those with cancer and those without cancer. DISCERN would identify
those genes who are conditionally dependent on a different group of genes
in the two networks.

## Installation

DISCERN is available on PyPi, making installation as easy as running

```
pip install discern
```

Alternatively, you can clone this repo to get the cutting edge.

DISCERN makes use of rpy2, which can be a pain to install. If you do not have it
installed, you have to set environment variables R_HOME and R_USER, where R_HOME
is where your R installation is, and R_USER is where your python executable is
(C:\Python27 or C:\Anaconda) usually. Once those are set properly, rpy2 is also
pip installable with http://www.lfd.uci.edu/~gohlke/pythonlibs/

```
pip install rpy2
```

Many people have difficulty installing rpy2, but that means there is help online
if you're having issues installing it. If you are on a windows, check out the windows
binaries available at 

## Tutorial

Using DISCERN is simple. Here is a sample script, assuming that A
and B are two datasets with the same features, but allowing different
numbers of samples. DISCERN takes in pandas dataframes exclusively.

```
from discern import DISCERN

A, B = ** some pandas dataframes **
discern = DISCERN()

# Split the data into training and testing
A_train, A_test = A[::2], A[1::2]
B_train, B_test = B[::2], B[1::2]

scores = discern.fit_score( A_train, A_test, B_train, B_test, names=A.columns,
	l=0.3 )

print scores
```

This will return the scores. DISCERN makes use of Elastic Net regressions
in order to build the underlying network, and so parameters lambda (l) and
alpha (alpha) should be specified. alpha=1.0 means pure LASSO, and alpha=0.0
means pure ridge regression. 

If you want to automatically determine lambda, you can pass in l='auto', or
do it yourself with `discern.lambda_opt( data, names... )`, which will return
both the optimal lambda and SSE across all regression problems which the optimal
lambda minimizes.

DISCERN has built-in parallelization from R. For any function, simply pass in
`n_cores=n` where n is the number of cores you want to use. Since these methods
are natively scalable, the more cores the better. Not passing anything in assumes
only using 1 core. 