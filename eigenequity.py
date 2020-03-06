import numpy as np
import scipy.linalg
import sys

################################################################################
### PARAMETERS
################################################################################
# The alpha parameter determines how much equity should be allocated according
# to pure EigenEquity, and the remainder by 
# By default, we use 100% EigenEquity. You should only need to change this
# if you have an adversarial, highly fragmented organization with clusters of
# people who rate other people or teams at 0. In this case, you probably have
# bigger problems to worry about.
# Remark: An alpha value of 0.85 will make this equivalent to Google PageRank's 
# random surfer model for ranking web sites by their hyperlink structure.
alpha = 1.0


################################################################################
### MAIN
################################################################################
if len(sys.argv) > 0:
	votes_file = sys.argv[1]
else:
	print ("Usage: {} <CSV file>".format (sys.argv[0]))
	print ("The CSV file should contain a header row with N columns of individual names, followed by N rows with N numbers between 0-1.0, all fields separated by a tab. The rows are in the same order as the columns, so if Alice is the name in the first column, then Alice's votes are the first row after the header row. Each row represents the equity allocation desired by that person, and should sum to 1 (100%). For an example, see example.csv")
	sys.exit(0)

# Read data from CSV file
try:
	votes = np.genfromtxt (votes_file, delimiter='\t', names=True, skip_header=0, skip_footer=0)
except:
	print ("Unable to open file or parse votes matrix in file {}.".format (votes_file))
	print ("Try following the example in example.csv")
	sys.exit(-1)

names = list(sorted(votes.dtype.fields))
n = len(names)
M = votes.view((float, n))
origM = M

# Sanity check: ensure everybody used up their full votes
if np.linalg.norm (M.sum(axis=1) - np.ones(n)) > 1e-8:
	print ("At least one row in the file does not add up to 1 (100%).")
	sys.exit(-2)

# Get rid of people's self-votes
M = (M - np.diag(M)*np.eye(n))
M = M / M.sum(axis=1)

print ("Alpha value of EigenEquity:\t{}".format (alpha))

# Ensure we don't have components by using the PageRank trick
M = alpha * M + (1-alpha) * np.ones((n,n))
M = M / M.sum(axis=1)

# Find left eigenvector with eigenvalue 1
(eigval,eigvec) = np.linalg.eig(M.T) # NumPy's eig find the right eigenvector, thus the transpose
domidx = np.argmin(np.abs (eigval - 1.0))

# Ensure we have eigenvalue 1
if abs(eigval[domidx] - 1.0) > 1e-8:
	print ("Input matrix is not Markovian (largest eigenvalue is not 1) -- this should not happen!")
	sys.exit(-3)

# Normalize the stationary vector into a probability distribution
pdist = eigvec[:,domidx]
pdist /= np.sum(pdist)

# Print out the EigenEquity allocation
for who,what,how in zip(names, pdist, np.diag(origM)):
	print ("{} should receive:\t\t{}%\t\t(Wanted {}%)".format (who, np.around(what * 100.0, 2), np.around(how*100.0, 2)))



