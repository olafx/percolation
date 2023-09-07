# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from scipy.optimize import curve_fit
import os

# %%
# Options.

# Bond probability.
p = 0.5
# Lattice size.
N = 256
# Number of bins for histograms.
N_bins = 16
# Font size.
fs = 18

rng = np.random.default_rng()

# %%
# Generate the lattice of bonds, label the clusters, and calculate the
# cluster sizes.

def gen_bonds(p) -> np.ndarray, int, np.ndarray:
  # Returns a lattice with labeled clusters, the number of clusters, and the
  # size of each cluster.
  lb = rng.uniform(size=(N, N)) < p
  # The closed connection background is label 0, and is not included in the
  # number of labels `nl`.
  lbl, nl = ndimage.label(lb)
  cs = np.zeros(nl+1)
  for i in range(nl+1): cs[i] = (lbl == i).sum()
  return lbl, nl, cs

lbl, nl, cs = gen_bonds(p)

print(f'{nl} unique clusters of bonds')

# %%
# Histogram of the cluster sizes. This can be used to test the subcritical
# behavior.

os.makedirs('out', exist_ok=True)

plt.rcParams['font.family'] = 'CMU'
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (5, 3)

FIT_FISHER = True # Fitting critical power law behavior.

if FIT_FISHER:
  S_C, C = np.histogram(cs[1:], density=True, bins=N_bins) # Counts, bin edges.
  bw = np.diff(C)[0] # Widths.
  C = (C[:-1]+C[1:])/2 # Bin edge to bin center.
  S_C_fit = lambda C_, alpha, tau: alpha*C_**-tau
  i_start = 8 # Start for fit, because only supposed to work in large size limit.
  alpha, tau = curve_fit(S_C_fit, C[i_start:], S_C[i_start:])[0]
  print(f'{tau:.2e}')

plt.figure(1)
plt.yscale('log')
if FIT_FISHER:
  plt.bar(C, S_C, width=bw, color='black')
  plt.plot(C[i_start:], S_C_fit(C[i_start:], alpha, tau), color='red')
else: plt.hist(cs[1:], density=True, bins=N_bins, color='black')
plt.title(f'${N}\\times{N}$, $p={p}$, ${nl}$ clusters', size=fs)
plt.xlabel('$|C|$', size=fs)
plt.ylabel('$S_{{|C|}}(p)$', size=fs)
plt.tight_layout()
plt.savefig('out/perc_1.pdf', bbox_inches='tight')

# %%
# Assign colors to the clusters of bonds and to the closed background.

# Each label gets a random color, except the background is black. Create a
# `mcolors.Colormap` object that maps the labels to their color.

def gen_cmap(nl):
  colors = plt.get_cmap('hsv')(np.linspace(0, 1, nl))
  rng.shuffle(colors)
  colors = np.vstack((mcolors.to_rgba('black'), colors))
  return mcolors.ListedColormap(colors)

# %%
# Plot the bonds.

os.makedirs('out', exist_ok=True)

plt.rcParams['font.family'] = 'CMU'
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (5, 3)

plt.figure(2)
plt.xticks([])
plt.yticks([])
plt.imshow(lbl, cmap=gen_cmap(nl), interpolation='none') # The interpolation algorithm gets confused here.
plt.title(f'${N}\\times{N}$, $p={p}$', size=fs)
plt.tight_layout()
plt.savefig('out/perc_2.pdf', bbox_inches='tight')

# %%
# Generating many lattices at different parameters `p` to measure the
# probability of percolation, and fit the associated power law.

os.makedirs('out', exist_ok=True)

plt.rcParams['font.family'] = 'CMU'
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (5, 3)

N_p = 10 # Number of `p` values.
n = 10 # Number of samples per `p` value.
thetas = np.zeros(N_p) # Percolation probabilities.
ps = np.linspace(.5, .55, N_p) # Parameter domain.

for i, p in enumerate(ps):
  j = 0
  while j < n:
    lbl, nl, cs = gen_bonds(p)
    # We test if the bond at the origin is also found on the edge.
    l = lbl[N//2,N//2]
    if l == 0: continue # Not a bond.
    # I considered percolation here as being connected to any edge. Got a
    # comment about this on the exam, this is supposed to be all technically,
    # but I'm leaving it in because it's just definitions and doesn't
    # qualitatively change much.
    if (np.vstack([lbl[0], lbl[-1], lbl[:,0], lbl[:,-1]]) == l).any():
      thetas[i] += 1/n
    j += 1

# Now we fit this data.
FIT_THETA = True
if FIT_THETA:
  theta_f = lambda p, alpha, beta: alpha*(p-.5)**beta
  alpha, beta = curve_fit(theta_f, ps, thetas)[0]
  print(f'{beta:2e}')

plt.figure(3)
plt.grid()
plt.xlim(ps[0], ps[-1])
plt.plot(ps, thetas, c='black')
if FIT_THETA: plt.plot(ps, theta_f(ps, alpha, beta), c='red')
plt.title(f'${N}\\times{N}$, $n={n}$', size=fs)
plt.xlabel('$p$', size=fs)
plt.ylabel('$\\theta(p)$', size=fs)
plt.tight_layout()
plt.savefig('out/perc_3.pdf', bbox_inches='tight')
