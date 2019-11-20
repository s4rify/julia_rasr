# import library
using PosDefManifold

## Example
 # generate a 15x15 symmetric positive definite matrix
 P=randP(15)
 # distance from P to the identity matrix according to the logdet0 metric
 d=distance(logdet0, P)

Q = randP(15)

d = distanceSqr(Fisher, P, Q)

P = randP(15,4)

G = mean(Fisher, P)

# unweighted mean
G, iter, conv = geometricMean(P;  ⍰=true) # or G, iter, conv = geometricMean(𝐏)

# show convergence information
evalues, maps, iter, conv=spectralEmbedding(Fisher, P, 2; ⍰=true)

using Plots
# check eigevalues and eigenvectors
plot(diag(evalues))
plot(maps[:, 1])
plot!(maps[:, 2])

plot(maps[:, 1], maps[:, 2], seriestype=:scatter, title="Spectral Embedding", label="Pset")

P = randP(15,200)

# eeg would be ordered samples in the columns and channels in the rows
