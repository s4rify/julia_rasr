function fit_eeg_distribution(X, min_clean_fraction, max_dropout_fraction)
# Estimate the mean and standard deviation of clean EEG from contaminated data.
# [Mu,Sigma,Alpha,Beta] = fit_eeg_distribution(X,MinCleanFraction,MaxDropoutFraction,FitQuantiles,StepSizes,ShapeRange)
#
# This function estimates the mean and standard deviation of clean EEG from a sample of amplitude
# values (that have preferably been computed over short windows) that may include a large fraction
# of contaminated samples. The clean EEG is assumed to represent a generalized Gaussian component in
# a mixture with near-arbitrary artifact components. By default, at least 25% (MinCleanFraction) of
# the data must be clean EEG, and the rest can be contaminated. No more than 10%
# (MaxDropoutFraction) of the data is allowed to come from contaminations that cause lower-than-EEG
# amplitudes (e.g., sensor unplugged). There are no restrictions on artifacts causing
# larger-than-EEG amplitudes, i.e., virtually anything is handled (with the exception of a very
# unlikely type of distribution that combines with the clean EEG samples into a larger symmetric
# generalized Gaussian peak and thereby "fools" the estimator). The default parameters should be
# fine for a wide range of settings but may be adapted to accomodate special circumstances.
#
# The method works by fitting a truncated generalized Gaussian whose parameters are constrained by
# MinCleanFraction, MaxDropoutFraction, FitQuantiles, and ShapeRange. The alpha and beta parameters
# of the gen. Gaussian are also returned. The fit is performed by a grid search that always finds a
# close-to-optimal solution if the above assumptions are fulfilled.
#
# In:
#   X : vector of amplitude values of EEG, possible containing artifacts
#       (coming from single samples or windowed averages)
#
#   MinCleanFraction : Minimum fraction of values in X that needs to be clean
#                      (default: 0.25)
#
#   MaxDropoutFraction : Maximum fraction of values in X that can be subject to
#                        signal dropouts (e.g., sensor unplugged) (default: 0.1)
#
# Out:
#   Mu : estimated mean of the clean EEG distribution
#
#   Sigma : estimated standard deviation of the clean EEG distribution
#
#   Alpha : estimated scale parameter of the generalized Gaussian clean EEG distribution (optional)
#
#   Beta : estimated shape parameter of the generalized Gaussian clean EEG distribution (optional)
#
# This function is part of the clean_rawdata toolbox, copyright Christian Kothe 2013

# this package includes inverse gamma function and others
using SpecialFunctions
using StatsBase

min_clean_fraction = 0.25
max_dropout_fraction = 0.1
quants = [0.022, 0.6]
step_sizes = [0.01, 0.01]
beta = [1.7:0.15:3.5;]'
zbounds = Array{Float64, 2}(undef, length(beta), 2)
rescale = zeros(1,length(beta))
# sort data so we can access quantiles directly
X = sort(X[:])
n = length(X)

# calc z bounds for the truncated standard generalized Gaussian pdf and pdf rescaler
for b=1:length(beta)
    y = sign.(quants .- 0.5) .* (2 .* quants .- 1)
    aa = 1 ./ beta[b]
    # inspect function source: @edit gamma_inc_inv(0.5,0.5,0.5)
    gamma_1 = gamma_inc_inv(aa, y[2], y[2])
    gamma_2 = gamma_inc_inv(aa, y[1], 1-y[1])
    g = [gamma_2, gamma_1]

    zbounds[b,:] = sign.(quants.- 0.5) .* g .^ (1/beta[b])
    rescale[b] = beta[b]/(2*gamma(1/beta[b]))
end

# determine the quantile-dependent limits for the grid search
lower_min = minimum(quants)
max_width = diff(quants)
min_width = min_clean_fraction*max_width

# get matrix of shifted data ranges
indexrange = ((1 : round(n*max_width[1]))' .+ round.(n*(lower_min:step_sizes[1]:lower_min+max_dropout_fraction)))
indexrange = Int64.(indexrange)
X = X[indexrange]
X = X' # to stay compatible with matlab
X1 = X[1,:]
X = X .- X1'

opt_val = Inf
# for each interval width...
for m in [n .* collect(range(max_width[1], step=-step_sizes[2], stop=min_width[1]))]
    # scale and bin the data in the intervals
    nbins = round.(3*log2.(1 .+ m/2))
    H = X[Int64.(1:m),:] .* (nbins ./ X[Int64(m),:])'

    # the histogram counts are computed in a loop here
    edges = [0:nbins-1; Inf]
    #all_weights = Array{Float64, 2}(undef, size(edges)[1], size(H)[2])
    logq = Array{Float64, 2}(undef, size(edges)[1], size(H)[2])
    for w in 1:size(H)[2]
        hw = fit(Histogram, H[:,w], edges)
        # the implementation of histc in matlab slightly varies from julia and returns one extra value which
        # is always zero here bc. the last edge is defined as Inf. Just add this extra zero manually for now
        push!(hw.weights, 0.0)
        #all_weights[:,w] = hw.weights
        logq[:,w] = log.(hw.weights .+ 0.01)
    end

    # for each shape value...
    for b in (1:length(beta))
        bounds = zbounds[b,:];
        # evaluate truncated generalized Gaussian pdf at bin centers
        x = bounds[1] .+ (0.5:(nbins-0.5)) ./ nbins .* diff(bounds)
        p = exp.(-abs.(x) .^ beta[b]) * rescale[b]
        p = p'./sum(p);

        # calc KL divergences
        kl = sum((p' .* (log.(p)' .- logq[(1:end-1),:])), dims=1) .+ log(m)

        # update optimal parameters
        min_val = minimum(kl)
        idx = argmin(kl) # alternative: findmin
        if min_val < opt_val
            opt_val = min_val
            opt_beta = beta[b]
            opt_bounds = bounds
            # note to myself: this is not how I should be using the CartesianIndex, is it?
            opt_lu = (X1[Int64(idx[1])], X1[idx].+X[Int64(m),Int64(idx[1])])
        end
    end
end

# recover distribution parameters at optimum
alpha = (opt_lu[2]-opt_lu[1])./diff(opt_bounds)
mu = opt_lu[1]-opt_bounds[1]*alpha[1]
beta = opt_beta

# calculate the distribution's standard deviation from alpha and beta
sig = sqrt.((alpha.^2) * gamma(3/beta) / gamma(1/beta))

# is this how we define a tuple?
return (mu,sig,alpha,beta)
