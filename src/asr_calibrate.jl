# asr_calibrate
function asr_calibrate(X, srate)

  # import libraries
  using PosDefManifold
  using DSP
  #using Plots
  using LinearAlgebra

  # determine size of input, channels and samples
  C = size(X,1)
  S = size(X,2)

  # define filter coefficients, hardcoded. TODO this should be computed
  # based on the sampling rate:
  # [B,A] = yulewalk(8,[[0 2 3 13 16 40 min(80,srate/2-1)]*2/srate 1],[3 0.75 0.33 0.33 1 1 3 3]);
  B = [1.7587013141770287, -4.3267624394458641,
       5.7999880031015953, -6.2396625463547508,
       5.3768079046882207, -3.7938218893374835,
       2.1649108095226470, -0.8591392569863763,  0.2569361125627988]

  A = [1.0000000000000000, -1.7008039639301735,
       1.9232830391058724, -2.0826929726929797,
       1.5982638742557307, -1.0735854183930011,
       0.5679719225652651, -0.1886181499768189,  0.0572954115997261]

  # define some default parameters
  cutoff = 3
  blocksize = 10
  window_len = 0.1
  window_overlap = 0.5
  max_dropout_fraction = 0.1
  min_clean_fraction = 0.3
  N = round(window_len*srate)

  # filter incoming signal using the predefined filter coefficients
  # note: filter in Julia only operates along the first dimension, so make that the right one,
  # i.e. the same one (second) as in matlab toolbox
  X = X'
  Y = filt(B, A , X)
  # transpose result to be in line with matlab again
  X = Y'

  # compute estimator for covariance matrix
  U = (1/S) * (X * X')

  # mixing matrix is equivalent to using the covariance matrix directly
  # M = sqrt(U)

  # decompose covariance matrix using PCA (this could be nonlinear decomposition as well)
  V = eigvecs(U)

  # project input data into component space (TODO why abs here, squaring anyway afterwards..)
  X = broadcast(abs, (X'*V))

  # for every channel, compute rms and stats to define the threshold matrix
  for c in [C:-1:1]
    c=1
      # compute RMS amplitude for each window...
      rms = X[:,c].^2
      indices = round.([1:(N*(1-window_overlap)):(S-N);])' .+  [0:(N-1);]
      rms = sqrt.(sum(rms[Int.(indices)], dims=1) ./ N)
      # fit a distribution to the clean part
      out = fit_eeg_distribution(rms,min_clean_fraction,max_dropout_fraction)
      out[0] = mu[c]
      out[1] = sig[c]
  end

  T = diag(mu + cutoff*sig)*V';
  return T
