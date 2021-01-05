# Original comment:
# Calibration function for the Artifact Subspace Reconstruction (ASR) method.
# State = asr_calibrate(Data,SamplingRate,Cutoff,BlockSize,FilterB,FilterA,WindowLength,WindowOverlap,MaxDropoutFraction,MinCleanFraction)
#
# The input to this data is a multi-channel time series of calibration data. In typical uses the
# calibration data is clean resting EEG data of ca. 1 minute duration (can also be longer). One can
# also use on-task data if the fraction of artifact content is below the breakdown point of the
# robust statistics used for estimation (50% theoretical, ~30% practical). If the data has a
# proportion of more than 30-50% artifacts then bad time windows should be removed beforehand. This
# data is used to estimate the thresholds that are used by the ASR processing function to identify
# and remove artifact components.
#
# The calibration data must have been recorded for the same cap design from which data for cleanup
# will be recorded, and ideally should be from the same session and same subject, but it is possible
# to reuse the calibration data from a previous session and montage to the extent that the cap is
# placed in the same location (where loss in accuracy is more or less proportional to the mismatch
# in cap placement).
#
# The calibration data should have been high-pass filtered (for example at 0.5Hz or 1Hz using a
# Butterworth IIR filter).
#
# In:
#   Data : Calibration data [#channels x #samples]; *zero-mean* (e.g., high-pass filtered) and
#          reasonably clean EEG of not much less than 30 seconds length (this method is typically
#          used with 1 minute or more).
#
#   SamplingRate : Sampling rate of the data, in Hz.
#
#
#   The following are optional parameters (the key parameter of the method is the RejectionCutoff):
#
#   RejectionCutoff: Standard deviation cutoff for rejection. Data portions whose variance is larger
#                    than this threshold relative to the calibration data are considered missing
#                    data and will be removed. The most aggressive value that can be used without
#                    losing too much EEG is 2.5. A quite conservative value would be 5. Default: 5.
#
#   Blocksize : Block size for calculating the robust data covariance and thresholds, in samples;
#               allows to reduce the memory and time requirements of the robust estimators by this
#               factor (down to Channels x Channels x Samples x 16 / Blocksize bytes). Default: 10
#
#   FilterB, FilterA : Coefficients of an IIR filter that is used to shape the spectrum of the signal
#                      when calculating artifact statistics. The output signal does not go through
#                      this filter. This is an optional way to tune the sensitivity of the algorithm
#                      to each frequency component of the signal. The default filter is less
#                      sensitive at alpha and beta frequencies and more sensitive at delta (blinks)
#                      and gamma (muscle) frequencies. Default:
#                      [b,a] = yulewalk(8,[[0 2 3 13 16 40 min(80,srate/2-1)]*2/srate 1],[3 0.75 0.33 0.33 1 1 3 3]);
#
#   WindowLength : Window length that is used to check the data for artifact content. This is
#                  ideally as long as the expected time scale of the artifacts but short enough to
#				   allow for several 1000 windows to compute statistics over. Default: 0.5.
#
#   WindowOverlap : Window overlap fraction. The fraction of two successive windows that overlaps.
#                   Higher overlap ensures that fewer artifact portions are going to be missed (but
#                   is slower). Default: 0.66
#
#   MaxDropoutFraction : Maximum fraction of windows that can be subject to signal dropouts
#                        (e.g., sensor unplugged), used for threshold estimation. Default: 0.1
#
#   MinCleanFraction : Minimum fraction of windows that need to be clean, used for threshold
#                      estimation. Default: 0.25
#
#
# Out:
#   State : initial state struct for asr_process
#
# Notes:
#   This can run on a GPU with large memory and good double-precision performance for faster processing
#   (e.g., on an NVIDIA GTX Titan or K20), but requires that the Parallel Computing toolbox is
#   installed.
#
#                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
#                                2012-08-31

# asr_calibrate_version<1.03> -- for the cache

# UC Copyright Notice
# This software is Copyright (C) 2013 The Regents of the University of California. All Rights Reserved.
#
# Permission to copy, modify, and distribute this software and its documentation for educational,
# research and non-profit purposes, without fee, and without a written agreement is hereby granted,
# provided that the above copyright notice, this paragraph and the following three paragraphs appear
# in all copies.
#
# This software program and documentation are copyrighted by The Regents of the University of
# California. The software program and documentation are supplied "as is", without any accompanying
# services from The Regents. The Regents does not warrant that the operation of the program will be
# uninterrupted or error-free. The end-user understands that the program was developed for research
# purposes and is advised not to rely exclusively on the program for any reason.
#
# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
# SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF
# THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
# CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
# MODIFICATIONS.

# import libraries
using PosDefManifold
using DSP
using LinearAlgebra

function asr_calibrate(X, srate)
  # determine size of input, channels and samples
  C = size(X, 1)
  S = size(X, 2)

  # define filter coefficients, hardcoded. TODO this should be computed
  # based on the sampling rate:
  # [B,A] = yulewalk(8,[[0 2 3 13 16 40 min(80,srate/2-1)]*2/srate 1],[3 0.75 0.33 0.33 1 1 3 3]);
  B = [
    1.7587013141770287,
    -4.3267624394458641,
    5.7999880031015953,
    -6.2396625463547508,
    5.3768079046882207,
    -3.7938218893374835,
    2.1649108095226470,
    -0.8591392569863763,
    0.2569361125627988,
  ]

  A = [
    1.0000000000000000,
    -1.7008039639301735,
    1.9232830391058724,
    -2.0826929726929797,
    1.5982638742557307,
    -1.0735854183930011,
    0.5679719225652651,
    -0.1886181499768189,
    0.0572954115997261,
  ]

  # define some default parameters
  cutoff = 3
  blocksize = 10
  window_len = 0.1
  window_overlap = 0.5
  max_dropout_fraction = 0.1
  min_clean_fraction = 0.3
  N = round(window_len * srate)

  # filter incoming signal using the predefined filter coefficients
  # note: filter in Julia only operates along the first dimension, so make that the right one,
  # i.e. the same one (second) as in matlab toolbox
  X = X'
  Y = filt(B, A, X)
  # transpose result to be in line with matlab again
  X = Y'

  # compute estimator for covariance matrix
  U = (1 / S) * (X * X')

  # mixing matrix is equivalent to using the covariance matrix directly
  M = sqrt(U)

  # decompose covariance matrix using PCA (this could be nonlinear decomposition as well)
  V = eigvecs(M)

  # project input data into component space
  X = broadcast(abs, (X' * V))

  # initialize arrays for distribution estimation
  out = Array{Float64,2}(undef, 1,2)
  mu = Array{Float64,2}(undef, 1,C)
  sig = Array{Float64,2}(undef, 1,C)

  # for every channel, compute rms and stats to define the threshold matrix
  for c in C:-1:1
    # compute RMS amplitude for each window...
    rms = X[:, c] .^ 2
    indices = round.([1:(N * (1 - window_overlap)):(S - N);])' .+ [0:(N-1);]
    rms = sqrt.(sum(rms[Int.(indices)], dims = 1) ./ N)
    # fit a distribution to the clean part
    #FIXME X is not exactly the same as in Matlab anymore (very small difference)
    out = fit_eeg_distribution(rms, min_clean_fraction, max_dropout_fraction)
    mu[c] = out[1][1]
    sig[c] = out[2][1]
  end

  # assemble output
  T = diag(mu + cutoff * sig) .* V'
  T = T * -1 # correct for sign of the eigenvectors (in this testcase flipped compared to Matlab)
  return (T,M,B,A)

end
