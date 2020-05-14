# Original comment:
# Processing function for the Artifact Subspace Reconstruction (ASR) method.
# [Data,State] = asr_process(Data,SamplingRate,State,WindowLength,LookAhead,StepSize,MaxDimensions,MaxMemory,UseGPU)
#
# This function is used to clean multi-channel signal using the ASR method. The required inputs are
# the data matrix, the sampling rate of the data, and the filter state (as initialized by
# asr_calibrate). If the data is used on successive chunks of data, the output state of the previous
# call to asr_process should be passed in.
#
# In:
#   Data : Chunk of data to process [#channels x #samples]. This is a chunk of data, assumed to be
#          a continuation of the data that was passed in during the last call to asr_process (if
#          any). The data should be *zero-mean* (e.g., high-pass filtered the same way as for
#          asr_calibrate).
#
#   SamplingRate : sampling rate of the data in Hz (e.g., 250.0)
#
#   State : initial filter state (determined by asr_calibrate or from previous call to asr_process)
#
#   WindowLength : Length of the statistcs window, in seconds (e.g., 0.5). This should not be much
#                  longer than the time scale over which artifacts persist, but the number of samples
#                  in the window should not be smaller than 1.5x the number of channels. Default: 0.5
#
#   LookAhead : Amount of look-ahead that the algorithm should use. Since the processing is causal,
#               the output signal will be delayed by this amount. This value is in seconds and should
#               be between 0 (no lookahead) and WindowLength/2 (optimal lookahead). The recommended
#               value is WindowLength/2. Default: WindowLength/2
#
#   StepSize : The statistics will be updated every this many samples. The larger this is, the faster
#              the algorithm will be. The value must not be larger than WindowLength*SamplingRate.
#              The minimum value is 1 (update for every sample) while a good value is 1/3 of a second.
#              Note that an update is always performed also on the first and last sample of the data
#              chunk. Default: 32
#
#   MaxDimensions : Maximum dimensionality of artifacts to remove. Up to this many dimensions (or up
#                   to this fraction of dimensions) can be removed for a given data segment. If the
#                   algorithm needs to tolerate extreme artifacts a higher value than the default
#                   may be used (the maximum fraction is 1.0). Default 0.66
#
#   MaxMemory : The maximum amount of memory used by the algorithm when processing a long chunk with
#               many channels, in MB. The recommended value is at least 256. To run on the GPU, use
#               the amount of memory available to your GPU here (needs the parallel computing toolbox).
#               default: min(5000,1/2 * free memory in MB). Using smaller amounts of memory leads to
#               longer running times.
#
#   UseGPU : Whether to run on the GPU. This makes sense for offline processing if you have a a card
#            with enough memory and good double-precision performance (e.g., NVIDIA GTX Titan or
#            K20). Note that for this to work you need to have the Parallel Computing toolbox.
#            Default: false
#
# Out:
#   Data : cleaned data chunk (same length as input but delayed by LookAhead samples)
#
#   State : final filter state (can be passed in for subsequent calls)
#
#                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
#                                2012-08-31

# UC Copyright Notice
# This software is Copyright (C) 2013 The Regents of the University of California. All Rights Reserved.
#
# Permission to copy, modify, and distribute this software and its documentation for educational,
# research and non-profit purposes, without fee, and without a written agreement is hereby granted,
# provided that the above copyright notice, this paragraph and the following three paragraphs appear
# in all copies.
#
# Permission to make commercial use of this software may be obtained by contacting:
# Technology Transfer Office
# 9500 Gilman Drive, Mail Code 0910
# University of California
# La Jolla, CA 92093-0910
# (858) 534-5815
# invent@ucsd.edu
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


# Adapted by Sarah Blum, 2018.
# Riemannian processing was added to the processing function in the estimation of the covariance matrix and
# the averaging of covariance matrices. For more details please refer to the paper Blum et al. 2018 (in
# preparation).


# import libraries
using PosDefManifold
using DSP
#using Plots
using LinearAlgebra

function asr_process(data, srate, T)
