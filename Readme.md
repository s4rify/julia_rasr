# Riemannian ASR in Julia (Julia rASR)

This is a (work in progress) implementation of the Riemannian ASR method (rASR, [1]), recently published to the BCI community.

rASR is an online artifact handling method which applies principal component analysis to detect deviating data in the component subspace and corrects potentially artifactual data. As compared to ASR [2], rASR exploits Riemannian geometry to achieve higher performance.

So far, rASR has been implemented only in Matlab [github.com/sccn/clean_rawdata]. Here, we present an implementation of rASR in the Julia programming language. 

With Julia rASR we hope to contribute to the growing field of artifact handling methods for mobile EEG-based BCI systems. Our implementation will be presented on the CORTICO meeting: https://corticodays.sciencesconf.org/ 

This implementation has a dependency to the PosDefManifold toolbox: https://github.com/Marco-Congedo/PosDefManifoldML.jl



[1] Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019). A Riemannian Modification of Artifact Subspace Reconstruction for EEG Artifact Handling. Frontiers in Human Neuroscience, 13, 141. 

[2] Mullen, T. R., Kothe, C. A. E., Chi, Y. M., Ojeda, A., Kerth, T., Makeig, S., … Cauwenberghs, G. (2015). Real-Time Neuroimaging and Cognitive Monitoring Using Wearable Dry EEG. IEEE Transactions on Bio-Medical Engineering, 62(11), 2553–2567. 


# Updates
13.03.2020 The asr_calibrate method has been translated, but not been tested. The asr_process method is work in progress and is currently being translated.