% asr julia tests

% make sure we use the same input
X = csvread('C:\Users\sarah\Documents\PhD\Julia\data.csv');
srate = 250;

% define some default parameters
cutoff = 3;
blocksize = 10;
window_len = 0.1;
window_overlap = 0.5;
max_dropout_fraction = 0.1;
min_clean_fraction = 0.3;
N = round(window_len*srate);

% get size of input
[C,S] = size(X);

% hardcode filter coefficients for now
B = [1.7587013141770287, -4.3267624394458641, ...
    5.7999880031015953, -6.2396625463547508, ...
    5.3768079046882207, -3.7938218893374835, ...
    2.1649108095226470, -0.8591392569863763,  0.2569361125627988];

A = [1.0000000000000000, -1.7008039639301735, ...
    1.9232830391058724, -2.0826929726929797, ...
    1.5982638742557307, -1.0735854183930011, ...
    0.5679719225652651, -0.1886181499768189,  0.0572954115997261];

% filter the input
Y = filter(B,A,double(X),[],2);

% estimate a covariance matrix
U = (1/S) * (Y*Y')

% compute mixing matrix
M = sqrtm(U)

% decompose mixing matrix
[V,D] = eig(M)

% project input data into component space
Y = abs(Y'*V)
for c = C:-1:1
    % compute RMS amplitude for each window...
    rms = Y(:,c).^2;
    rms1 = sqrt(sum(rms(bsxfun(@plus,round(1:N*(1-window_overlap):S-N),(0:N-1)')))/N);
    % fit a distribution to the clean part
    [mu(c),sig(c)] = fit_eeg_distribution(rms,min_clean_fraction,max_dropout_fraction);
end

