%% second order calibration for revision
clear all

%% Simulate data


% Three Component matrices of rank three
B1 = [gaussmf(1:160,[10 45])' gaussmf(1:160,[5 50])' gaussmf(1:160,[10 55])']; 
C1 = rand(30,3);
CB1 = B1*C1';
CB1 = CB1./vecnorm(CB1);

B2 = [gaussmf(1:160,[10 55])' gaussmf(1:160,[5 60])' gaussmf(1:160,[10 70])']; 
C2 = rand(30,3);
CB2 = B2*C2';
CB2 = CB2./vecnorm(CB2);

B3 = [gaussmf(1:160,[10 70])' gaussmf(1:160,[5 75])' gaussmf(1:160,[10 80])']; 
C3 = rand(30,3);
CB3 = B3*C3';
CB3 = CB3./vecnorm(CB3);

Cbr(:,1,:) = CB1;
Cbr(:,2,:) = CB2;
Cbr(:,3,:) = CB3;

% Spectra for rank-1 mode
A1 = [gaussmf(1:100,[5 15])' gaussmf(1:100,[5 30])' gaussmf(1:100,[5 45])'];
c1 = rand(3,1);
A1c1 = A1*c1;
A1c1 = A1c1./vecnorm(A1c1);
figure;plot(A1c1)
A2 = [gaussmf(1:100,[10 10])' gaussmf(1:100,[5 20])'];
c2 = rand(2,1);
A2c2 = A2*c2;
A2c2 = A2c2./vecnorm(A2c2);
figure;plot(A2c2)
A3 = [gaussmf(1:100,[5 30])' gaussmf(1:100,[10 45])'];
c3 = rand(2,1);
A3c3 = A3*c3;
A3c3 = A3c3./vecnorm(A3c3);
A = [A1c1 A2c2 A3c3];

% define calibration samples of defined concentrations
c1 = [0.25; 0.5; 0.75; 0; 1; 0; 0.25; 1; 0.5];
c2 = [0; 0.25; 0; 0.5; 0; 0.75; 0; 1; 0];
c3 = [zeros(9,1)];
C = [c1 c2 c3];
B1 = zeros(60,9);
B2 = zeros(60,9);
B3 = zeros(60,9);
smpls = [randperm(30,9)];
B1 = squeeze(Cbr(:,1,smpls));
B1 = B1./vecnorm(B1);
CB1 = B1.*c1';

B2 = squeeze(Cbr(:,2,smpls));
B2 = B2./vecnorm(B2);
CB2 = B2.*c2';

B3 = squeeze(Cbr(:,3,smpls));
B3 = B3./vecnorm(B3);
CB3 = B3.*c3';

% creating the three-way array of the calibration data

X1 = A(:,1)*reshape(CB1,1,size(CB1,1)*size(CB1,2));
X2 = A(:,2)*reshape(CB2,1,size(CB2,1)*size(CB2,2));
X3 = A(:,3)*reshape(CB3,1,size(CB3,1)*size(CB3,2));

X_cal = X1+X2+X3;
X_cal_3way = reshape(X_cal,size(A,1),size(B1,1),size(C,1));

% creating concentrations for the testsamples
testsmpls = setdiff(1:30,smpls);
c1_test_all = rand(length(testsmpls),1);
c2_test_all = rand(length(testsmpls),1);
c3_test_all = rand(length(testsmpls),1);

%
for iii = 1:9 % fit models with Lr=1-9
for i = 1:length(testsmpls)

c1_test = c1_test_all(i,1);
c2_test = c2_test_all(i,1);
c3_test = c3_test_all(i,1);

B1_test = zeros(160,1);
B2_test = zeros(160,1);
B3_test = zeros(160,1);

B1_test = squeeze(Cbr(:,1,testsmpls(i)));
B1_test = B1_test./vecnorm(B1_test);
CB1_test = B1_test.*c1_test;
X1_test = A(:,1)*reshape(CB1_test,1,size(CB1_test,1)*size(CB1_test,2));
B2_test = squeeze(Cbr(:,2,testsmpls(i)));
B2_test = B2_test./vecnorm(B2_test);
CB2_test = B2_test.*c2_test';
X2_test = A(:,2)*reshape(CB2_test,1,size(CB2_test,1)*size(CB2_test,2));
B3_test = squeeze(Cbr(:,3,testsmpls(i)));
B3_test = B3_test./vecnorm(B3_test);
CB3_test = B3_test.*c3_test';
X3_test = A(:,3)*reshape(CB3_test,1,size(CB3_test,1)*size(CB3_test,2));

X_test = X1_test+X2_test+X3_test;

X_final = X_cal_3way;
X_final(:,:,size(X_cal_3way,3)+1) = X_test;

X_noisy = addTensorNoise(X_final,40); % requires probabilistic Tensor Toolbox (https://github.com/JesperLH/prob-tensor-toolbox)
X_noisy = permute(X_noisy,[2 3 1]);
for ii = 1:5
ncomp = 3;
options_2.MaxIter = 1000;
options_2.ConvCrit = 10e-08;
options_2.Constr = [1 1];
options_2.Init = 0;
% setting the reconstruction threshold low so that always maximum number of
% allowed block terms is used
options_2.sist_threshold = [0.001 iii];   

m_x{ii} = MCRBTD(X_noisy,ncomp,options_2);
fit_all(ii) = m_x{ii}.detail.fit.PercVar; %collect all fits

% find model with highest fit
if ii == 1
    fit_opt = fit_all(ii);
    ind_x = 1;
else
    if fit_all(ii) > fit_opt
        fit_opt = fit_all(ii);
        ind_x = ii;
    end
end

end
% best fitted model
m_temp = m_x{ind_x};

% convergence plots
fitdif{i,iii} = m_temp.detail.fit.fitdif;
pccf_all{i,iii} = m_temp.detail.pccf_all;

cos_angle = m_temp.spectra'*A;

[~,ind_1] = max(cos_angle(:,1));
[~,ind_2] = max(cos_angle(:,2));
% root mean squared error 
abs_er_1(i,iii) = abs(c1_test-m_temp.scores(10,ind_1));
abs_er_2(i,iii) = abs(c2_test-m_temp.scores(10,ind_2));

spectral_sim_1(i,iii) = max(cos_angle(:,1));
spectral_sim_2(i,iii) = max(cos_angle(:,2));
end


% creating example plots of grund truth vs estimated factors
colors_ = colororder;
figure;

for ii = 1:5
    subplot(1,2,1)
    hold on
    plot(m_x{ii}.spectra,'LineWidth',3,'Color',[0.7 0.7 0.7])
    hold on
    plot(A,'LineWidth',2,'Color','k','LineStyle',':')
end

for ii = 1:5
    subplot(1,2,2)
    for i = 1:3
        CB_hat(:,i,:) = squeeze(m_x{ii}.elutionprofiles(:,i,:)).*m_x{ii}.scores(:,i)';
    end
    hold on
    plot(squeeze(CB_hat(:,:,10)),'LineWidth',3,'Color',[0.7 0.7 0.7])
    hold on
    plot([CB1_test CB2_test CB3_test],'LineWidth',2,'Color','k','LineStyle',':')
end

savefig(['comps_' num2str(iii)])
close all

end
%% plot an example of a data set
colors_ = colororder;
figure;
subplot(2,3,[1 3])
plot(sum([X1 X1_test],1),'LineWidth',2);
hold on
plot(sum([X2 X2_test],1),'LineWidth',2);
hold on
plot(sum([X3 X3_test],1),'LineWidth',2);
xline(1440,'k:','LineWidth',1.5)

subplot(2,3,4)
plot(A(:,1),'LineWidth',2,'Color',colors_(1,:))
subplot(2,3,5)
plot(A(:,2),'LineWidth',2,'Color',colors_(2,:))
subplot(2,3,6)
plot(A(:,3),'LineWidth',2,'Color',colors_(3,:))

