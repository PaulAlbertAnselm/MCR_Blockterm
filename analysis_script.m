
%% Block-PF data example %%
clear all
load Data_exmpl.mat
ncomp = 3; % three component system
nmodels = 20; % number of repetitive fits

%% Construct three-way data from factors
mgc = [156,31,30];
for n = 1:ncomp
    wCbr(:,n,:) = squeeze(Cbr(:,n,:))*(eye(30,30).*C(:,n)*3);
end

X = zeros(mgc);

for n = 1:ncomp
    for i = 1:mgc(1)
        X_temp(i,:,:) = squeeze(wCbr(:,n,:))*A(i,n);
    end
    X = X+X_temp;
end

X_noisy = addTensorNoise(X,40); % requires probabilistic Tensor Toolbox

%% Step 1: Make plots of raw data
CBr1 = squeeze(Cbr(:,1,:));
CBr2 = squeeze(Cbr(:,2,:));
CBr3 = squeeze(Cbr(:,3,:));

figure;
subplot(3,1,3)
plot(CBr1./vecnorm(CBr1),'LineWidth',2,'Color',[0 0.4470 0.7410])
hold on
axis tight
subplot(3,1,1)
plot(CBr2./vecnorm(CBr2),'LineWidth',2,'Color',[0 0.4470 0.7410])
hold on
axis tight
subplot(3,1,2)
plot(CBr3./vecnorm(CBr3),'LineWidth',2,'Color',[0 0.4470 0.7410])
axis tight

figure;
subplot(3,1,3)
plot(A(:,1),'LineWidth',2)
hold on
axis tight
subplot(3,1,1)
plot(A(:,2),'LineWidth',2)
hold on
axis tight
subplot(3,1,2)
plot(A(:,3),'LineWidth',2)
axis tight


%% Step 2: Run models

fit_all = [];
for i = 1:nmodels
    options_2.MaxIter = 1000;
    options_2.ConvCrit = 1e-08;
    options_2.Constr = [1 1];
    options_2.Init = 0;
    options_2.stri.pccf = 1;
    options_2.stri.fit_stri = 0;
    options_2.stri.fit_pf = 1;
    options_2.stri.pccf_ts = [0.001 3]; % [reconstruction error max_comps]

m_x{i} = Stri(X_noisy,ncomp,options_2);
fit_all(2,i) = m_x{i}.pfm.detail.all_fit(end); %collect all fits

% find model with highest fit
if i == 1
    fit_opt = fit_all(1,i);
    ind_x = 1;
else
    if fit_all(1,i) > fit_opt
        fit_opt = fit_all(1,i);
        ind_x = i;
    end
end

end

% select only model with highest fit out of 'nmodels'
m = m_x{ind_x};

%% Step 3: Plot the results (Figure 4+6 in paper)

% Sorting components to match order of the references 
A_hat = m.Factors{1};  
ind = sort_comps(A,A_hat);
A_hat = A_hat(:,ind);
B_hat = m.Factors{2}(:,ind,:);
C_hat = m.Factors{3}(:,ind);

CBr1_hat = (squeeze(B_hat(:,1,:))'.*C_hat(:,1))';
CBr2_hat = (squeeze(B_hat(:,2,:))'.*C_hat(:,2))';
CBr3_hat = (squeeze(B_hat(:,3,:))'.*C_hat(:,3))';

% Plot recovery of the spectra (Figure 4+6A)
figure; 
subplot(3,1,3)
plot(A(:,1),'Color',[0 0.4470 0.7410],'LineWidth',2)
hold on
plot(A_hat(:,1),'Color',[0.8500 0.3250 0.0980],'LineWidth',4,'LineStyle','--')
%legend({'True';'Estimate'})
axis tight
subplot(3,1,1)
plot(A(:,2),'Color',[0 0.4470 0.7410],'LineWidth',2)
hold on
plot(A_hat(:,2),'Color',[0.8500 0.3250 0.0980],'LineWidth',4,'LineStyle','--')
axis tight

subplot(3,1,2)
plot(A(:,3),'Color',[0 0.4470 0.7410],'LineWidth',2)
hold on
plot(A_hat(:,3),'Color',[0.8500 0.3250 0.0980],'LineWidth',4,'LineStyle','--')
axis tight

% Plot recovery of elution profiles (Figure 4+6B)
figure; 
subplot(3,1,3)
plot(squeeze(wCbr(:,1,1:3)),'Color',[0 0.4470 0.7410],'LineWidth',2);
hold on
plot(CBr1_hat(:,1:3),'Color',[0.8500 0.3250 0.0980],'LineWidth',4,'LineStyle','--')
axis tight

subplot(3,1,1)
hold on
plot(squeeze(wCbr(:,2,1:3)),'Color',[0 0.4470 0.7410],'LineWidth',2);
hold on
plot(CBr2_hat(:,1:3),'Color',[0.8500 0.3250 0.0980],'LineWidth',4,'LineStyle','--')
axis tight

subplot(3,1,2)
hold on
plot(squeeze(wCbr(:,3,1:3)),'Color',[0 0.4470 0.7410],'LineWidth',2);
hold on
plot(CBr3_hat(:,1:3),'Color',[0.8500 0.3250 0.0980],'LineWidth',4,'LineStyle','--')
axis tight


% Plot differences in fit values, to see if global optimum was reached (Figure 4,6C)
figure; 
bar(sort(-log10(abs((fit_all(1,1:end)-max(fit_all(1,:))+1e-12))),'descend'))
hold on
yline(-log10(options_2.ConvCrit*10),'Color','r','LineWidth',2,'LineStyle','--')


%% Step 4: Visualize AFS for Figure 5A --> for newly simulated data this requires AFS calculation in FACPACK software

load results_facpack_all_D_rs_1.mat
afs_rs = reshape(TransformedAFS{1},size(TransformedAFS{1},1),size(X,2),size(X,3));

CBr1_matr = CBr1';
CBr1_matr = CBr1_matr(:);
CBr2_matr = CBr2';
CBr2_matr = CBr2_matr(:);
CBr3_matr = CBr3';
CBr3_matr = CBr3_matr(:);

figure;

for i = 1:3
subplot(3,3,(1+(i-1)*3))
plot(squeeze(afs_rs(:,:,i)')./vecnorm(squeeze(afs_rs(:,:,i)')),'Color',[0.6 0.6 0.6])


hold on
plot(CBr1(:,i)./norm(CBr1(:,i),'fro'),'Color',[0 0.4470 0.7410],'LineWidth',2)
axis tight

subplot(3,3,(2+(i-1)*3))
plot(squeeze(afs_rs(:,:,i)')./vecnorm(squeeze(afs_rs(:,:,i)')),'Color',[0.6 0.6 0.6])
axis tight

hold on
plot(CBr2(:,i)./norm(CBr2(:,i),'fro'),'Color',[0 0.4470 0.7410],'LineWidth',2)
axis tight

subplot(3,3,(3+(i-1)*3))
plot(squeeze(afs_rs(:,:,i)')./vecnorm(squeeze(afs_rs(:,:,i)')),'Color',[0.6 0.6 0.6])

hold on
plot(CBr3(:,i)./norm(CBr3(:,i),'fro'),'Color',[0 0.4470 0.7410],'LineWidth',2)
axis tight
end



%% Step 5: Visualize AFS for Figure 5B --> for newly simulated data this requires AFS calculation in FACPACK software

load results_facpack_all_D_rs_2.mat

figure;
subplot(3,1,3)
plot(TransformedAFS{1}'./vecnorm(TransformedAFS{1}'),'Color',[0.6 0.6 0.6])
hold on
plot(A(:,1),'Color',[0 0.4470 0.7410],'LineWidth',2)
axis tight
subplot(3,1,1)
plot(TransformedAFS{1}'./vecnorm(TransformedAFS{1}'),'Color',[0.6 0.6 0.6])
hold on
plot(A(:,2),'Color',[0 0.4470 0.7410],'LineWidth',2)
axis tight
subplot(3,1,2)
plot(TransformedAFS{1}'./vecnorm(TransformedAFS{1}'),'Color',[0.6 0.6 0.6])
hold on
plot(A(:,3),'Color',[0 0.4470 0.7410],'LineWidth',2)
axis tight

%% Figure 5C
figure; 
bar(sort(-log10(abs(fit_all(2:end)-max(fit_all))),'descend'))
hold on
yline(-log10(options.ConvCrit*10),'Color','r','LineWidth',2,'LineStyle','--')


function ind = sort_comps(A_ref,A_emp)

cm = A_emp'*A_ref;
ind = [];
for i = 1:size(A_ref,2)
    [~,ind(i)] = max(cm(:,i));
end
if ~isempty(setdiff(1:max(size(cm)),ind))
   ind_missing = setdiff(1:max(size(cm)),ind);
   [~,ind_fill] = max(cm(ind_missing,:));
   ind(ind_fill) = ind_missing;
end

end


