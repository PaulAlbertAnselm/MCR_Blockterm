
function model = Stri_update_2(X,Fac,options)
% I/O
%
% I:
% X(spectral chanels x elutiontimes x samples) = Low rank chromatographic data set
% Fac = Number of latent variables to be fitted
% options.MaxIter   :   maximum number of iterations (default = 1000)
% options.ConvCrit  :   convergence criterion (default = 1e-09)
% options.Constr    :   [Spectra, Elutionprofiles&Scores]
%                       e.g [1 1] (default)
%                       0 = unconstrained
%                       1 = non-negativity
% options.Init      :   Initialization
%                   :   0 = random
%                   :   2 = custom (requires options.InitLoads)
% options.InitLoads :   insert customized Loadings
%
% options.stri          :   set of options specifying how trilinearity (multilinearity is imposed)
% options.stri.fit_stri :   0 = bilinear model
%                           1 = apply trilinearity (multilinearity)
% options.stri.fit_pf   :   is set to zero if "options.stri.fit_stri = 0"
%                           1 = refine stri solution in every iteration
%                           2 = fit PARAFAC when loss function isnt decreasing
%                           3 = fit PARAFAC model after convergence
%
% options.stri.pccf     :   0 = trilinear model
%                           1 = fitting blockterm model
% options.stri.pccf_ts  :   [tol_err #svs]
%
%
%O:
%model.spectra                      = (spectral chanels x Fac)
%model.elutionprofiles              = (elutiontimes x Fac x samples)
%model.scores                       = (samples x Fac)
%model.detail.fit.X_sum_sq          = Total Sum of Squares;
%model.detail.fit.res_sum_sq        = Residual Sum of Squares;
%model.detail.fit.PercVar           = Explained Variance [%];
%model.detail.fit.fitdif            = Final difference in Fit;
%model.detail.lossfunc              = Loss Function values over all iterations;
%model.detail.lossfunc_dif          = Fit difference over all iterations;
%model.detail.residuals             = Residuals in dimensions of input data;
%model.detail.iterations            = Number of Iterations
%model.detail.time                  = Computation time;
%model.detail.Profiles.PeakType     = PeakType
%model.detail.Profiles.Niceness     = Niceness
%model.detail.Profiles.PeakTypeHelp = '0 = weird, 1 = peak, 2 = cutoff, 3 = baseline';


%% Unfold X
mgc    = size(X);
X1 = matricizing(X,1);

%'Mode 1 is elution time * sample
%'Mode 2 is m/z fragments
%% Arguments and option settings
tic()
ncomp           = Fac;
if nargin < 3
    init        = 1;
    maxit       = 1000;
    constr      = [1 1];
    convcrit    = 1e-09;
    pccf        = 1;
else
    if isfield(options,'Init')
        init        = options.Init;
    else
        init    = 1;
    end
    maxit       = options.MaxIter;
    if isfield(options,'Constr')
        constr      = options.Constr;
    else
        constr = [1 1];
    end
    convcrit    = options.ConvCrit;
    if isfield(options,'stri') 
        if isfield(options.stri,'fit_stri')
            fit_stri = options.stri.fit_stri;
        else
            fit_stri = 1; % per default tri-linear
        end
        if isfield(options.stri,'fit_pf')
            fit_pf = options.stri.fit_pf;
        end
        if isfield(options.stri, 'pccf')
            if options.stri.pccf == 1
                pccf_set        = options.stri.pccf;
                if isfield(options.stri, 'pccf_ts')
                    params_sist = options.stri.pccf_ts;
                else
                    disp('Warning: No thresholds found, default settings might lead to errors');
                    params_sist = [0.01 3];
                end
            else
                pccf_set        = options.stri.pccf;
                params_sist     = [];
            end
        end
    else
        disp('Warning: No arguments specifying which algorithm should be used')
        disp('Fitting MCR-Trilinearity and PARAFAC model when loss gets negative')
        fit_stri = 1;
        params_sist = [];
        fit_pf = 2;
        pccf_set = 0;
    end

end

%% Initialization of S or C
% pure sample / spectra

if init == 0
    CB = randn(size(X1,2),ncomp);
elseif init == 2
    CB = options.InitLoads;
end

%% Initialization of Loss
SST = sum(X1.^2,'all');
SSE = zeros(1,2);
SSE(1) = 2*SST;
SSE(2) = SST;
fitdif = SSE(1)-SSE(2);
fit = [];
iter = [];
it = 0;
%% calculating mcr solution
% X = CS'
% S has Norm 1
% S = S/||S||
while fitdif(end) > convcrit & it < maxit
    it = it+1;
    CtC = CB'*CB;
    CtX = CB'*X1';
    if      constr(1) == 0
        S = CtX'*pinv(CtC);
    elseif  constr(1) == 1
        try
            S = fcnnls(CB, X1, CtC, CtX)';
            for k =1:ncomp
                if sum(S(:,k)) == 0
                    S(:,k) = rand(1,size(S,1));
                end
            end
        catch
            CB = randn(size(CB));
            CtC = CB'*CB;
            CtX = CB'*X1';
            S = fcnnls(CB, X1, CtC, CtX)';
        end
    end
    for k = 1:ncomp
        S(:,k) = S(:,k)*1/norm(S(:,k),'fro');
    end
    StS = S'*S;
    StXt = S'*X1;
    if      constr(2) == 0
        CB = StXt'*pinv(StS);
    elseif  constr(2) == 1
        try
            CB = fcnnls([],[],StS,StXt)';
        catch
            S = randn(size(S));
            for k = 1:ncomp
                S(:,k) = S(:,k)*1/norm(S(:,k),'fro');
            end
            StS = S'*S;
            StXt = S'*X1;
            CB = fcnnls([],[],StS,StXt)';
        end
    end

    if fit_stri == 1
        [CB_hat,pccf_temp] = stri_constraint(CB,mgc,ncomp,params_sist,pccf_set);
        CB = CB_hat;
    elseif fit_pf == 1
        [~,B,C,pccf_temp,~] = pf(CB,S,X,X1,SST,maxit,convcrit,constr,ncomp,pccf_set,params_sist,fit_pf);
        CB_hat = krbc(C,B,pccf_temp);
        CB = CB_hat;
    end

    %evaluating loss function
    SSE(2) = SSE(1);
    SSE(1) = SST+sum(sum((CB'*CB) .*(S'*S)))-2*sum((S'*X1)' .* CB,'all');
    all_fit(it) = (1-SSE(1)/SST)*100;
    fitdif(it) = abs((SSE(1)-SSE(2))/SSE(2));
    all_err(it) = SSE(1);
    if fit_stri ~= 0
        all_pccf(:,it) = pccf_temp;
    end

end

if fit_pf == 2
        [S,B,C,pccf_temp,pfm] = pf(CB,S,X,X1,SST,maxit,convcrit,constr,ncomp,pccf_set,params_sist,fit_pf);
        CB_hat = krbc(C,B,pccf_temp);
        CB = CB_hat;
        B = [];
        C = [];
end

% Outputs
CB = reshape(CB,mgc(2),mgc(3),ncomp);

for i = 1:ncomp
    for ii = 1:mgc(3)
    C(ii,i) = norm(squeeze(CB(:,ii,i)),'fro');
    B(:,i,ii) = squeeze(CB(:,ii,i))./C(ii,i);
    %[U,T,V] = svd(CBr,'econ');
    %B(:,i)  = abs(U(:,1));
    %C(:,i)  = abs(T(1:1,1:1)*V(:,1));
    end
end
model.Factors{1}   = S;
model.Factors{2}   = B;
model.Factors{3}   = C;

model.detail.fit.X_sum_sq        = SST;
model.detail.fit.res_sum_sq      = SSE(1);
model.detail.fit.PercVar         = all_fit(end);
model.detail.fit.fitdif          = fitdif;
model.detail.fit.allfit          = all_fit;
model.detail.fit.allSSE          = all_err;
if fit_stri ~= 0
    model.detail.fit.pccf            = all_pccf;
end
model.detail.iterations    = it;
model.detail.time          = toc();

if fit_pf == 2
    model.pfm = pfm;
end

end

function Xn = matricizing(X,n)
mgc = size(X);
norder = 1:length(mgc);
order_unfold = setdiff(norder,n);

Xn = permute(X,[n order_unfold]);
Xn = reshape(Xn,mgc(n),prod(mgc(order_unfold)));

end

function pccf = get_pccf(t_temp,params_sist)

        d_ttemp = diag(t_temp);
        ixi = 1;
        while (1-sum(d_ttemp(1:ixi).^2)/sum(d_ttemp.^2)) > params_sist(1)
            ixi = ixi+1;
        end
        if ixi < params_sist(2)
            pccf = ixi;
        else
            pccf = params_sist(2);
        end

end

function S = krbc(C,B,pccf_temp)

if size(C,2) == size(B,2)
   ncomp = length(pccf_temp);
   Lr = [0 0];
    for i = 1:ncomp
        Lr(2) = pccf_temp(i);
        C_Lr = C(:,Lr(1)+1:(Lr(2)+Lr(1)));
        B_Lr = B(:,Lr(1)+1:(Lr(2)+Lr(1)));
        S(:,i) = sum(krb(C_Lr,B_Lr),2);
        Lr(1) = sum(Lr);
    end
elseif size(C,2) > size(B,2)
       ncomp = length(pccf_temp);
       Lr = [0 0];
       for i = 1:ncomp
           Lr(2) = pccf_temp(i);
           C_Lr = C(:,Lr(1)+1:(Lr(2)+Lr(1)));
           B_rep = repmat(B(:,i),1,Lr(2));
           S(:,Lr(1)+1:(Lr(2)+Lr(1))) = krb(C_Lr,B_rep);
           Lr(1) = sum(Lr);
       end
elseif size(C,2) < size(B,2)
       ncomp = length(pccf_temp);
       Lr = [0 0];
       for i = 1:ncomp
           Lr(2) = pccf_temp(i);
           B_Lr = B(:,Lr(1)+1:(Lr(2)+Lr(1)));
           C_rep = repmat(C(:,i),1,Lr(2));
           S(:,Lr(1)+1:(Lr(2)+Lr(1))) = krb(C_rep,B_Lr);
           Lr(1) = sum(Lr);
       end
end
end

function [CB_hat,pccf_temp] = stri_constraint(CB,mgc,ncomp,params_sist,pccf_set)

CB_hat = zeros(size(CB));
U = zeros(mgc(2),ncomp);
T = zeros(ncomp,ncomp);
V = zeros(mgc(3),ncomp);
for r = 1:ncomp
    CBr = reshape(CB(:,r),mgc(2),mgc(3));%shiftmap(copt1');
    %making trilinearity for shifted data

    [u_temp,t_temp,v_temp] = svd(CBr,'econ');
    if pccf_set == 1
        pccf = get_pccf(t_temp,params_sist);
    else
        pccf = 1;
    end
    pccf_temp(r) = pccf;
    %storing results as elutionprofiles and relative concentrations
    CBr_hat = (u_temp(:,1:pccf)*t_temp(1:pccf,1:pccf))*v_temp(:,1:pccf)';
    CB_hat(:,r) = CBr_hat(:);

end

end

function [A,B,C,pccf_temp,pfm] = pf(CB,S,X,X1,SST,maxit,convcrit,constr,ncomp,pccf_set,params_sist,fit_pf)

% precompute matricized X;
% X(I x K x J) --> X1 (I x JK), already available
% X(I x K x J) --> X2 (J x IK), need to compute
% X(I x K x J) --> X3 (K x JI), need to compute
mgc = size(X);
CB = reshape(CB,mgc(2),mgc(3),ncomp);

for i = 1:ncomp
    CBr = squeeze(CB(:,:,i));
    [U,T,V] = svd(CBr,'econ');
    if pccf_set == 1
       pccf = get_pccf(T,params_sist); 
       B_temp{i} = U(:,1:pccf);
       C_temp{i} = V(:,1:pccf)*T(1:pccf,1:pccf);
       pccf_temp(i) = pccf;
    else
       pccf = 1; 
       B(:,i) = abs(U(:,1:pccf));
       C(:,i) = abs(T(1:pccf,1:pccf)*V(:,1:pccf));
       pccf_temp = ones(1,ncomp);
    end

end


if pccf_set == 1
   Lr = [0 0];
   for l = 1:length(pccf_temp)
       Lr(2) = pccf_temp(l);
       B(:,Lr(1)+1:(Lr(2)+Lr(1))) = B_temp{l};
       C(:,Lr(1)+1:(Lr(2)+Lr(1))) = C_temp{l};
       Lr(1) = sum(Lr);
   end
end

X1 = X1;
X2 = matricizing(X,2);
X3 = matricizing(X,3);
% Initialize PF loads with resutls from MCR-tri-linearity
A = S;
B_ = B;
C_ = C;

SSE = zeros(1,2);
SSE(1) = 2*SST;
SSE(2) = SST;
fitdif = SSE(1)-SSE(2);
it_pf = 0;
if fit_pf == 1
    maxit = 2;
else
    maxit = 5000;
end

if pccf_set == 0

while fitdif(end) > convcrit & it_pf < maxit
    it_pf = it_pf+1;
    BC = krb(C,B);
    BCtBC = BC'*BC;
    BCtX = BC'*X1';
    if      constr(1) == 0
        A = BCtX'*pinv(BCtBC);
    elseif  constr(1) == 1
        try
            A = fcnnls(BC, X1, BCtBC, BCtX)';
            for k =1:ncomp
                if sum(A(:,k)) == 0
                    A(:,k) = rand(1,size(A,1));
                end
            end
        catch
            BC = randn(size(BC));
            BCtBC = BC'*BC;
            BCtX = BC'*X1';
            A = fcnnls(BC, X1, BCtBC, BCtX)';
        end
    end
    A = A./vecnorm(A);
    AC = krb(C,A);
    ACtAC = AC'*AC;
    ACtXt = AC'*X2';
    if      constr(2) == 0
        B = ACtXt'*pinv(ACtAC);
    elseif  constr(2) == 1
        try
            B = fcnnls(AC,X2,ACtAC,ACtXt)';
        catch
            B = randn(size(B));
            for k = 1:ncomp
                B(:,k) = B(:,k)*1/norm(B(:,k),'fro');
            end
            ACtAC = AC'*AC;
            ACtXt = X2*AC;
            B = fcnnls(AC,X2,ACtAC,ACtXt)';
        end
        B = B./vecnorm(B);
    end
    AB = krb(B,A);
    ABtAB = AB'*AB;
    ABtXt = X3*AB;
    if      constr(2) == 0
        C = ABtXt'*pinv(ABtAB);
    elseif  constr(2) == 1
        try
            C = fcnnls(AB,X3,ABtAB,ABtXt')';
        catch
            C = randn(size(C));
            for k = 1:ncomp
                C(:,k) = C(:,k)*1/norm(C(:,k),'fro');
            end
            ABtAB = AB'*AB;
            ABtXt = AB'*X3';
            C = fcnnls(AB,X3,ABtAB,ABtXt)';
        end
    end
    SSE(2) = SSE(1);
    SSE(1) = SST+sum(sum((BC'*BC) .*(A'*A)))-2*sum((A'*X1)' .* BC,'all');
    all_fit(it_pf) = (1-SSE(1)/SST)*100;
    fitdif(it_pf) = abs((SSE(1)-SSE(2))/SSE(2));
    all_err(it_pf) = SSE(1);

end

elseif pccf_set == 1
    constr(1) = 0;
    constr(2) = 0;
    while fitdif(end) > convcrit & it_pf < maxit
    it_pf = it_pf+1;
    BC = krbc(C,B,pccf_temp);
    BCtBC = BC'*BC;
    BCtX = BC'*X1';
    if      constr(1) == 0
        A = BCtX'*pinv(BCtBC);
    elseif  constr(1) == 1
        try
            A = fcnnls(BC, X1, BCtBC, BCtX)';
            for k =1:ncomp
                if sum(A(:,k)) == 0
                    A(:,k) = rand(1,size(A,1));
                end
            end
        catch
            BC = randn(size(BC));
            BCtBC = BC'*BC;
            BCtX = BC'*X1';
            A = fcnnls(BC, X1, BCtBC, BCtX)';
        end
    end
    A = A./vecnorm(A);
    AC = krbc(C,A,pccf_temp);
    ACtAC = AC'*AC;
    ACtXt = AC'*X2';
    if      constr(2) == 0
        B = ACtXt'*pinv(ACtAC);
    elseif  constr(2) == 1
        try
            B = fcnnls(AC,X2,ACtAC,ACtXt)';
        catch
            B = randn(size(B));
            for k = 1:ncomp
                B(:,k) = B(:,k)*1/norm(B(:,k),'fro');
            end
            ACtAC = AC'*AC;
            ACtXt = X2*AC;
            B = fcnnls(AC,X2,ACtAC,ACtXt)';
        end
    end
    B = B./vecnorm(B);
    AB = krbc(B,A,pccf_temp);
    ABtAB = AB'*AB;
    ABtXt = X3*AB;
    if      constr(2) == 0
        C = ABtXt*pinv(ABtAB);
    elseif  constr(2) == 1
        try
            C = fcnnls(AB,X3,ABtAB,ABtXt')';
        catch
            C = randn(size(C));
            for k = 1:ncomp
                C(:,k) = C(:,k)*1/norm(C(:,k),'fro');
            end
            ABtAB = AB'*AB;
            ABtXt = AB'*X3';
            C = fcnnls(AB,X3,ABtAB,ABtXt)';
        end
    end
    BC = krbc(C,B,pccf_temp);
    SSE(2) = SSE(1);
    SSE(1) = SST+sum(sum((BC'*BC) .*(A'*A)))-2*sum((A'*X1)' .* BC,'all');
    all_fit(it_pf) = (1-SSE(1)/SST)*100;
    fitdif(it_pf) = abs((SSE(1)-SSE(2))/SSE(2));
    all_err(it_pf) = SSE(1);
end
end

if fit_pf == 2
    pfm.detail.all_err = all_err;   
    pfm.detail.all_fit = all_fit; 
    pfm.detail.all_fitdif = fitdif;
    pfm.detail.iterations = it_pf;
end

end

function [K, Pset] = fcnnls(C, A, CtC, CtA)
% NNLS using normal equations and the fast combinatorial strategy
%
% I/O: [K, Pset] = fcnnls(C, A);
% K = fcnnls(C, A);
%
% C is the nObs x lVar coefficient matrix
% A is the nObs x pRHS matrix of observations
% K is the lVar x pRHS solution matrix
% Pset is the lVar x pRHS passive set logical array
%
% Pset: set of passive sets, one for each column
% Fset: set of column indices for solutions that have not yet converged
% Hset: set of column indices for currently infeasible solutions
% Jset: working set of column indices for currently optimal solutions
%
% Implementation is based on [1] with bugfixes, direct passing of sufficient stats,
% and preserving the active set over function calls.
%
% [1] Van Benthem, M. H., & Keenan, M. R. (2004). Fast algorithm for the
%   solution of large‐scale non‐negativity‐constrained least squares problems.
%   Journal of Chemometrics: A Journal of the Chemometrics Society, 18(10), 441-450.


% Check the input arguments for consistency and initialize
if nargin == 2
    error(nargchk(2,2,nargin))
    [nObs, lVar] = size(C);

    if size(A,1)~= nObs, error('C and A have imcompatible sizes'), end
    if size(C,1) == size(C,2)
        %         warning('A square matrix "C" was input, ensure this is on purpose.')
    end
    pRHS = size(A,2);
    % Precompute parts of pseudoinverse
    CtC = C'*C; CtA = C'*A;
else
    [lVar,pRHS] = size(CtA);

end

if nargin == 2 && size(C,1) == size(C,2)
    warning('fcnnls - The coefficient matrix (C) was square - is this true or are you passing C''C?')
end

W = zeros(lVar, pRHS);
iter = 0;
maxiter = 6*lVar;


% Obtain the initial feasible solution and corresponding passive set
K = cssls(CtC, CtA);
Pset=K>0;
K(~Pset) = 0;
D=K;
Fset = find(~all(Pset));

% Active set algorithm for NNLS main loop
iter_outer = 1;
while ~isempty(Fset) && iter_outer < maxiter
    iter_outer = iter_outer + 1;
    % Solve for the passive variables (uses subroutine below)
    K(:,Fset) = cssls(CtC, CtA(:,Fset), Pset(:,Fset));
    % Find any infeasible solutions
    %     Hset = Fset(find(any(K(:,Fset) < 0)));
    Hset = Fset((any(K(:,Fset) < 0)));

    % Make infeasible solutions feasible (standard NNLS inner loop)
    if ~isempty(Hset)
        nHset = length(Hset);
        alpha = zeros(lVar, nHset);

        while ~isempty(Hset) && (iter < maxiter)
            iter = iter + 1;
            alpha(:,1:nHset) = Inf;
            % Find indices of negative variables in passive set
            [i, j] = find(Pset(:,Hset) & (K(:,Hset) < 0));
            hIdx = sub2ind([lVar nHset], i, j);
            %             if length(i) ~= length(j)
            %                 keyboard
            %             end
            %             negIdx = sub2ind(size(K), i, Hset(j)'); % org
            negIdx = sub2ind(size(K), i, reshape(Hset(j),size(i)));  % jlh mod
            alpha(hIdx) = D(negIdx)./(D(negIdx) - K(negIdx));
            [alphaMin,minIdx] = min(alpha(:,1:nHset));
            alpha(:,1:nHset) = repmat(alphaMin, lVar, 1);
            D(:,Hset) = D(:,Hset)-alpha(:,1:nHset).*(D(:,Hset)-K(:,Hset));
            idx2zero = sub2ind(size(D), minIdx, Hset);
            D(idx2zero) = 0;
            Pset(idx2zero) = 0;
            K(:, Hset) = cssls(CtC, CtA(:,Hset), Pset(:,Hset));
            Hset = find(any(K < 0));
            nHset = length(Hset);
        end
    end
    % Make sure the solution has converged
    %if iter == maxiter, warning('Maximum number iterations exceeded'), end
    % Check solutions for optimality
    W(:,Fset) = CtA(:,Fset)-CtC*K(:,Fset);
    Jset = find(all(~Pset(:,Fset).*W(:,Fset) <= 0));
    Fset = setdiff(Fset, Fset(Jset));
    % For non-optimal solutions, add the appropriate variable to Pset
    if ~isempty(Fset)
        [mx, mxidx] = max(~Pset(:,Fset).*W(:,Fset));
        Pset(sub2ind([lVar pRHS], mxidx, Fset)) = 1;
        D(:,Fset) = K(:,Fset);
    end
end
end
% ****************************** Subroutine****************************
function [K] = cssls(CtC, CtA, Pset)
% Solve the set of equations CtA = CtC*K for the variables in set Pset
% using the fast combinatorial approach
K = zeros(size(CtA));
if (nargin == 2) || isempty(Pset) || all(Pset(:))
    K = CtC\CtA; % Not advisable if matrix is close to singular or badly scaled
    %     K = pinv(CtC)*CtA;
else
    [lVar, pRHS] = size(Pset);
    codedPset = 2.^(lVar-1:-1:0)*Pset;
    [sortedPset, sortedEset] = sort(codedPset);
    breaks = diff(sortedPset);
    breakIdx = [0 find(breaks) pRHS];
    for k = 1:length(breakIdx)-1
        cols2solve = sortedEset(breakIdx(k)+1:breakIdx(k+1));
        vars = Pset(:,sortedEset(breakIdx(k)+1));
        K(vars,cols2solve) = CtC(vars,vars)\CtA(vars,cols2solve);
        %         K(vars,cols2solve) = pinv(CtC(vars,vars))*CtA(vars,cols2solve);
    end
end
end