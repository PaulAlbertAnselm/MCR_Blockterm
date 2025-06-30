
function model = MCRBTD(X,Fac,options)
% I/O
%
% I:
% X(elutiontimes x samples x spectral chanels) = Low rank chromatographic data set
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
% options.classify  :   1 = Runs a DeepNeuralNetwork to classify
%                           elutionprofiles as '0=weird, 1=peak, 2=cutoff, 3=baseline';
% options.PeakshapeCorrection : If set to vals > 1, tri-linearity
%                               constraint is relaxed. Alternatively,
%                               "sist_threshold" parameter can be used to adaptively find
%                               the number of singular values used in the
%                               tri-linearity constraint.
% options.sist_threshold :  Threshold value for the variance explained that needs
%                           to be captured by the leading "n" singular values.
% options.compression : makes sense if size(X,i)*size(X,j) << size(X,k)
% options.compression.do : 0 = no compression, 1 = compression
% options.compression.basis: orthogonal basis for compression
%
%O:
% model.spectra                      = (Fac x spectral chanels)
% model.elutionprofiles              = (elutiontimes x Fac x samples)
% model.scores                       = (Fac x samples)
% model.detail.fit.X_sum_sq          = Total Sum of Squares;
% model.detail.fit.res_sum_sq        = Residual Sum of Squares;
% model.detail.fit.PercVar           = Explained Variance [%];
% model.detail.fit.fitdif            = Final difference in Fit;
% model.detail.iterations            = Number of Iterations
% model.detail.pccf_all              = singular values used in the
%                                      tri-linearity constraint for each
%                                      factor
% model.detail.time                  = Computation time;


%% Unfold X
X1 = X;
mgc    = size(X1);

X1 = reshape(X1,[mgc(1)*mgc(2) mgc(3)]);
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
        init    = 0;
    end
    maxit       = options.MaxIter;
    if isfield(options,'Constr')
        constr      = options.Constr;
    else
        constr = [1 1];
    end
    convcrit    = options.ConvCrit;
    if isfield(options,'PeakshapeCorrection')
        pccf        = options.PeakshapeCorrection;
        pccf_set    = 0;
    else
        pccf = 1;
        pccf_set    = 1;
        param_sist = options.sist_threshold;
    end
%     if isfield(options,'compression')
%         compression = options.compression.do;
%         basis_f     = options.compression.basis;
%     else
%         compression = 0;
%     end

    if ~isfield(options,'classify');
        options.classify = 1;
    end
end

%% Initialization of S or C
% pure sample / spectra

if init == 0
    CB = randn(size(X1,1),ncomp);
elseif init == 2
    S = options.InitLoads;
    StS = S'*S;
    StXt = S'*X1';
    CB = fcnnls([],[],StS,StXt)';
end

SST = sum(X1.^2,'all');
SSE = zeros(1,2);
SSE(1) = 2*SST;
SSE(2) = SST;
fitdif = SSE(1)-SSE(2);
fit = [];
iter = [];
tictic = 0;

%% calculating mcr solution

it = 0;
while fitdif > convcrit & it < maxit
    it = it+1;
    CtC = CB'*CB;
    CtX = CB'*X1;
    if      constr(1) == 0
        S = CtX'*pinv(CtC);
        S(S<0) = 0;
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
            CtX = CB'*X1;
            S = fcnnls(CB, X1, CtC, CtX)';
        end

        for k = 1:ncomp
            S(:,k) = S(:,k)*1/norm(S(:,k),'fro');
        end

    end
    StS = S'*S;
    StXt = S'*X1';
    if  constr(2) == 0
        CB = StXt'*pinv(StS);
        CB(CB<0) = 0;
    elseif  constr(2) == 1
        try
            CB = fcnnls([],[],StS,StXt)';
        catch
            S = randn(size(S));
            for k = 1:ncomp
                S(:,k) = S(:,k)*1/norm(S(:,k),'fro');
            end
            StS = S'*S;
            StXt = S'*X1';
            CB = fcnnls([],[],StS,StXt)';
        end
    end


    %% Shift invariant trilinearity on CB
    copt1 = [];
    Bft = [];
    Cft = [];
    Bneu = {};
    Cneu = [];

    for j = 1:ncomp

        %[y,ya,p]= shiftmap(reshape(CB(:,j),mgc(1),mgc(2)));%shiftmap(copt1');
        %making trilinearity for shifted data
        y= reshape(CB(:,j),mgc(1),mgc(2));%shiftmap(copt1');
        [U,T,V] = svd(y,'econ');

        if pccf_set == 1
            if fitdif(end) < 100*convcrit | tictic > 0
                tictic = 1;
                ttemp = diag(T);
                ixi = 1;
                while (1-sum(ttemp(1:ixi).^2)/sum(ttemp.^2)) > param_sist(1)
                    ixi = ixi+1;
                end
                if ixi < param_sist(2)
                    pccf = ixi;
                else
                    pccf = param_sist(2);
                end
            else
                pccf = 1;
            end
            
            pccf_all(j,it) = pccf;
        end

        pccf_all(j,it) = pccf;

        %storing results as elutionprofiles and relative concentrations
        Bft = U(:,1:pccf)*T(1:pccf,1:pccf);
        Cft = V(:,1:pccf)';
        %shifting back
        %Bneu{1,j} = shiftmap(Bft*Cft,ya);
        Bneu{1,j} = Bft*Cft;
        Bneu{1,j} = Bneu{1,j}(1:mgc(1),:);
    end
    %reshaping to get CB back
    BB = zeros(mgc(1),mgc(2), ncomp);
    for i = 1:(ncomp)
        BB(:,:,i) = Bneu{1,i};
    end

    CBneu = reshape(BB,mgc(1)*mgc(2),ncomp);
    CB = CBneu;


    %evaluating loss function
    SSE(2) = SSE(1);
    SSE(1) = SST+sum(sum((CB'*CB) .*(S'*S)))-2*sum((X1*S) .* CB,'all');
    fit(it) = (1-SSE(1)/SST)*100;
    fitdif(it) = abs(SSE(2)-SSE(1))/SSE(2);

end


CB = reshape(CB,mgc(1),mgc(2),ncomp);
model.spectra   = S;
for i = 1:size(CB ,3)
    for ii = 1:size(CB ,2)
        Cneu(ii,i) = norm(squeeze(CB(:,ii,i)),'fro');
        CB(:,ii,i) = CB(:,ii,i)./Cneu(ii,i);
    end
end
model.elutionprofiles  = CB;
model.elutionprofiles  = permute(model.elutionprofiles,[1 3 2]);
model.scores   = Cneu;
model.detail.fit.X_sum_sq        = SST;
model.detail.fit.res_sum_sq      = SSE(1);
model.detail.fit.PercVar         = fit(end);
model.detail.fit.fitdif          = fitdif;
model.detail.iterations    = it;
model.detail.time          = toc();
model.detail.pccf_all      = pccf_all;

    function [y,ya,p] = shiftmap(x,p)
        %
        %  For calculating |fft| and phase map:
        %  INPUTS:
        %     x = MxN operates on the columns
        %  OPTIONAL INPUT:
        %     p = nonnegative scalar length of padding for fft
        %         if not included p = 2.^nextpow2(M);
        %  OUTPUTS:
        %     y = |fft(x)|
        %    ya = phase map of FFT: fft(x)./|fft(x)|
        %     p = p = 2.^nextpow2(n); [used when (p) not input].
        %
        %I/O: [y,ya,p] = shiftmap(x,p);
        %
        %  For calculating shifted profiles from |fft| and phase map:
        %  INPUTS:
        %     y = |fft(xhat)|, e.g., Xhat from a 1 PC PCA model of |FFT(x)|
        %    ya = phase map of FFT: fft(x)./|fft(x)|
        %  OUTPUTS:
        %     x = real(ifft(y.*ya))
        %
        %I/O: [x] = shiftmap(y,ya);

        yr    = 1e-6;             %regularization for angle map

        m     = size(x);

        if nargin<2||isscalar(p)  %calculate |FFT| and angle map
            if nargin<2
                p   = 2.^nextpow2(m(1));
            end

            % FFT
            %             if isdataset(x)
            %                 z   = fft(x.data,p);
            %             else
            %     z   = fft(x,p);       %assume x is class double
            z = fft(x);
            %             end
            % |FFT| and angle map
            y     = abs(z);
            a     = y;
            a(y<yr)   = yr;
            ya    = z./a;
        else %p = ya (phase map)

            % IFFT
            %             if isdataset(x)
            %                 y   = real(ifft(x.data.*p));
            %             else
            y   = real(ifft(x.*p));   %assume x is class double
            %            end
            ya    = [];
            p     = [];


        end
    end



    function [X] = makeXfromABC(A,B,C)

        na = size(A,1);
        nb = size(B,1);
        nc = size(C,1);
        ncomp = size(C,2);

        X = zeros(na,nb,nc);
        for i = 1:nc
            Di = zeros(ncomp,ncomp);
            for ii = 1:ncomp
                Di(ii,ii) = C(i,ii);
            end
            X(:,:,i) = A*Di*B(:,:,i)';
        end

        X = permute(X,[3,2,1]);
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
%                 Pset(sub2ind([lVar pRHS], mxidx, Fset)) = 1;
                D(:,Fset) = K(:,Fset);
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
    end
end
