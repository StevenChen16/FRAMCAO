classdef MCAOAnalyzerMatlab < handle
    properties
        alpha   % Weight for fractional derivative
        beta    % Weight for coupling term
        eta     % Weight for global term
        gamma   % Fractional derivative order
        tau     % Time window
    end
    
    methods
        function obj = MCAOAnalyzerMatlab(alpha, beta, eta, gamma, tau)
            if nargin < 5
                alpha = 0.1;
                beta = 0.2;
                eta = 0.15;
                gamma = 0.5;
                tau = 1.0;
            end
            obj.alpha = alpha;
            obj.beta = beta;
            obj.eta = eta;
            obj.gamma = gamma;
            obj.tau = tau;
        end
        
        function fd = fractionalDerivative(obj, f, dt)
            n = length(f);
            fd = zeros(size(f));
            
            for i = 2:n
                sum_term = 0;
                for j = 0:i-1
                    coeff = (-1)^j * gamma(obj.gamma + 1) / ...
                        (gamma(j + 1) * gamma(obj.gamma - j + 1));
                    if i-j >= 1
                        sum_term = sum_term + coeff * f(i-j);
                    end
                end
                fd(i) = sum_term / (dt^obj.gamma);
            end
        end
        
        function s = couplingFunction(obj, x, y)
            phi = 1.0;
            s = tanh(phi * (x - y)) .* tanh(abs(x) + abs(y));
        end
        
        function w = dynamicWeights(obj, f)
            n = length(f);
            w = zeros(n);
            
            for i = 1:n
                for j = 1:n
                    if i ~= j
                        d_ij = abs(f(i) - f(j));
                        w(i,j) = exp(-d_ij);
                    end
                end
            end
            
            % Normalize weights
            row_sums = sum(w, 2);
            w = w ./ row_sums;
        end
        
        function analyzeStability(obj, n_points)
            % Parameter space analysis for stability
            if nargin < 2
                n_points = 20;
            end
            
            alpha_range = linspace(0, 0.5, n_points);
            beta_range = linspace(0, 0.5, n_points);
            stability_map = zeros(n_points, n_points);
            
            t = linspace(0, 2*pi, 50);
            f_init = sin(t);
            
            for i = 1:n_points
                for j = 1:n_points
                    obj.alpha = alpha_range(i);
                    obj.beta = beta_range(j);
                    
                    % Run system and check stability
                    traj = obj.analyzeConvergence(f_init);
                    stability_map(i,j) = obj.computeStabilityMetric(traj);
                end
            end
            
            % Plot stability map
            figure;
            imagesc(alpha_range, beta_range, stability_map);
            colorbar;
            xlabel('Alpha');
            ylabel('Beta');
            title('Stability Analysis in Parameter Space');
            colormap('jet');
        end
        
        function metric = computeStabilityMetric(obj, trajectory)
            % Compute stability metric based on trajectory variation
            n_steps = size(trajectory, 1);
            last_steps = trajectory(floor(n_steps/2):end, :);
            metric = std(std(last_steps));
        end
        
        function traj = analyzeConvergence(obj, f_init, n_steps, dt)
            if nargin < 4
                dt = 0.01;
            end
            if nargin < 3
                n_steps = 100;
            end
            
            f = f_init;
            traj = zeros(n_steps+1, length(f_init));
            traj(1,:) = f;
            
            for step = 1:n_steps
                delta = obj.mcaoStep(f, dt);
                f = f + delta;
                traj(step+1,:) = f;
            end
        end
        
        function delta = mcaoStep(obj, f, dt)
            % Compute single MCAO step
            fd_term = obj.alpha * obj.fractionalDerivative(f, dt);
            
            % Coupling term
            w = obj.dynamicWeights(f);
            n = length(f);
            coupling_term = zeros(size(f));
            for i = 1:n
                for j = 1:n
                    if i ~= j
                        coupling_term(i) = coupling_term(i) + ...
                            w(i,j) * obj.couplingFunction(f(i), f(j));
                    end
                end
            end
            coupling_term = obj.beta * coupling_term;
            
            % Global term (simplified)
            g_term = obj.eta * mean(f);
            
            delta = fd_term + coupling_term + g_term;
        end
        
        function analyzeBoundedness(obj, n_samples)
            if nargin < 2
                n_samples = 1000;
            end
            
            % Generate random initial conditions
            f_samples = randn(n_samples, 50);  % 50 points per sample
            max_vals = zeros(n_samples, 1);
            
            for i = 1:n_samples
                traj = obj.analyzeConvergence(f_samples(i,:));
                max_vals(i) = max(max(abs(traj)));
            end
            
            % Analyze distribution of maximum values
            figure;
            histogram(max_vals, 50);
            xlabel('Maximum Absolute Value');
            ylabel('Frequency');
            title('Boundedness Analysis: Distribution of Maximum Values');
            
            % Add statistical information
            hold on;
            xline(mean(max_vals), 'r-', 'Mean', 'LineWidth', 2);
            xline(mean(max_vals) + 2*std(max_vals), 'g--', 'Mean + 2σ');
            xline(mean(max_vals) - 2*std(max_vals), 'g--', 'Mean - 2σ');
            
            fprintf('Boundedness Statistics:\n');
            fprintf('Mean max value: %.4f\n', mean(max_vals));
            fprintf('Std of max values: %.4f\n', std(max_vals));
            fprintf('Max observed value: %.4f\n', max(max_vals));
        end
        
        function analyzeExistenceUniqueness(obj, n_trials)
            if nargin < 2
                n_trials = 10;
            end
            
            % Test convergence from different initial conditions
            t = linspace(0, 2*pi, 50);
            final_states = zeros(n_trials, length(t));
            
            figure;
            hold on;
            for i = 1:n_trials
                f_init = sin(t + 2*pi*rand()) + 0.5*randn(size(t));
                traj = obj.analyzeConvergence(f_init);
                final_states(i,:) = traj(end,:);
                
                % Plot trajectories
                plot(t, traj(end,:), 'LineWidth', 1);
            end
            xlabel('Space');
            ylabel('Final State');
            title('Existence/Uniqueness Analysis: Convergence from Different Initial Conditions');
            
            % Analyze clustering of final states
            D = pdist(final_states);
            Z = linkage(D);
            
            figure;
            dendrogram(Z);
            title('Clustering of Final States');
            ylabel('Distance');
        end
        
        function analyzeEigenstructure(obj, f)
            % Compute Jacobian numerically
            n = length(f);
            J = zeros(n);
            eps = 1e-6;
            
            for i = 1:n
                f_perturbed = f;
                f_perturbed(i) = f_perturbed(i) + eps;
                delta_plus = obj.mcaoStep(f_perturbed, 0.01);
                
                f_perturbed = f;
                f_perturbed(i) = f_perturbed(i) - eps;
                delta_minus = obj.mcaoStep(f_perturbed, 0.01);
                
                J(:,i) = (delta_plus - delta_minus)/(2*eps);
            end
            
            % Analyze eigenvalues
            [V, D] = eig(J);
            eig_vals = diag(D);
            
            % Plot eigenvalue spectrum
            figure;
            subplot(1,2,1);
            plot(real(eig_vals), imag(eig_vals), 'ko');
            grid on;
            xlabel('Real Part');
            ylabel('Imaginary Part');
            title('Eigenvalue Spectrum');
            
            % Plot leading eigenvectors
            subplot(1,2,2);
            plot(abs(V(:,1:min(3,n))));
            xlabel('Component');
            ylabel('Magnitude');
            title('Leading Eigenvectors');
            legend('1st', '2nd', '3rd');
        end
        
        function analyzeLyapunovSpectrum(obj, f_init, n_steps)
            if nargin < 3
                n_steps = 1000;
            end
            
            n = length(f_init);
            n_exponents = min(3, n);  % Compute first 3 Lyapunov exponents
            
            % Initialize orthonormal perturbation vectors
            Q = eye(n, n_exponents);
            lyap_sum = zeros(n_exponents, 1);
            
            f = f_init;
            dt = 0.01;
            
            % Main loop for Lyapunov spectrum computation
            for step = 1:n_steps
                % Evolve reference trajectory
                f = f + obj.mcaoStep(f, dt);
                
                % Evolve perturbation vectors
                for i = 1:n_exponents
                    f_pert = f + eps * Q(:,i);
                    delta_pert = obj.mcaoStep(f_pert, dt) - obj.mcaoStep(f, dt);
                    Q(:,i) = delta_pert / norm(delta_pert);
                    
                    % Accumulate Lyapunov exponents
                    lyap_sum(i) = lyap_sum(i) + log(norm(delta_pert)/eps);
                end
                
                % Reorthogonalize perturbation vectors
                Q = modifiedGramSchmidt(Q);
            end
            
            % Compute Lyapunov exponents
            lyap_exponents = lyap_sum / (n_steps * dt);
            
            % Display results
            figure;
            bar(lyap_exponents);
            xlabel('Index');
            ylabel('Lyapunov Exponent');
            title('Lyapunov Spectrum Analysis');
            grid on;
            
            fprintf('Lyapunov Exponents:\n');
            for i = 1:n_exponents
                fprintf('λ_%d = %.4f\n', i, lyap_exponents(i));
            end
        end
        
        function analyzeBifurcation(obj, param_range, param_type)
            if nargin < 3
                param_type = 'alpha';
            end
            if nargin < 2
                param_range = linspace(0, 1, 100);
            end
            
            % Initialize
            t = linspace(0, 2*pi, 50);
            f_init = sin(t);
            
            % Store final states for each parameter value
            final_states = zeros(length(param_range), length(t));
            
            % Compute trajectories for each parameter value
            for i = 1:length(param_range)
                % Set parameter
                switch param_type
                    case 'alpha'
                        obj.alpha = param_range(i);
                    case 'beta'
                        obj.beta = param_range(i);
                    case 'gamma'
                        obj.gamma = param_range(i);
                end
                
                % Compute trajectory and store final state
                traj = obj.analyzeConvergence(f_init);
                final_states(i,:) = traj(end,:);
            end
            
            % Plot bifurcation diagram
            figure;
            plot(param_range, final_states, '.k', 'MarkerSize', 1);
            xlabel(sprintf('%s', param_type));
            ylabel('Final State Components');
            title('Bifurcation Analysis');
            grid on;
        end
    end
    
    methods(Static)
        function Q = modifiedGramSchmidt(V)
            % Modified Gram-Schmidt orthogonalization
            [n, m] = size(V);
            Q = zeros(n,m);
            
            for i = 1:m
                q = V(:,i);
                for j = 1:i-1
                    q = q - (Q(:,j)'*q)*Q(:,j);
                end
                Q(:,i) = q/norm(q);
            end
        end
    end
end