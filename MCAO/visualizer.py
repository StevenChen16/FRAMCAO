import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MCAOAnalyzer:
    def __init__(self, alpha=0.1, beta=0.2, eta=0.15, gamma=0.5, tau=1.0):
        self.alpha = alpha  # Weight for fractional derivative
        self.beta = beta    # Weight for coupling term
        self.eta = eta      # Weight for global term
        self.gamma = gamma  # Fractional derivative order
        self.tau = tau      # Time window
        
    def fractional_derivative(self, f, dt):
        """
        Compute fractional derivative using Grünwald-Letnikov method
        """
        n = len(f)
        fd = np.zeros_like(f)
        
        for i in range(1, n):
            sum_term = 0
            for j in range(i + 1):
                coeff = (-1)**j * gamma(self.gamma + 1) / (gamma(j + 1) * gamma(self.gamma - j + 1))
                if i-j >= 0:
                    sum_term += coeff * f[i-j]
            fd[i] = sum_term / (dt**self.gamma)
            
        return fd
    
    def coupling_function(self, x, y):
        """
        Nonlinear coupling function σ(x,y)
        """
        phi = 1.0  # Parameter for difference sensitivity
        psi = lambda z: np.tanh(z)  # Magnitude response function
        
        return np.tanh(phi * (x - y)) * psi(np.abs(x) + np.abs(y))
    
    def dynamic_weights(self, f, t):
        """
        Compute dynamic weights w_ij(t)
        """
        n = len(f)
        w = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    d_ij = np.abs(f[i] - f[j])  # Simple distance metric
                    w[i,j] = np.exp(-d_ij)
                    
        # Normalize weights
        row_sums = w.sum(axis=1)
        w = w / row_sums[:, np.newaxis]
        
        return w
    
    def global_state(self, f, t):
        """
        Compute global state function G(f,t)
        """
        return np.mean(f)  # Simple mean field approximation
    
    def kernel(self, t):
        """
        Exponential decay kernel K(t)
        """
        kappa = 1.0  # Decay rate
        return np.exp(-kappa * t)
    
    def mcao_step(self, f, dt):
        """
        Single step of MCAO operator
        """
        n = len(f)
        result = np.zeros_like(f)
        
        # Compute fractional derivative term
        fd_term = self.alpha * self.fractional_derivative(f, dt)
        
        # Compute coupling term
        w = self.dynamic_weights(f, 0)  # Current time weights
        coupling_term = np.zeros_like(f)
        for i in range(n):
            for j in range(n):
                if i != j:
                    coupling_term[i] += w[i,j] * self.coupling_function(f[i], f[j])
        coupling_term *= self.beta
        
        # Compute global influence term (simplified)
        g_term = self.eta * self.global_state(f, 0)
        
        result = fd_term + coupling_term + g_term
        return result
    
    def analyze_continuity(self, f_base, epsilon=0.01, n_points=100):
        """
        Analyze continuity by perturbing input
        """
        perturbations = np.linspace(-epsilon, epsilon, n_points)
        outputs = []
        
        for p in perturbations:
            f_perturbed = f_base + p
            out = self.mcao_step(f_perturbed, 0.01)
            outputs.append(np.mean(out))
            
        return perturbations, outputs
    
    def analyze_convergence(self, f_init, n_steps=100, dt=0.01):
        """
        Analyze convergence through iteration
        """
        f = f_init.copy()
        trajectory = [f.copy()]
        
        for _ in range(n_steps):
            f = f + self.mcao_step(f, dt)
            trajectory.append(f.copy())
            
        return np.array(trajectory)
    
    def plot_analysis(self, n_points=50):
        """
        Create visualization plots for operator analysis
        """
        # Setup
        t = np.linspace(0, 2*np.pi, n_points)
        f_init = np.sin(t)  # Initial condition
        
        # Plot 1: Continuity Analysis
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        pert, out = self.analyze_continuity(f_init)
        plt.plot(pert, out)
        plt.title('Continuity Analysis')
        plt.xlabel('Perturbation')
        plt.ylabel('Output Change')
        
        # Plot 2: Convergence Analysis
        plt.subplot(132)
        traj = self.analyze_convergence(f_init)
        plt.imshow(traj.T, aspect='auto', cmap='viridis')
        plt.title('Convergence Analysis')
        plt.xlabel('Time Step')
        plt.ylabel('Space')
        plt.colorbar(label='Value')
        
        # Plot 3: Phase Space (for first two components)
        plt.subplot(133)
        plt.plot(traj[:,0], traj[:,1])
        plt.title('Phase Space (2D Projection)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        plt.tight_layout()
        plt.show()

# Example usage and analysis
analyzer = MCAOAnalyzer()
n_points = 50
t = np.linspace(0, 2*np.pi, n_points)
f_init = np.sin(t)

# Run analysis and create plots
analyzer.plot_analysis(n_points)