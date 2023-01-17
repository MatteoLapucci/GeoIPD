import numpy as np
from time import time
from scipy.optimize import minimize
from gurobipy import *


class PDSolver:
    '''Implementation of (inexact) Penalty Decomposition approach from [Kanzow and Lapucci,
    "Inexact Penalty Decomposition Methods for Optimization Problems with Geometric 
    Constraints" (2023)]'''
    
    def __init__(self, problem, tau0=1, tau_gr=1.1, eta=0.8, tau_max=1e8, lambda_max=1e8, outer_eps=1e-5, inner_eps=1e-5, method='gd', multipliers=False, no_eq_mult=False, debug=False):
        
        # starting solution
        self.x = problem.get_x0()
        self.y = np.copy(self.x)
        
        # starting penalty parameter and increase rate
        self.tau = tau0
        self.tau_gr = tau_gr
        
        # problem to be solved 
        self.problem = problem
        
        # tolerance for stopping conditions of inner and outer loop
        self.inner_eps = inner_eps
        self.outer_eps = outer_eps
        
        # algorithm to be used for the x-update step
        self.method = method
        
        # upper bound on penalty parameter for numerical reasons
        self.tau_max = tau_max
        
        # Lagrange multipliers vectors: used or not
        self.multipliers = multipliers
        
        # multipliers for original constraints and constraint x-y=0
        self.lambdas = np.zeros_like(problem.G(self.x)) if multipliers else None
        self.lambdas_eq = np.zeros_like(self.x) if multipliers else None
        
        # upper bound (safeguard) on absolute value of multipliers
        self.lambda_max = lambda_max if multipliers else None
        
        # auxiliary function improvement weight
        self.eta = eta
        
        # enable logging prints
        self.debug = debug
        
        # use multipliers only with original constraints: yes or no
        self.no_eq_mult = no_eq_mult




    def penalty_f(self, x, y):
        '''Penalty function defined according to equation (3.1) [Kanzow and Lapucci,
        "Inexact Penalty Decomposition Methods for Optimization Problems with Geometric 
        Constraints" (2023)]'''
        
        if self.multipliers:
            
            # augmented lagrangian
            return self.problem.f(x) + 0.5*self.tau*(self.problem.dist(x+self.lambdas_eq/self.tau,y)**2+self.problem.dist2_C(self.problem.G(x)+self.lambdas/self.tau))
        else:
            
            # simple quadratic penalty function
            return self.problem.f(x) + 0.5*self.tau*(self.problem.dist(x,y)**2+self.problem.dist2_C(self.problem.G(x)))

    def grad_x_penalty_f(self, x,y):
        '''Gradient w.r.t. x variables of penalty function'''
        
        if self.multipliers:
            
            # augmented lagrangian
            return self.problem.grad_f(x) + self.tau*(x + self.lambdas_eq/self.tau - y)  + 0.5*self.tau*np.matmul(self.problem.jac_G(x).T, self.problem.grad_dist2_C(self.problem.G(x)+self.lambdas/self.tau)).reshape(x.shape)
        else:
            
            # simple quadratic penalty function
            return self.problem.grad_f(x) + self.tau*(x-y)  + 0.5*self.tau*np.matmul(self.problem.jac_G(x).T, self.problem.grad_dist2_C(self.problem.G(x))).reshape(x.shape)

    def unconstrained_solve_penalty(self, x, alpha0=1, grad_tol=1e-5, max_iters=1000, min_step=1e-10, gamma=1e-5, delta=0.5, method='gd'):
        '''call to actual solver for the x-update step'''
        
        if method == 'gd':
            
            # call standard gradient descent, custom implementation
            return self.solve_GD(x, alpha0=alpha0, grad_tol=grad_tol, max_iters=max_iters, min_step=min_step, gamma=gamma, delta=delta)
        
        elif method == 'scipy_qn':
            
            # call BFGS, scipy implementation
            return self.solveScipyQN(x, g_tol=grad_tol, max_iters=max_iters)
        
        elif method=='scipy_lbfgs':
            
            # call L-BFGS, scipy implementation
            return self.solveScipyLBFGS(x, g_tol=grad_tol, max_iters=max_iters)
        
        elif method=='scipy_lbfgs_inexact':
            
            # call L-BFGS, scipy implementation, stopping conditions set to low accuracy
            return self.solveScipyLBFGS(x, g_tol=1e-1, max_iters=10)
        
        elif method=='exact_correlation':
            
            # call closed-form solver for Nearest Low-Rank Correlation Matrix problem
            return self.solveExactCorr(x)
        
        elif method=='exact_correlation_constraint':
            
            # call closed-form solver for Nearest Low-Rank Correlation Matrix problem, explicitly handle diag(X) = (1,...,1)
            return self.solveExactCorrConstr(x)
        
        elif method=='scipy_cg_inexact':
            
            # call conjugate gradient, scipy implementation, stopping conditions set to low accuracy
            return self.solveScipyCG(x, g_tol=1e-1, max_iters=5)
        
        elif method=='scipy_cg':
            
            # call conjugate gradient, scipy implementation, stopping conditions set to relatively high accuracy
            return self.solveScipyCG(x, g_tol=1e-3, max_iters=20)
        
        elif method=='scipy_cg_mid':
            
            # call conjugate gradient, scipy implementation, stopping conditions set to mid accuracy
            return self.solveScipyCG(x, g_tol=5e-2, max_iters=8)
        
        else:
            
            # unkown method
            # FIXME: raise Exception!
            pass




    def solveExactCorr(self,x):
        '''closed-form solver for x-update with Nearest Low-Rank Correlation Matrix problems'''
        
        if self.multipliers:
            x_tmp = (self.problem.C+self.tau*self.y-self.lambdas_eq)/(1+self.tau)
        else:
            x_tmp = (self.problem.C+self.tau*self.y)/(1+self.tau)
        for i in range(x_tmp.shape[0]):
            x_tmp[i,i] = 1
        return x_tmp, 0


    def solveExactCorrConstr(self,x):
        '''closed-form solver for x-update with Nearest Low-Rank Correlation Matrix problems, explicitly handling diag(X) = (1,...,1)'''

        if self.multipliers:
            x_tmp = (self.problem.C+self.tau*self.y-self.lambdas_eq)/(1+self.tau)
            for i in range(x_tmp.shape[0]):
                x_tmp[i,i] = (self.problem.C[i,i]+self.tau*self.y[i,i]-self.lambdas_eq[i,i] + self.tau-self.lambdas[i])/(1+2*self.tau)
        else:
            x_tmp = (self.problem.C+self.tau*self.y)/(1+self.tau)
            for i in range(x_tmp.shape[0]):
                x_tmp[i,i] = (self.problem.C[i,i]+self.tau*self.y[i,i]+self.tau)/(1+2*self.tau)
        return x_tmp, 0

    def solveScipyLBFGS(self,x,g_tol,max_iters):
        '''solve x-update subproblem, starting at x, using scipy implementation of L-BFGS'''
        

        if x.ndim >1:
            
            # variables are matrices, scipy.minimize only handles vectors, so x is temporarily reshaped
            shapes = x.shape
            x = x.flatten()
            
            # objective function of subproblem; internally carries back variables to matrix form
            def objf(x):
                return self.penalty_f(x.reshape(shapes), self.y)
            
            # gradient of subproblem; internally carries back variables to matrix form
            def objg(x):
                return self.grad_x_penalty_f(x.reshape(shapes),self.y).flatten()
            
            # solve subproblem by L-BFGS
            lbfgs = minimize(objf, x, jac=objg, method="L-BFGS-B", options={"iprint": -1, "maxcor": 10, "gtol": g_tol, "ftol": 10*np.finfo(float).eps, "maxiter": max_iters, "maxls": 20})
            
            #return solution in matrix form
            return lbfgs.x.reshape(shapes), 0
        
        else:
            # variables are one-dimensional arrays
            
            # objective function of subproblem;
            def objf(x):
                return self.penalty_f(x, self.y)
            
            # gradient function of subproblem;
            def objg(x):
                return self.grad_x_penalty_f(x,self.y)
            
            # solve subproblem by L-BFGS and return the solution
            lbfgs = minimize(objf, x, jac=objg, method="L-BFGS-B", options={"iprint": -1, "maxcor": 10, "gtol": g_tol, "ftol": 10*np.finfo(float).eps, "maxiter": max_iters, "maxls": 20})
            return lbfgs.x, 0

    def solveScipyCG(self,x,g_tol,max_iters):
        '''solve x-update subproblem, starting at x, using scipy implementation of Conjugate Gradient method'''
        
        if x.ndim>1:
            
            # variables are matrices, scipy.minimize only handles vectors, so x is temporarily reshaped
            shapes = x.shape
            x= x.flatten()
            
            # objective function of subproblem; internally carries back variables to matrix form
            def objf(x):
                return self.penalty_f(x.reshape(shapes), self.y)
            
            # gradient of subproblem; internally carries back variables to matrix form
            def objg(x):
                return self.grad_x_penalty_f(x.reshape(shapes), self.y).flatten()
            
            #solve subproblem by Conjugate Gradient
            cg = minimize(objf,x,jac=objg, method='CG', options={"disp": False, "gtol": g_tol, "maxiter": max_iters})
            
            #return solution in matrix form
            return cg.x.reshape(shapes), 0
        
        else:
            # variables are one-dimensional arrays
            
            # objective function of subproblem;
            def objf(x):
                return self.penalty_f(x, self.y)
            
            # gradient function of subproblem;
            def objg(x):
                return self.grad_x_penalty_f(x,self.y)
            
            #solve subproblem by Conjugate Gradient and return the solution
            cg = minimize(objf,x,jac=objg, method='CG', options={"disp": False, "gtol": g_tol, "maxiter": max_iters})
            return cg.x, 0


    def solveScipyQN(self, x, g_tol, max_iters):
        '''solve x-update subproblem, starting at x, using scipy implementation of BFGS method'''
        
        if x.ndim > 1:
            # variables are matrices, scipy.minimize only handles vectors, so x is temporarily reshaped
            
            shapes = x.shape
            x = x.flatten()
            
            # objective function of subproblem; internally carries back variables to matrix form
            def objf(x):
                return self.penalty_f(x.reshape(shapes), self.y)
            
            # gradient of subproblem; internally carries back variables to matrix form
            def objg(x):
                return self.grad_x_penalty_f(x.reshape(shapes),self.y).flatten()
            
            # solve subproblem by BFGS
            bfgs = minimize(objf, x, jac=objg, method="BFGS", options={"disp": False, "gtol": g_tol, "maxiter": max_iters})
            
            #return solution in matrix form
            return bfgs.x.reshape(shapes), 0
        
        else:
            # variables are one-dimensional arrays
            
            # objective function of subproblem;
            def objf(x):
                return self.penalty_f(x, self.y)
            
            # gradient function of subproblem;
            def objg(x):
                return self.grad_x_penalty_f(x,self.y)
            
            # solve subproblem by BFGS and return the solution
            bfgs = minimize(objf, x, jac=objg, method="BFGS", options={"disp": False, "gtol": g_tol, "maxiter": max_iters})
            return bfgs.x, 0



    


    def solve_GD(self, x, alpha0=1, grad_tol=1e-5, max_iters=1000, min_step=1e-10, gamma=1e-5, delta=0.5):
        '''Custom implementation of gradient descent method with Armijo line search, to solve x-update subproblem starting at x'''
        
        # initialize starting solution and iteration counter
        xk = np.copy(x)
        n_iters=0
        
        # gradient descent loop
        while True:
            
            # compute objecitve function and gradient at current solution
            f, g = self.penalty_f(xk, self.y), self.grad_x_penalty_f(xk,self.y)
            
            # compute gradient norm; if lower than tolerance or the max number of iterations 
            # was reached, stop
            g_norm = self.problem.norm(g)
            if g_norm < grad_tol or n_iters >= max_iters:
                break
            
            # initialize step for line search
            alpha = alpha0
            
            # line search stops when Armijo condition is satisfied or the stepsize is too small
            while(self.penalty_f(xk-alpha*g, self.y) > f - gamma*alpha*g_norm**2 and alpha>min_step):
                
                # line search step: decrease stepsize by factor delta 
                alpha *= delta
                
            # take step along negative gradients with stepsize from line search
            xk -= alpha*g
            
            #update iterations counter
            n_iters += 1

        # loop stopped, return solution
        return xk, {'iters': n_iters, "f": f, 'g_norm': g_norm}

    def solve(self):
        '''Inexact Penalty Decomposition algorithm from [Kanzow and Lapucci,
        "Inexact Penalty Decomposition Methods for Optimization Problems with Geometric 
        Constraints" (2023)]'''
        
        # start measuring runtime and iterations
        tic = time()
        iteration = 1
        
        # initialization of auxiliary function values
        V_1 = np.inf
        V = np.inf
        
        # counter for inner iterations
        inner_iters = 0

        # initialization of time counter for specific parts of the algorithm
        time_mult_update = 0
        time_x_step = 0
        time_y_step = 0
        time_f_comp = 0

        # Outer loop
        while True:
            ticc = time()
            
            # compute value of penalty function with current penalty parameter (which has just been updated)
            prev_obj = self.penalty_f(self.x,self.y)
            
            time_f_comp += time() - ticc
            
            #Inner loop
            while True:
                inner_iters += 1
                ticc = time()
                
                # carry out x-update step using specified method
                self.x, _ =  self.unconstrained_solve_penalty(self.x,method=self.method)
                
                time_x_step += time()-ticc
                ticc = time()
                
                # carry out y update step in closed form
                if self.multipliers:
                    self.y = self.problem.proj_D(self.x+self.lambdas_eq/self.tau)
                else:
                    self.y = self.problem.proj_D(self.x)
                    
                time_y_step += time()-ticc
                ticc = time()
                
                # compute value of penalty function at the new solution
                obj = self.penalty_f(self.x, self.y)
                
                
                time_f_comp += time()-ticc
                
                # stop inner loop if decrease of penalty function is under given threshold
                if prev_obj-obj < self.inner_eps:
                    break
                
                # update latest value observed for penalty function
                prev_obj = obj
            
            # log for debugging purposes at each outer iteration
            if self.debug:
                print("Iter {}, tau {:.4f}".format(iteration, self.tau))
                print("\t -> f(y) {:.5f}, ||x-y||: {}, dist_C: {}".format(self.problem.f(self.y), self.problem.dist(self.x,self.y), self.problem.dist2_C(self.problem.G(self.x))**(1/2)))
                
            # increase outer iterations counter
            iteration += 1

            # if current solution is (approximately) feasible, stop the outer loop 
            if self.problem.dist(self.x,self.y) < self.outer_eps and self.problem.dist2_C(self.problem.G(self.x))**(1/2) < self.outer_eps:
                break
            
            ticc = time()
            
            # Lagrange multipliers update by Hestenes-Powell-Rockafellar rule with safeguarding 
            if self.multipliers:
                
                # update of multipliers based on current constraints violations and penalty parameter
                self.lambdas = 0.5*self.tau*self.problem.grad_dist2_C(self.problem.G(self.x)+self.lambdas/self.tau)
                self.lambdas_eq = self.lambdas_eq + self.tau*(self.x-self.y)
                
                # Apply safeguarding rule to all multipliers
                # XXX maybe this can be done in matrix form with numpy?
                for i in range(len(self.lambdas)):
                    self.lambdas[i] = max(-self.lambda_max,min(self.lambdas[i], self.lambda_max))
                shape_lambdas_eq = self.lambdas_eq.shape
                self.lambdas_eq = self.lambdas_eq.flatten()
                for i in range(self.lambdas_eq.size):
                    self.lambdas_eq[i] = max(-self.lambda_max, min(self.lambdas_eq[i],self.lambda_max))
                self.lambdas_eq = self.lambdas_eq.reshape(shape_lambdas_eq)
                
                # Compute auxiliary function for PD version with Lagrange multipliers
                V = self.problem.norm(0.5*self.problem.grad_dist2_C(self.problem.G(self.x)+self.lambdas/self.tau)-self.lambdas/self.tau) + self.problem.norm(self.x+self.lambdas_eq/self.tau-self.y)
               
                # XXX if multipliers are not used with x-y=0, implicitly remove them setting to zero
                if self.no_eq_mult:
                    self.lambdas_eq = np.zeros_like(self.lambdas_eq)
                    
            # Increase penalty parameter if: 
            # a) the pure PD scheme without multipliers is used, or
            # b) the auxiliary function did not sufficiently decrease
            if  not self.multipliers or V >= self.eta * V_1:
                self.tau *= self.tau_gr
                self.tau = min(self.tau, self.tau_max)
            
            #update former value of auxiliary function
            V_1 = V
            
            time_mult_update += time()-ticc

        # log for debugging purposes at the end of the algorithm
        if self.debug:
            print('time_x', time_x_step, 'time_y', time_y_step, 'time_mult', time_mult_update, 'time_f_comp', time_f_comp)
            
        # always print solution process final information
        print('f(y*) = {}, violation: {}, time = {}, n_iters = {}, inner_iters = {}'.format(self.problem.f(self.y), self.problem.dist2_C(self.problem.G(self.y))**(1/2), time()-tic, iteration-1, inner_iters))

        # return solution and optimization process information
        return {'f':self.problem.f(self.y),'t':time()-tic,'sol':self.y,'vio':self.problem.dist2_C(self.problem.G(self.y))**(1/2),'iters':iteration-1}



class SpectralSolver:
    '''Implementation of the Spectral Gradient and the Augmented Lagrangian methods 
    from [Jia et al., "An augmented Lagrangian method for optimization problems
    with structured geometric constraints" (2022)]'''
    
    def __init__(self, problem, tau0=1, tau_gr=1.1, sigma = 1e-5, gamma0=1, eta=0.8, lambda_max=1e8, gamma_max=1e12, gamma_inc=2, tau_max=1e8, outer_eps=1e-5, inner_eps=1e-5, m=10, debug=False, max_sgm_iters=5000):
        
        # retrieve starting solution from problem
        self.x = problem.get_x0()
        
        # initialize Lagrange multipliers
        self.lambdas = np.zeros_like(problem.G(self.x))
        
        # set upper bound (safeguard) for multipliers
        self.lambda_max = lambda_max
        
        # set starting penalty parameter; if invalid value, set to 1
        self.tau = tau0
        if tau0 <= 0:
            self.tau = 1
            
        # set problem to be solved 
        self.problem = problem
        
        # set increase rate of penalty parameter 
        self.tau_gr = tau_gr
        
        # set tolerances for stopping conditions of inner and outer loops
        self.inner_eps = inner_eps
        self.outer_eps = outer_eps
        
        # set upper bound on penalty parameter value for numerical reasons
        self.tau_max = tau_max
        
        # parameters for spectral gradient; variables are named according to notation by Jia et al. (2022)
        self.m = m
        self.gamma0 = gamma0
        self.gamma_max = gamma_max
        self.gamma_inc = gamma_inc
        self.sigma = sigma
        self.eta = eta
        
        # enable logging prints 
        self.debug = debug
        
        # initialize counters
        self.inner_iters = 0
        self.time_proj = 0
        
        # maximum number of spectral gradient iterations
        self.max_sgm_iters = max_sgm_iters

    
    def penalty_f(self, x):
        '''augmented Lagrangian function defined according to Jia et al. (2022)'''
        
        return self.problem.f(x) + 0.5*self.tau*(self.problem.dist2_C(self.problem.G(x)+self.lambdas/self.tau))

    def grad_penalty_f(self, x):
        '''derivative of the augmented Lagrangian function defined according to Jia et al. (2022)'''
        
        return self.problem.grad_f(x) + 0.5*self.tau*np.matmul(self.problem.jac_G(x).T, self.problem.grad_dist2_C(self.problem.G(x)+self.lambdas/self.tau)).reshape(x.shape)


    def solve(self):
        '''proxy function for actual solver'''
        
        return self.solve_alm()
        


    def solve_alm(self):
        '''augmented Lagrangian method from [Jia et al., "An augmented Lagrangian 
        method for optimization problems with structured geometric constraints" (2022)]'''
        
        # start measuring runtime and iterations
        tic = time()
        iteration = 1
        
        # initialize former auxiliary function value
        V_1 = np.inf
        
        
        # Outer loop
        while True:
            
            # compute value of penalty function with current penalty parameter (which has just been updated)
            prev_obj = self.penalty_f(self.x)
            
            # Inner loop
            while True:

                # update solution by spectral gradient method 
                self.x, _ =  self.sgm(self.x)
                
                # compute value of augmented Lagrangian function at the new solution
                obj = self.penalty_f(self.x)
                
                # stop inner loop if decrease of augmented Lagrangian function is under given threshold 
                if prev_obj-obj < self.inner_eps:
                    break
                
                # update latest value observed for penalty function
                prev_obj = obj

            # log for debugging purposes at each outer iteration
            if self.debug:
                print("Iter {}, tau {:.4f}".format(iteration, self.tau))
                print("\t -> f(y) {:.5f}, violation: {}".format(self.problem.f(self.x), self.problem.dist2_C(self.problem.G(self.x))**(1/2)))

            iteration += 1

            # if current solution is (approximately) feasible, stop the outer loop 
            if self.problem.dist2_C(self.problem.G(self.x))**(1/2) < self.outer_eps:
                break
            
            
            # Lagrange multipliers update by Hestenes-Powell-Rockafellar rule with safeguarding 
            # XXX maybe this can be done in matrix form with numpy?
            self.lambdas = 0.5*self.tau*self.problem.grad_dist2_C(self.problem.G(self.x)+self.lambdas/self.tau)
            for i in range(len(self.lambdas)):
                self.lambdas[i] = max(-self.lambda_max,min(self.lambdas[i], self.lambda_max))
            
            # compute auxiliary function
            V = self.problem.norm(0.5*self.problem.grad_dist2_C(self.problem.G(self.x)+self.lambdas/self.tau)-self.lambdas/self.tau) 
            
            # increase penalty parameter if auxiliary function did not decrease enough
            if  V >= self.eta * V_1:
                self.tau *= self.tau_gr
                self.tau = min(self.tau, self.tau_max)
            
            # update former value of auxiliary function
            V_1 = V
            
            
        # log for debugging purposes at the end of the algorithm
        if self.debug:
            print('time_proj', self.time_proj)

        # always print solution process final information        
        print('f(y*) = {}, violation: {}, time = {}, n_iters = {}, inner_iters = {}'.format(self.problem.f(self.x), self.problem.dist2_C(self.problem.G(self.x))**(1/2), time()-tic, iteration-1, self.inner_iters))

        # return solution and optimization process information
        return {'f':self.problem.f(self.x),'t':time()-tic,'sol':self.x,'vio':self.problem.dist2_C(self.problem.G(self.x))**(1/2),'iters':iteration-1}



    def sgm(self,xk):
        '''Solve subproblem for current penalty function with Spectral Gradient
        method from [Jia et al., "An augmented Lagrangian method for optimization 
        problems with structured geometric constraints" (2022)]'''
        
        # intitialize iteration counter
        iteration = 0
        
        #nonmonotone strategy memory
        past = []
        
        # clone current solution
        x = np.copy(xk)
        
        # main spectral gradient loop
        while True:
            
            # set tentative stepsize
            gamma = self.gamma0
            
            # compute objective function (augmented Lagrangian) at current solution
            f = self.penalty_f(x)
            
            # store value in memory
            past.append(f)
            
            # if memory is full, delete oldest element
            if len(past) > self.m:
                past.pop(0)
                
            # compute gradient at current point
            g = self.grad_penalty_f(x)
            
            # increase counter for total number of inner iterations
            self.inner_iters += 1
            
            # Nonmonotone projected gradient line search
            while True:
                
                # descent step along negative gradients with tentative stepsize
                p = x - 1/gamma*g
                
                tic = time()
                
                # project the point obtained by gradient descent onto the difficult constraint D
                x_trial = self.problem.proj_D(p)
                
                self.time_proj += time() - tic
                
                # if nonmonotone Armijo condition is satisfied accept the step and update the solution
                if self.penalty_f(x_trial) <= np.max(past) + self.sigma*self.problem.inner_prod(g,x_trial-x) or gamma*self.gamma_inc>self.gamma_max:
                    x = x_trial
                    break
                
                # otherwise decrease tentative stepsize
                else:
                    gamma = min(gamma*self.gamma_inc, self.gamma_max)
                    
            # increase iteration counter
            iteration += 1
            
            # stop if all values in memory are close or max number of iterations reached    
            if np.max(past) - np.min(past[:] + [self.penalty_f(x)]) < self.inner_eps or iteration>self.max_sgm_iters:
                break
        
        # return obtained solution
        return x, 0


    

class MIPSolverPortfolio(object):
    '''Solver for spare portfolio selection problems bas on MINLP and Gurobi solver'''
    
    def __init__(self, problem):

        # set problem to be solved
        self.problem = problem


    def solve(self):
        '''build Gurobi model and solve it'''
        
        # Creating Gurobi model
        model = Model()
        debug = False
        
        if not debug:
            # Quieting Gurobi output
            model.setParam("OutputFlag", False)
        
        # set Gurobi time limit
        time_limit = 1000
        model.setParam("TimeLimit", time_limit)
        
        tic = time()

        
        
        # set integer precision to highest possible
        model.setParam("IntFeasTol", 1e-09)


        # retrieve number of vatriables
        n = len(self.problem.get_x0())


        # Add variables to the model
        x, z = [], []

        for j in range(n):
            
            # original (continuous) variable, with bounds [0,1] 
            x.append(model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x{}".format(j)))
            
            # binary auxiliary variable
            z.append(model.addVar(vtype=GRB.BINARY, name="z{}".format(j)))

        # define and set objective function
        f = 0.5*quicksum(x[i]*x[j]*self.problem.Q[i,j] for i in range(n) for j in range(n)) + self.problem.lam * quicksum(x[i]*self.problem.c[i] for i in range(n))
        model.setObjective(f)

        # big M value set to 2 is enough (>1 would be sufficient)
        M = 2


        for j in range(n):
            # big-M constraint to model logical implication z=0 => x=0
            model.addConstr(x[j] <= M*z[j])

        # unit simplex constraint
        model.addConstr(quicksum(x[i] for i in range(n)) <= 1)
        model.addConstr(quicksum(x[i] for i in range(n)) >= 1)
        
        # cardinality constraint
        model.addConstr(quicksum(z[i] for i in range(n)) <= self.problem.s)

        # Solve
        model.optimize()

        # print result for debugging aims
        if debug:
            for v in model.getVars():
                print("Var: {}, Value: {}".format(v.varName, v.x))

        # retrieve solution
        sol = np.array([model.getVarByName("x{}".format(j)).x for j in range(n)])
           
        # always print solution process final information                
        print('f(y*) = {}, violation: {}, time = {}'.format(self.problem.f(sol), self.problem.dist2_C(sol)**(1/2), time()-tic))

        # return solution and optimization process information        
        return {'f':self.problem.f(sol),'t':time()-tic,'sol':sol,'vio':self.problem.dist2_C(sol)**(1/2),'iters':0}




class SingleTaskSolver(object):
    '''Simple solver to independently train logistic regression models in multi-task setting'''
    
    def __init__(self,mtp, nu=0.1):
        # multi-task logistic regression problem
        self.mtp = mtp
        
        # number of features
        self.nw = mtp.n_feat
        
        # regularization term
        self.nu = nu


    def solve(self, verbose=False, g_tol=1e-5):
        # initialize weights
        W = np.zeros((self.mtp.n_tasks, self.nw))
        
        # fit each task independently
        for i in range(self.mtp.n_tasks):
            
            # logs
            if verbose:
                print('Training task {}'.format(i))
                print('Starting loss task {}: {}'.format(i, self.mtp.task_loss(i,W[i,:])))
                
            # define objective function (regularized log loss) and gradient
            def objf(t):
                return self.mtp.task_loss(i, t)/self.mtp.tasks[i]['Y'].shape[0] + 0.5*self.nu* np.linalg.norm(t)**2
            def objg(t):
                return self.mtp.task_derivative(i, t)/self.mtp.tasks[i]['Y'].shape[0] + self.nu*t
            
            # fit model with L-BFGS and store solution
            w = minimize(objf, W[i,:], jac=objg, method="L-BFGS-B", options={"iprint": -1, "maxcor": 10, "gtol": g_tol, "ftol": 10*np.finfo(float).eps, "maxiter": 500, "maxls": 20})
            W[i,:] = w.x
            
            #log
            if verbose:
                print('Training loss task {}: {}'.format(i, self.mtp.task_loss(i,W[i,:])/self.mtp.tasks[i]['Y'].shape[0]))

        # always print solution process final information       
        print('Total loss: {}'.format(self.mtp.full_loss(W) + 0.5*self.nu*np.linalg.norm(W)**2))

        # return all models in matrix form
        return W



class EnumerationSolver:
    '''(Abstract) class for brute force solver for disjunctive programming problems with disjoint linear constraints'''
    
    def __init__(self,DP):
        
        # set problem to be solved
        self.DP = DP


    def solve(self):
        '''solve disjunctive programming problem solving independently for each component of disjoint set'''
        
        tic = time()
        
        # initialize best value found so far
        best_f = np.inf
        
        # flag: a feasible solution has been found
        feasible = False
        
        # for each component of disjoint set
        for i in range(self.DP.A.shape[0]):
            
            # find optimal solution for problem subject to D_i
            obji, feas_i = self.single_solve(i)

            # if solution is feasible and improves best objective found so far, save it
            if obji < best_f and feas_i:
                best_f = obji
                feasible = True
        
        # log
        print('Optimal solution: ', best_f, 'time: ', time()-tic)
        
        # return feasibility flag for the overall problem
        return feasible

    def single_solve(self,h):
        '''(abstract) method, solves problem only taking into account disjoint set D_h'''
        pass



class NonlinearEnumerationSolver(EnumerationSolver):
    ''''Brute force solver for disjunctive programming problems with disjoint linear constraints, 
    nonlinear objective and a shared nonlinear constraint as in Section 6.3 of Kanzow and Lapucci (2023)'''
    
    
    def __init__(self,DP):
        super().__init__(DP)


    def single_solve(self,h):
        '''Solves problem only taking into account component D_h of disjoint set D'''
        
        # Coefficients of linear constraints for set D_h
        A, b = self.DP.A[h], self.DP.b[h]

        # initialize list of constraints for SLSQP
        constrs = []
        
        # linear inequality constraints that define D_h
        for i in range(A.shape[0]):
            constrs.append({'type':'ineq', 'fun': self.constr_i(i,A,b)})
        
        # shared nonlinear constraint
        constrs.append({'type':'ineq', 'fun': lambda t: -np.dot(self.DP.coeff,np.power(t-self.DP.point,4))+self.DP.t})
        
        #solve 
        sol = minimize(self.DP.f, np.zeros(self.DP.n), method='SLSQP', constraints=constrs)
        
        # return solution and feasibility check; N.B. SLSQP still returns a solution even if problem is not feasible
        return self.DP.f(sol.x), np.all([constrs[j]['fun'](sol.x) >= -1e-6 for j in range(len(constrs))])


    def constr_i(self,i,A,b):
        '''auxiliary function for defining linear constraints'''
        
        def _(t):
            return - np.dot(A[i,:],t)+b[i]
        return _

class DifficultEnumerationSolver(NonlinearEnumerationSolver):
    ''''Brute force solver for disjunctive programming problems with disjoint linear constraints, 
    nonlinear objective and multiple shared nonlinear constraints as in Section 6.3 of Kanzow and Lapucci (2023).
    Extends NonlinearEnumerationSolver'''
    
    
    def __init__(self,DP):
        super().__init__(DP)

    def single_solve(self,h):
        '''Solves problem only taking into account component D_h of disjoint set D'''
        
        # Coefficients of linear constraints for set D_h
        A, b = self.DP.A[h], self.DP.b[h]

        # initialize list of constraints for SLSQP
        constrs = []
        
        # linear inequality constraints that define D_h
        for i in range(A.shape[0]):
            constrs.append({'type':'ineq', 'fun': self.constr_i(i,A,b)})
        
        # shared nonlinear constraints
        for i in range(len(self.DP.ts)):
            constrs.append({'type':'ineq', 'fun': self.diff_constr_i(i,self.DP.points[i], self.DP.coeffs[i], self.DP.ts[i])})
        
        #solve 
        sol = minimize(self.DP.f, np.zeros(self.DP.n), method='SLSQP', constraints=constrs)
        
        # return solution and feasibility check; N.B. SLSQP still returns a solution even if problem is not feasible
        return self.DP.f(sol.x), np.all([constrs[j]['fun'](sol.x) >= -1e-6 for j in range(len(constrs))])


    def diff_constr_i(self,i,point,coeffs,t):
        '''auxiliary function for defining nonlinear constraints'''
        
        def _(x):
            return -np.dot(coeffs,np.power(x-point,4))+t
        return _
