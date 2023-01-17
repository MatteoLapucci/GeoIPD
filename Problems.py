import numpy as np
import scipy.linalg
from utils import sigmoid
from gurobipy import *

class Problem:
    ''' "Abstract" class for problems addressed in [Kanzow and Lapucci,
    "Inexact Penalty Decomposition Methods for Optimization Problems with Geometric 
    Constraints" (2023)]'''
    
    def __init__(self):
        
        # set starting solution with default one
        self.x0 = self.default_x0()

    def f(self, x):
        '''objective function'''
        
        # abstract method
        pass


    def grad_f(self, x):
        '''gradient of objective function'''
        
        # abstract method
        pass


    def is_feasible(self, x):
        '''check feasibility of point x'''
        
        # abstract method
        pass

    def dist_D(self, x):
        '''compute distance from x to difficult set D'''
        
        # abstract method
        pass

    def get_x0(self):
        '''getter function for starting solution'''
        
        return np.copy(self.x0)
    
    def default_x0(self):
        '''return default starting solution for the problem'''
        
        # abstract method
        pass
    
    def set_x0(self,x):
        '''set new starting solution'''
        
        self.x0 = np.copy(x)

    def proj_D(self, x):
        '''projection operator onto difficult set D'''
        
        # abstract method
        pass

    def dist(self, x, y):
        '''distance between to points in variables space'''
        
        # distance is computed based on norm of variables spaces, specified in self.norm function 
        return self.norm(x-y)
    
    def G(self,x):
        '''compute differentiable constraints G at x'''
        
        # abstract method
        pass
    
    def jac_G(self,x):
        '''compute Jacobian matrix of differentiable constraints at x'''
        
        # abstract method
        pass

    def dist2_C(self,y):
        '''compute squared distance of point y in Y (image space of G) from convex set C'''
        
        # abstract method
        pass

    def grad_dist2_C(self, y):
        '''compute squared distance of point y in Y (image space of G) from convex set C'''
        
        # abstract method
        pass

    def norm(self, x):
        '''compute norm of point in variable space'''
        
        # default norm is Euclidean norm for vectors, Frobenius for matrices 
        return np.linalg.norm(x)

    def inner_prod(self, x, y):
        '''compute inner product of two points in variables space'''
        
        # abstract method
        pass

    def identity_mapping(self):
        '''return matrix representing identity mapping in (possibly flattened) variables space'''
        
        # abstract method
        pass

    def linear_map(self, A, x):
        '''apply linear mapping A to point x; A is represented in matrix form and is applied to 
        a possibly flattend version of x; in that case the result is then recast to original shape'''
        
        # abstract method
        pass

    def outer_prod(self, x, y):
        '''outer product in variables space'''
        
        # abstract method
        pass


class VectorProblem(Problem):
    ''' "Abstract" class for problems with 1-D vectors variables'''
    
    def __init__(self):
        super().__init__()


    def inner_prod(self, x,y):
        '''inner product in vector space'''
        
        # standard dot product
        return np.dot(x,y)

    def identity_mapping(self):
        '''identity mapping in vector space'''
        
        # return standard identity matrix
        return np.identity(len(self.get_x0()))


    def linear_map(self, A, x):
        '''linear mapping in vector space'''
        
        # return result of standard matrix-vector multiplication
        return np.matmul(A,x)

    def outer_prod(self, x, y):
        '''outer product in vector space'''
        
        # return standard outer product
        return np.outer(x,y)
    
    def G(self,x):
        '''compute differentiable constraint G at x'''
        
        # by default, no additional differentiable constraint is present; G is identity, C is entire space
        return x
    
    def jac_G(self,x):
        '''compute Jacobian matrix of differentiable constraint G at x'''
        
        # by default, no additional differentiable constraint is present; G is identity, C is entire space
        return self.identity_mapping()


class MatrixProblem(Problem):
    ''' "Abstract" class for problems with matrix variables'''
    
    def __init__(self):
        super().__init__()

    def inner_prod(self, x,y):
        '''compute Frobenius inner product'''
        
        return np.dot(x.flatten(),y.flatten())

    def identity_mapping(self):
        '''identity matrix of flattened matrices'''
        
        tmp = self.get_x0()
        return np.identity(len(tmp.flatten()))

    def linear_map(self, A, x):
        '''apply matrix A to flattened version of matrix x, then reshape back to original shape'''
        
        flattened_result =  np.matmul(A,x.flatten())
        return flattened_result.reshape(x.shape)

    def outer_prod(self, x, y):
        '''outer product between x and y after flattening'''
        
        return np.outer(x.flatten(),y.flatten())
    
    
    def G(self,x):
        '''compute differentiable constraint G at x'''
        
        # by default, no additional differentiable constraint is present; G is identity, C is entire space
        return x
    
    def jac_G(self,x):
        '''compute Jacobian matrix of differentiable constraint G at x'''
        
        # by default, no additional differentiable constraint is present; G is identity, C is entire space
        return self.identity_mapping()



class CardConstrProblem(VectorProblem):
    '''Sparsity constrained optimization problems'''
    
    def __init__(self, s):
        super().__init__()
        
        # bound of carsinality constraint
        self.s = s


    def l0(self,x, threshold=1e-6):
        '''compute l0 norm of vector x; variables such that |x_i|<threshold are considered zero'''
        
        return np.sum(np.abs(x) > threshold)

    def is_feasible(self, x):
        '''check overall feasibility of point x'''
        
        return self.l0(x) <= self.s and self.dist2_C(self.G(x))**(1/2) <= 1e-10


    def proj_D(self, x):
        '''projection onto the sparse set; set to zero the n-s smallest components in absolute value'''
        
        # number of problem variables
        n = x.shape[0]
        
        #number of variables to be set to zero
        n_zeros = n - self.s
        
        # initialize projection as zeros
        new_x = np.zeros(n)
        
        # partition variables vector in two groups: last n-s indices denote largest variables in absoulte value
        indices = np.argpartition(np.abs(x), n_zeros)
        
        # largest variables keep the value as the original point
        new_x[indices[n_zeros:]] = x[indices[n_zeros:]]
        
        #return projection
        return new_x




class SparseQuadraticProblem(CardConstrProblem):
    '''Quadratic problem with sparsity constraints'''
    
    def __init__(self, s, Q, c, lam=1):
        
        # Hessian matrix
        self.Q  = Q
        
        #linear coefficients
        self.c = c
        
        # trade-off coefficient
        self.lam = lam
        
        super().__init__(s)
        


    def f(self, x):
        '''compute 1/2 x^TQx + lambda*c^Tx'''
        
        return 0.5*np.dot(x, np.matmul(self.Q,x)) + self.lam*np.dot(x,self.c)


    def grad_f(self, x):
        '''gradient of f is given by Qx + lambda*c'''
        
        return np.matmul(self.Q,x) + self.lam*self.c

    def default_x0(self):
        '''default starting solution is the origin'''
        
        return np.zeros(len(self.c))



    def dist2_C(self,x):
        '''no additional constraints, C is the entire space'''
        
        return 0

    def grad_dist2_C(self, x):
        '''gradient of null function is vector of zeors'''
        
        return np.zeros(len(self.c))



class SparsePortfolioProblem(SparseQuadraticProblem):
    '''Portfolio selection problem with sparsity constraints; inherits objective by SparseQuadraticProblem'''
    
    def __init__(self, s, Q, c, lam=1, b=1):
        
        # budget parameter
        self.b = b
        
        super().__init__(s,Q,c,lam)
        


    def default_x0(self):
        '''default starting solution is (b/n,...,b/n)'''
        
        n = len(self.c)
        return self.b*np.ones(n)/n

    def dist2_C(self,y):
        '''distance from C computed as distance between point and projection onto C'''
        
        return np.linalg.norm(y-self.simplex_proj(y))**2

    def grad_dist2_C(self, y):
        '''gradient of squared distance from computed following equation (2.1) in Kanzow and Lapucci (2023)'''
        
        return 2*(y-self.simplex_proj(y))

    def simplex_proj(self, x):
        '''projection onto simplex following Wang and Carreira-Perpinán,"Projection 
        onto the probability simplex: An efficient algorithm with a simple proof, and an application" (2013).'''
        
        y = np.sort(x)[::-1]
        y /= self.b
        rho = 0
        for j in range(len(y)):
            if y[j]+(1-np.sum(y[:j+1]))/(j+1) >0:
                rho = j+1
            else:
                break
        tau = (1-sum(y[:rho+1]))/rho
        return self.b * np.array([max(x[i]+tau,0) for i in range(len(y))])


class LowRankProblem(MatrixProblem):
    '''Matrix optimization problem with low-rank constraints'''
    
    def __init__(self, s):
        super().__init__()
        
        # upper bound on solution rank
        self.s = s
    
    
    def rank(self,x):
        '''compute rank of matrix x'''
        
        return np.linalg.matrix_rank(x) 

    def is_feasible(self, x):
        '''check overall feasibility of point x'''
        
        return self.rank(x) <= self.s and self.dist2_C(self.G(x))**(1/2) <= 1e-10


    def proj_D(self, x):
        '''project matrix x onto low-rank space'''
        
        return self.svd_proj(x)

    def svd_proj(self,x):
        '''actual projection function onto low-rank space; compute SVD, then set to zero smallest (in absolute value) singular values'''
        
        # compute svd of matrix x in compact form
        U, SV, V = np.linalg.svd(x, full_matrices=False)
        
        # length of the singular values diagonal SV
        n = SV.shape[0]
        
        # number of singular values to be set to zero
        n_zeros = max(n - self.s,0)

        # initialize new singular values matrix as all zeros
        new_SV = np.zeros(n)
        
        # partition elements of SV in two groups: last n-s indices denote largest singular values in absoulte value
        indices = np.argpartition(np.abs(SV), n_zeros)
        
        # largest singular values are presereved 
        new_SV[indices[n_zeros:]] = SV[indices[n_zeros:]]
        
        # returned the reconstructed new low-rank matrix
        return np.dot(U * new_SV, V)


    
    
class LowRankPSDP(LowRankProblem):
    '''Positive definite matrix optimization problem; only difference w.r.t. LowRankProblem is in the projection operation,
    that is cheaper here'''
    
    def __init__(self, s):
        super().__init__(s)


    def proj_D(self, x):
        '''compute projection of matrix x onto the low-rank squared positive semi-definite matrices space'''
        
        # size of the squared matrix
        n = x.shape[0]
        
        # compute largest s eigenvalues and corresponding eigenvectors 
        eigvals, eigvects = scipy.linalg.eigh(x, subset_by_index=[n-self.s, n-1])
        eigvects = eigvects.T
        
        # reconstruct matrix by eigenvalue decomposition only using s largest eigenvalues
        return sum(max(0,eigvals[i])*np.outer(eigvects[i],eigvects[i]) for i in range(len(eigvals)))


class NearestCorrelationMatrixProblem(LowRankPSDP):
    '''Nearest low-rank correlation matrix problem'''
    
    def __init__(self, s, C):
        
        # reference correlation matrix
        self.C = C
        
        # all ones vector (diagonal of C)
        self.e = np.ones(C.shape[0])
        
        super().__init__(s)
        
        # initialize Jacobian of G; since it is constant, we will compute it only once and then store it
        self.jG = None

    def f(self, x):
        '''objective function is 1/2||x-C||^2_F'''
        
        return 0.5*self.norm(x-self.C)**2


    def grad_f(self, x):
        '''derivative of the objective function is x-C'''
        
        return x-self.C

    def default_x0(self):
        '''default starting solution is the identity matrix'''
        
        return np.identity(self.C.shape[0])

    
    def G(self,x):
        '''y = diag(x)'''
        
        return np.array([x[i,i] for i in range(self.C.shape[0])])

    def jac_G(self,x):
        '''jG is a matrix such that each row i contains the (flattened) gradient of x w.r.t. f(x) = x_i, i.e., 
        has value 1 at positions (i, i*(n+1)) for i=0,...,n and 0 elsewhere; since it is constant, it is
        computed once and then stored in memory'''
        
        if self.jG is None:
            A = np.zeros((self.C.shape[0], self.C.shape[0]**2))
            for i in range(A.shape[0]):
                A[i,i+i*A.shape[0]] = 1
            self.jG = A
        return self.jG

    def dist2_C(self,y):
        '''squared distance of y=diag(x) from e=(1,1,...,1)'''
        
        return np.linalg.norm(y-self.e)**2

    def grad_dist2_C(self, y):
        '''gradient of the squared distance of y from e'''
        
        return 2*(y-self.e)



class ExplicitCorrelationProblem(NearestCorrelationMatrixProblem):
    '''Nearest low-rank correlation matrix problem', but linear constraint 
    diag(x) = (1,1,...,1) is handled within the difficult set D'''
    
    def __init__(self, s, C):
        super().__init__(s,C)

    def f(self, x):
        '''objective function is 1/2||x-C||^2_F'''
        
        return 0.5*self.norm(x-self.C)**2


    def grad_f(self, x):
        '''derivative of the objective function is x-C'''
        
        return x-self.C

    def default_x0(self):
        '''default starting solution is the identity matrix'''
        
        return np.identity(self.C.shape[0])



    def dist2_C(self,y):
        '''here C is the entire space, so distance is always 0'''
        
        return 0

    def grad_dist2_C(self, y):
        '''gradient of null function is the zero vector'''
        
        return np.zeros_like(y)

    def G(self,x):
        '''no explicit differentiable constraint in the problem; for simplicity we set G(x) = 0,
        as the Jacobian is smaller to store and handle than the identity mapping'''
        
        return np.array([0])

    def jac_G(self,x):
        '''Jacobian of G(x)=0 is the null mapping'''
        
        return np.zeros_like(x).reshape((1,x.size))



class MTLRProblem(LowRankProblem):
    '''Low-rank Multi-task Logistic Regression problem
    variable matrices have the form x = [W, U, V], where W = U + V is the models matrix;
    the U parts are indepentently trained on each task, subject to a regularization penalty;
    V are not individually regularized, but v_i, i=1,...,m, lie in a common subspace'''
    
    def __init__(self, s, mtp, nu=0.1):
        
        # upper bound on low-rank component of the models matrix 
        self.s = s
        
        # the problem to be solved
        self.mtp = mtp
        
        # number of variables of each model
        self.nw = mtp.n_feat
        
        # regularization parameter
        self.nu = nu
        
        # initialize Jacobian of G; since it is constant, we will compute it only once and then store it
        self.jG = None
        
        super().__init__(s)


    def f(self,x):
        '''objective function is the sum of (normalized) losses of individual tasks for models W, plus a regularization term on U'''
        
        return self.mtp.full_loss(x[:,:self.nw]) + 0.5*self.nu*self.norm(x[:,self.nw:2*self.nw])**2

    def grad_f(self,x):
        '''derivative of objective funtion w.r.t. x'''
        
        # initialize derivatives w.r.t. W,U and V as 0  
        J = np.zeros_like(x)
        
        # derivative w.r.t. W is given by gradients of logistic loss of each task 
        J[:,:self.nw] = self.mtp.jacobian(x[:,:self.nw])
        
        # derivative of U is the derivative of the regularizer nu*||U||^2
        J[:,self.nw:2*self.nw] = self.nu*x[:,self.nw:2*self.nw]
        
        # V does not appear in objective function
        
        # return derivatives
        return J

    def default_x0(self):
        '''default starting solution is zeroes matrix'''
        
        x0 = np.zeros((self.mtp.n_tasks,3*self.nw))
        return x0

    def rank(self,x):
        '''compute rank of V part of x = [W,U,V]'''
        
        return np.linalg.matrix_rank(x[:,self.nw*2:]) 


    def proj_D(self, x):
        '''project V part of x = [W,U,V] onto low-rank space'''

        # extract V part of x        
        xv = x[:,2*self.nw:]
        
        # project V onto low-rank space by SVD strategy
        xlr = self.svd_proj(xv)
        
        # update V part of x with low rank projection
        xr = np.copy(x)
        xr[:,2*self.nw:] = xlr
        
        # return projected solutions
        return xr

    def G(self,x):
        '''given x = [W,U,V], G(x) = W - U -V'''
        
        G = x[:,:self.nw] - x[:,self.nw:2*self.nw] - x[:,2*self.nw:]
        
        # return flattened W-U-V   
        return G.flatten()

    def dist2_C(self,y):
        '''constraint is G(x) = 0, thus squared distance of y=G(x) from C = {0} is ||y-0||^2 = ||y||^2'''
        
        return np.linalg.norm(y)**2

    def grad_dist2_C(self,y):
        '''gradient of ||y||^2 is 2*y'''
        
        return 2*y


    def jac_G(self,x):
        '''compute, if not already stored, jacobian of G(x) = W-U-V (that is constant)'''
        
        if self.jG is None:
            A = np.zeros((self.nw*self.mtp.n_tasks, self.nw*self.mtp.n_tasks*3))
            for i in range(A.shape[0]):
                A[i,i%self.nw+3*self.nw*(i//self.nw)] = 1
                A[i,i%self.nw+3*self.nw*(i//self.nw) + self.nw] = -1
                A[i,i%self.nw+3*self.nw*(i//self.nw) + 2*self.nw] = -1
            self.jG = A
        return self.jG







class DisconnectedProblem(VectorProblem):
    '''Disjunctive programming problem with quadratic objective and linear constraints (both shared and disjoint)'''
    
    def __init__(self,A,b,C,d,Q,p):
        
        # number of variables
        self.n = A.shape[2]
        super().__init__()
        
        # lists of coefficient matrices for disjoint components
        self.A = A
        self.b = b
        
        # coefficient matrix and constants of shared linear constraints
        self.C = C
        self.d = d
        
        # Hessian matrix and linear coefficients of objective
        self.Q = Q
        self.p = p



    def is_feasible(self,x):
        '''check feasibiliy of x for the overall problem; x is feasible if it satisfies 
        all shared constraints Cx=d and at least one set of constraints A_i x <= b_i'''
        
        return np.any([np.all(np.dot(self.A[i],x) - self.b[i] <= 1e-10) for i in range(self.A.shape[0])]) and self.dist2_C(self.G(x))**(1/2) <= 1e-4


    def proj_lin(self,point,A,b):
        '''projection onto component of disjoint set D, defined by Ax<=b; the quadratic programming problem
        with linear constraints min_x ||x-point||^2 s.t. Ax <=b is solved with Gurobi optimizer'''
        
        # Creating Gurobi model
        model = Model()
        debug = False
        
        if not debug:
            # Quieting Gurobi output
            model.setParam("OutputFlag", False)
        
        # setting time limit
        time_limit = 100
        model.setParam("TimeLimit", time_limit)

        

        # number of variables
        n = self.n
        
        # variables vector
        x = []

        # define variables
        for j in range(n):
            # n continuous variables with foo bounds
            x.append(model.addVar(lb=-1000, ub=1000, vtype=GRB.CONTINUOUS, name="x{}".format(j)))

        # objective is ||x-point||^2
        f = quicksum((point[i]-x[i])**2 for i in range(n))
        model.setObjective(f)
        
        # constraint Ax <= b
        for j in range(A.shape[0]):
            model.addConstr(quicksum([A[j,i]*x[i] for i in range(A.shape[1])]) <= b[j])

        # Solve
        model.optimize()

        # logs for debugging purposes
        if debug:
            for v in model.getVars():
                print("Var: {}, Value: {}".format(v.varName, v.x))

        if model.status == GRB.OPTIMAL:
            # retrieve solution
            sol = np.array([model.getVarByName("x{}".format(j)).x for j in range(n)])
        else:
            
            # considered component of disjoint set (Ax <= b) is empty, no projection possible
            sol = None

        # return projection if set nonempty, else None
        return sol



    def proj_D(self,x):
        '''Projection onto the overall disjoint set D'''
        
        # initialize distance to closest feasible point
        dist = np.inf
        
        # initialize closest feasible point
        y = None
        
        # for all components D_i of the disjoint set D
        for i in range(self.A.shape[0]):
            
            # compute projection onto D_i
            p = self.proj_lin(x,self.A[i],self.b[i])
            
            if p is not None:
                # if set D_i is not empty and projection exists, compute distance from x to this projection
                dist_i = self.dist(x,p)
                
                # if this projection is closer to x than y, update y and distance of x to feasible set
                if dist_i < dist:
                    dist = dist_i
                    y = p

        # return closest feasible point y to x 
        return y

    def f(self, x):
        '''quadratic objective function: 1/2 x^TQx - p^Tx'''
        
        return 0.5*np.dot(x,np.dot(self.Q,x)) - np.dot(self.p,x)

    def grad_f(self, x):
        '''gradient of quadratic objective function: Qx-p'''
        
        return np.dot(self.Q,x) - self.p


    def default_x0(self):
        '''default starting solution is zero vector'''
        
        return np.zeros(self.n)


    def G(self,x):
        '''G(x) = Cx'''
        
        return np.matmul(self.C,x).squeeze()

    def jac_G(self,x):
        '''Jacobian of Cx is C'''
        
        return self.C



    def dist2_C(self,y):
        '''C = {d}, dist(y,C) = ||y-d||'''
        
        return self.norm(y-self.d)**2

    def grad_dist2_C(self, y):
        '''gradient of ||y-d||^2 = 2(y-d)'''
        
        return (2*(y-self.d))





class NonlinearDisconnectedProblem(DisconnectedProblem):
    '''Disjunctive programming problem with logistic objective function, disjoint 
    linear constraints and a nonlinear shared constraint (see Kanzow and Lapucci (2023), Section 6.3)'''

    
    def __init__(self,A,b,X,Y,coeff,t,point):
        
        # number of variables
        self.n = A.shape[2]
        
        # define disjoint linear constraints in DisconnectedProblem constructor
        super().__init__(A,b,None,None,None,None)
        
        # coefficients of nonlinear constraint
        self.coeff = coeff
        self.t = t
        self.point=point

        
        # data for logistic loss
        self.X = X
        self.Y = Y



    def is_feasible(self,x):
        '''check feasibiliy of x for the overall problem; x is feasible if it satisfies 
        shared constraint G(x) in C and at least one set of constraints A_i x <= b_i'''
        
        return np.any([np.all(np.dot(self.A[i],x) - self.b[i] <= 1e-10) for i in range(self.A.shape[0])]) and self.dist2_C(self.G(x))**(1/2) <= 1e-4

    def f(self,x):
        '''objective function: logistic loss of model x on data self.X, self.Y'''
        
        return np.sum(np.log(1+np.exp(np.multiply(-self.Y, np.dot(self.X,x)))))


    def grad_f(self,x):
        '''gradient of logistic loss function'''
        
        r = np.multiply(-self.Y,sigmoid(-np.multiply(self.Y,np.dot(self.X,x))))
        return np.matmul(self.X.T, r)

    def G(self,x):
        '''definition of nonlinear G(x) (only one constraint) as in Section 6.3 of Kanzow and Lapucci(2023)'''
        
        return np.dot(self.coeff,np.power(x-self.point,4))

    def jac_G(self,x):
        '''Jacobian of G(x)'''
        
        J = np.multiply(self.coeff,4*np.power(x-self.point,3))
        return np.expand_dims(J,axis=0)



    def dist2_C(self,y):
        '''squared distance of y=G(x) from C=[self.t, infty)'''
        
        return (max(0,y-self.t))**2

    def grad_dist2_C(self, y):
        '''gradient of dist2_C'''
        
        v = 2*max(y-self.t, 0)
        v = np.array(v)
        if len(v.shape) <1:
            v = np.expand_dims(v,axis=0)
        return v




class DifficultNDP(NonlinearDisconnectedProblem):
    '''Disjunctive programming problem with logistic objective function, disjoint 
    linear constraints and a set of nonlinear shared constraint (see Kanzow and Lapucci (2023), Section 6.3)'''
    
    def __init__(self,A,b,X,Y,coeffs,ts,points):
        
        # number of variables
        self.n = A.shape[2]
        
        # initialize disjoint set and data for logistic loss
        super().__init__(A,b,X,Y,None,None,None)
        
        # coefficients for nonlinear shared constraints
        self.coeffs = coeffs
        self.ts = ts
        self.points=points

    def G(self,x):
        '''definition of nonlinear G(x) as in Section 6.3 of Kanzow and Lapucci(2023)'''
        
        return np.sum(np.multiply(self.coeffs,np.power(x-self.points,4)),axis=1)

    def jac_G(self,x):
        '''definition of nonlinear G(x) as in Section 6.3 of Kanzow and Lapucci(2023)'''
        
        J = np.multiply(self.coeffs,4*np.power(x-self.points,3))
        return J


    def dist2_C(self,y):
        '''distance of y from C = [t_1,infty)x...x[t_m,infty)'''
        
        ypos = np.maximum(y-self.ts,0)
        return (self.norm(ypos))**2

    def grad_dist2_C(self, y):
        '''gradient of dist2_C'''
        
        v = 2*np.maximum(y-self.ts, 0)
        if len(v.shape) <1:
            v = np.expand_dims(v,axis=0)
        return v
