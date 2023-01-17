import scipy.io
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools

def sigmoid(z):
    '''sigmoid function'''
    return 1/(1 + np.exp(-z))

class MTLProblem:
    '''Multitask Logistic Regression Problem'''
    
    def __init__(self,tasks):
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.n_feat = tasks[0]['X'].shape[1]

    def task_loss(self,i,w):
        '''compute logistic loss for model w over data of task i'''
        return np.sum(np.log(1+np.exp(np.multiply(-self.tasks[i]['Y'], np.dot(self.tasks[i]['X'],w)))))

    def task_derivative(self,i,w):
        '''compute derivatives of logistic loss for model w over data of task i'''
        r = np.multiply(-self.tasks[i]['Y'],sigmoid(-np.multiply(self.tasks[i]['Y'],np.dot(self.tasks[i]['X'],w))))
        return np.matmul(self.tasks[i]['X'].T, r)


    def task_hessian(self, i, w):
        '''compute Hessian matrix of logistic loss for model w over data of task i'''
        a = [self.tasks[i]['Y'][j]*np.dot(w,self.tasks[i]['X'][j]) for j in range(self.tasks[i]['Y'].shape[0])]
        D = np.diag([sigmoid(-a[j])*sigmoid(a[j]) for j in range(len(a))])
        return np.matmul(self.tasks[i]['X'].T, np.matmul(D,self.tasks[i]['X']))


    def full_loss(self,W):
        '''compute total loss of MTL problem as sum of the losses of each task; each loss is normalized based on number of examples'''
        loss = 0
        for i in range(self.n_tasks):
            loss += self.task_loss(i, W[i])/self.tasks[i]['Y'].shape[0]
        return loss

    def jacobian(self,W):
        '''compute first order derivatives of full_loss w.r.t. all weights W'''
        J = np.zeros_like(W)
        for i in range(self.n_tasks):
            J[i] = self.task_derivative(i,W[i])/self.tasks[i]['Y'].shape[0]
        return J

def load_problem(directory, prob_name):
    '''load portfolio optimization problem'''
    
    path = directory + prob_name
    expectation_file = path + '/' + prob_name + 'expected_returns.mat'
    expected_returns = scipy.io.loadmat(expectation_file)
    c = -expected_returns['expected_returns']
    c =np.squeeze(c)

    variance_file = path + '/' + prob_name + 'variance_covariance.mat'
    variance_covariance = scipy.io.loadmat(variance_file)
    Q = variance_covariance['variance_covariance']


    return Q, c

def load_landmine(directory):
    '''load and parse multi-task regression dataset landmine'''
    
    path = directory + 'LandmineData/LandmineData.mat'
    data = scipy.io.loadmat(path)
    tasks = []
    for i in range(data['feature'][0].shape[0]):
        X = data['feature'][0][i]
        Y = data['label'][0][i].astype(int)
        Y = Y.squeeze()
        Y = 2*Y - 1
        tasks.append({'X':X, 'Y':Y})

    return tasks




def make_random_psd_matrix(size, ncond=None, eigenvalues=None):
    '''generate random positve semidefinite matrix given desired condition number 
    or eigenvalues according to equations (6.2)-(6.3) in [Kanzow and Lapucci,
    "Inexact Penalty Decomposition Methods for Optimization Problems with Geometric 
    Constraints" (2023)]'''
    
    y = np.random.uniform(low=-1, high=1, size=(size,1))
    if ncond:
        ncond_ = np.log(ncond)
        d = np.array([np.exp((i)/(size-1)*ncond_) for i in range(size)])
    elif eigenvalues:
        d = np.array(eigenvalues)
    D = np.diag(d)
    Y = np.identity(size) - 2/(np.linalg.norm(y)**2)*(y@y.T)
    return Y@D@Y


def generate_QP_benchmark():
    '''generate benchmark of sparse QP problems'''
    
    for n in [10, 100, 500]:
        for size in [20, 40, 60]:
            for i in range(10):
                generate_QP_problem(size, '{}'.format(i), ncond=n)


def generate_QP_problem(size, idf, ncond=1):
    '''generate random QP problem given size and condition number of Hessian matrix'''
    
    Q = make_random_psd_matrix(size, ncond=ncond)
    c = np.random.uniform(low=-1, high=1, size=size)
    P = {'Q': Q, 'c':c}
    with open('problems2/QP_{}_{}_{}.pkl'.format(size,ncond,idf), 'wb') as output:
            pickle.dump(P, output, pickle.HIGHEST_PROTOCOL)
            
def load_QP_problem(path):
    '''load previously saved QP problem (pickle format)'''
    
    P = pickle.load(open(path, "rb"))
    return P['Q'], P['c']



def make_perf_profile(scores, max_tau=100, step=0.01, labels=None, metric_name='runtime'):
    '''plot performance profiles of scores (a list of metric values for each considered algorithm);
    max_tau and step denote the scale and the stepsizes on the x axis'''
    
    n_solv = len(scores)
    n_probs = len(scores[0])

    taus = [x * step for x in range(int(1/step), int(max_tau*(1/step)))]




    if not labels:
        labels = ['']*n_solv

    perfs = []
    for i in range(n_solv):
        perfs.append({})

        for tau in taus:
            perfs[i][tau] = 0

    counted = 0
    for j in range(n_probs):
        ref = min([scores[i][j] for i in range(n_solv)])
        for tau in taus:
            for i in range(n_solv):
                if(scores[i][j]/ref <= tau):
                    perfs[i][tau] += 1

    for tau in taus:
        for i in range(n_solv):
            perfs[i][tau] /= n_probs

    for i in range(n_solv):
        print('{} wins {} times'.format(labels[i], perfs[i][taus[0]]))



    f, ax = plt.subplots(1,1)
    ax.set_xlim([1,max_tau])
    ax.set_ylim([-0.05,1.05])

    plt.xscale('log')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    linestyles = ['--', '-.', ':']
    markers = ['^', 'o',  's', '+', 'x', 'd', 'v', '<', '>']
    markers_cycler = itertools.cycle(markers)
    linestyle_cycler = itertools.cycle(linestyles)

    ax.set_xlabel('performance ratio - {}'.format(metric_name), fontsize=13)
    ax.set_ylabel('fraction of problems', fontsize=13)


    for i in range(n_solv):
        lists = sorted(perfs[i].items()) # sorted by key, return a list of tuples
        x, y = zip(*lists,)

        ax.plot(x,y,next(markers_cycler), markersize=8,
                    markevery=0.15, linestyle=next(linestyle_cycler),label=labels[i])


    plt.legend(loc=4)
    f.savefig('pp_{}.pdf'.format(metric_name))


def make_fval_profile(scores, max_tau=100, step=0.01, labels=None, metric_name='f_val'):
    '''plot cumulative distribution of absolute distance from optimum for group of solvers
    on problems benchmark; scores scores contains a list of metric values for each considered algorithm;
    max_tau and step denote the scale and the stepsizes on the x axis'''
    
    n_solv = len(scores)
    n_probs = len(scores[0])

    taus = [x * step for x in range(int(max_tau*(1/step)))]




    if not labels:
        labels = ['']*n_solv

    perfs = []
    for i in range(n_solv):
        perfs.append({})

        for tau in taus:
            perfs[i][tau] = 0

    counted = 0
    for j in range(n_probs):
        ref = min([scores[i][j] for i in range(n_solv)])
        for tau in taus:
            for i in range(n_solv):
                if(np.abs(scores[i][j]-ref)/np.abs(ref) <= tau):
                    perfs[i][tau] += 1

    for tau in taus:
        for i in range(n_solv):
            perfs[i][tau] /= n_probs

    for i in range(n_solv):
        print('{} wins {} times'.format(labels[i], perfs[i][taus[0]]))



    f, ax = plt.subplots(1,1)
    ax.set_xlim([0.01,max_tau])
    ax.set_ylim([-0.05,1.05])

    plt.xscale('log')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([0.05, 0.1,  0.5,  1, 5, 10, 50])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    linestyles = ['--', '-.', ':']
    markers = ['^', 'o',  's', '+', 'x', 'd', 'v', '<', '>']
    markers_cycler = itertools.cycle(markers)
    linestyle_cycler = itertools.cycle(linestyles)

    ax.set_xlabel('t', fontsize=13)
    ax.set_ylabel('p(rel_gap)<t', fontsize=13)


    for i in range(n_solv):
        if labels[i] == 'MIP':
            continue
        else:
            lists = sorted(perfs[i].items()) # sorted by key, return a list of tuples
            x, y = zip(*lists,)

            ax.plot(x,y,next(markers_cycler), markersize=8,
                    markevery=0.15, linestyle=next(linestyle_cycler),label=labels[i])


    plt.legend(loc=4)
    f.savefig('pp_{}.pdf'.format(metric_name))


def construct_corr_P1(size):
    '''construct matrix for nearest low-rank correlation matrix test problem'''
    
    C = np.identity(size)
    for i in range(size):
        for j in range(size):
            C[i,j] = 0.5+0.5*np.exp(-0.05*np.abs(i-j))
    return C, 'P1'


def construct_corr_P2(size):
    '''construct matrix for nearest low-rank correlation matrix test problem'''
    
    C = np.identity(size)
    for i in range(size):
        for j in range(size):
            C[i,j] = np.exp(-np.abs(i-j))
    return C, 'P2'


def construct_corr_P3(size):
    '''construct matrix for nearest low-rank correlation matrix test problem'''
    
    C = np.identity(size)
    for i in range(size):
        for j in range(size):
            C[i,j] = 0.6+0.4*np.exp(-0.1*np.abs(i-j))
    return C, 'P3'
