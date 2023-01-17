import numpy as np
from Solvers import PDSolver, SpectralSolver, MIPSolverPortfolio, SingleTaskSolver, NonlinearEnumerationSolver, DifficultEnumerationSolver
from Problems import SparsePortfolioProblem, SparseQuadraticProblem, NearestCorrelationMatrixProblem, ExplicitCorrelationProblem, MTLRProblem, NonlinearDisconnectedProblem, DifficultNDP
from utils import load_problem, load_QP_problem, make_perf_profile, make_fval_profile, load_landmine, MTLProblem, construct_corr_P1, construct_corr_P2, construct_corr_P3
import os
import pickle
from scipy.optimize import minimize

def test_inner_solvers():
    '''Experiment of Table 6.1 from Kanzow and Lapucci (2023)'''
    
    ncond = 10
    problems = []
    names = []
    lam = 5
    
    methods = ['gd', 'scipy_qn', 'scipy_lbfgs']
    
    for n in [10, 25, 50]:   
        prob_name = 'problems/QP_{}_{}_0.pkl'.format(n, ncond)
        Q, c = load_QP_problem(prob_name)
        problems.append(SparseQuadraticProblem(s=3, Q=Q, c=c, lam=lam))
        names.append(prob_name)
    for k, p in enumerate(problems):
        print('=================================')
        print('Problem {}'.format(names[k]))
        print('=================================')
        for t, met in enumerate(methods):
            print('ALGORITHM: {}'.format(met))
            print('=================================')
            solv = PDSolver(problem=p, method=met)
            solv.solve()
            print('=================================')
            
        print('ALGORITHM: {}'.format('PD_multipliers_lbfgs'))
        print('=================================')
        solv = PDSolver(problem=p, method='scipy_lbfgs', multipliers=True)
        solv.solve()
        print('=================================')
        print('ALGORITHM: {}'.format('ALM'))
        print('=================================')
        solv = SpectralSolver(problem=p, tau0=0)
        solv.solve()
        print('=================================')
        
        



def test_convergence():
    '''Experiment of Table 6.2 in Kanzow and Lapucci (2023)'''
    
    Q = np.ones((5,5)) + np.eye(5)
    c = np.array([-3,-2,-3,-12,-5])
    s = 2
    print(Q,c)
    
    P2 = SparseQuadraticProblem(s=s, Q=Q, c=c)
    
    sol_opt = np.array([0, -8/3, 0, 22/3, 0])
    print('f* = {}'.format(P2.f(sol_opt)))
    f_star = P2.f(sol_opt)
    count_conv = 0
    count_39 = 0
    count_36 = 0
    count_conv_mult = 0
    count_39_mult = 0
    count_36_mult = 0
    
    for k in range(1000):
        tau0, tau_gr = 10, 1.1
        P2.set_x0(np.random.uniform(-10,10,size=len(c)))

        SolverPd = PDSolver(problem=P2, method='scipy_lbfgs', tau0=tau0, tau_gr=tau_gr, debug=False)
        SolverPdM = PDSolver(problem=P2, method='scipy_lbfgs', multipliers=True, tau_gr=tau_gr, tau0=tau0, debug=False)
        sol = SolverPd.solve()['sol']
        f = P2.f(sol)
        if np.abs(f-f_star) <= 1e-2:
            count_conv += 1
        elif np.abs(f+39) <= 1e-2:
            count_39 += 1
        elif np.abs(f+36.3333) <= 1e-2:
            count_36+=1
        sol_mult = SolverPdM.solve()['sol']
        f_mult = P2.f(sol_mult)
        if np.abs(f_mult-f_star) <= 1e-2:
            count_conv_mult += 1
        elif np.abs(f_mult+39) <= 1e-2:
            count_39_mult += 1  
        elif np.abs(f_mult+36.3333) <= 1e-2:
            count_36_mult+=1

    print('PD: 41.33: {}, 39: {}, 36.33:{}, else: {}'.format(count_conv, count_39, count_36, 1000-count_conv-count_39-count_36))
    print('PDLM: 41.33: {}, 39: {}, 36.33:{}, else: {}'.format(count_conv_mult, count_39_mult, count_36_mult, 1000-count_conv_mult-count_39_mult-count_36_mult))

def test_SparseQP():
    '''Experiment in Figure 6.1 of Kanzow and Lapucci (2023)'''
    
    res_list = []
    res_header = ('filename', 'MIP_val', 'PD_val', 'PD_time', 'PDLM_val', 'PDLM_time', 'ALM_val', 'ALM_time')
    D = {'header':res_header}
    for filename in os.listdir('problems2/'):
        print(filename)
        P_data = pickle.load(open('problems2/'+filename, "rb"))

        Q, c = P_data['Q'], P_data['c']


        P = SparsePortfolioProblem(s=int(len(c)/7)+1, Q=Q, c=c, lam=1)

        SolverMIP = MIPSolverPortfolio(problem=P)
        SolverPd = PDSolver(problem=P, method='scipy_lbfgs')
        SolverPdM = PDSolver(problem=P, method='scipy_lbfgs', multipliers=True)
        SolverPSM = SpectralSolver(problem=P)

        print('=================')
        print('Problem {}'.format(filename))
        print('=================')
        print('MIP')
        mip_dict = SolverMIP.solve()
        print('=================')
        print('PD')
        pd_dict = SolverPd.solve()
        print('=================')
        print('PD - multipliers')
        pdlm_dict = SolverPdM.solve()
        print('=================')
        print('ALM')
        alm_dict = SolverPSM.solve()
        print('=================')
        data = (filename, mip_dict['f'], pd_dict['f'], pd_dict['t'], pdlm_dict['f'], pdlm_dict['t'], alm_dict['f'], alm_dict['t'])
        res_list.append(data)


    D['data'] = res_list
    
    # UNCOMMENT TO SAVE RESULTS  
    # with open('results_sparse.pkl', 'wb') as output:
    #     pickle.dump(D, output, pickle.HIGHEST_PROTOCOL)


def test_portfolio():
    '''Experiment of Table 6.3 in Kanzow and Lapucci (2023)'''
    
    problems = ['DTS1', 'DTS2', 'DTS3', 'FF10', 'FF17', 'FF48']
    spar = [2, 4, 6, 2, 2, 5]
    lams = [1e-3, 1e-3, 1e-3, 0.05, 0.05, 0.05]
    tau0 = 1e-2
    tau_gr = 1.01
    tau0alm = 1
    tau_gr_alm = 1.01
    for t, pn in enumerate(problems):
        Q, c = load_problem('datasets/', pn)
        P = SparsePortfolioProblem(s=spar[t], Q=Q, c=c, lam=lams[t])

        SolverMIP = MIPSolverPortfolio(problem=P)
        SolverPd = PDSolver(problem=P, method='scipy_lbfgs', tau0=tau0, tau_gr=tau_gr)
        SolverPdM = PDSolver(problem=P, method='scipy_lbfgs', multipliers=True, tau0=tau0, tau_gr=tau_gr)
        SolverPSM = SpectralSolver(problem=P, tau0=tau0alm, tau_gr=tau_gr_alm)

        print('=================')
        print('Problem {}'.format(pn))
        print('=================')
        print('MIP')
        SolverMIP.solve()
        print('=================')
        print('PD')
        SolverPd.solve()
        print('=================')
        print('PD - multipliers')
        SolverPdM.solve()
        print('=================')
        print('ALM')
        SolverPSM.solve()
        print('=================')


def perf_prof_ccqp(metric):
    '''extract performance profiles for results of test_sparse_QP'''
    
    D = pickle.load(open('results_sparse.pkl', "rb"))

    results = D['data']
    pd = []
    pdlm = []
    alm = []
    mip = []
    labels = ['MIP', 'PD', 'PDLM', 'ALM'] if metric == 'f_val' else ['PD', 'PDLM', 'ALM']
    for d in results:
        if metric == 'runtime':
            pd.append(d[3])
            pdlm.append(d[5])
            alm.append(d[7])
        elif metric == 'f_val':
            mip.append(d[1])
            pd.append(d[2])
            pdlm.append(d[4])
            alm.append(d[6])

    if metric == 'runtime':
        scores = [pd, pdlm, alm]
        make_perf_profile(scores, labels=labels, metric_name=metric, max_tau=20)
    if metric == 'f_val':
        scores = [mip, pd, pdlm, alm]
        make_fval_profile(scores, labels=labels, metric_name=metric, max_tau=50)





def test_correlation():
    '''Experiements for Tables 6.4-6.5-6.6 from Kanzow and Lapucci (2023)'''
    
    for r in [5,10,20]:
        for n in [200, 500]:
            P1 = construct_corr_P1(n)
            P2 = construct_corr_P2(n)
            P3 = construct_corr_P3(n)

            for p in [P1, P2, P3]:
                P = NearestCorrelationMatrixProblem(s=r,C=p[0])
                PE = ExplicitCorrelationProblem(s=r,C=p[0])
                print('========================================================================================')
                print('PROBLEM {} - n = {} - rank = {}'.format(p[1],n,r))
                print('PD_inexact_inacc_no_mult')
                S = PDSolver(problem=P, method='scipy_cg_inexact', multipliers=False, debug=False, tau_gr=1.2, tau_max=1e12)
                S.solve()

                print('PD_inexact_inacc_mult')
                S = PDSolver(problem=P, method='scipy_cg_inexact', multipliers=True, no_eq_mult=True, debug=False, tau_gr=1.2, tau_max=1e12)
                S.solve()

                print('PD_inexact_acc_no_mult')
                S = PDSolver(problem=P, method='scipy_cg', multipliers=False, debug=False, tau_gr=1.2, tau_max=1e12)
                S.solve()

                print('PD_inexact_acc_mult')
                S = PDSolver(problem=P, method='scipy_cg', multipliers=True, no_eq_mult=True, debug=False, tau_gr=1.2, tau_max=1e12)
                S.solve()


                print('PD_exact_no_mult')
                S = PDSolver(problem=P, method='exact_correlation_constraint', multipliers=False, debug=False, tau_gr=1.2, tau_max=1e12)
                S.solve()

                print('PD_exact_mult')
                S = PDSolver(problem=P, method='exact_correlation_constraint', multipliers=True, no_eq_mult=True, debug=False, tau_gr=1.2, tau_max=1e12)
                S.solve()



                print('PD_exact_explicit')
                S = PDSolver(problem=PE, method='exact_correlation', multipliers=False, debug=False, tau_gr=1.2, tau_max=1e12)
                S.solve()

                print('ALM')
                S2 = SpectralSolver(problem=P, debug=False, tau_gr=1.2, tau_max=1e12)
                S2.solve()


def test_landmine():
    '''Experiment for Figure 6.2 in Kanzow and Lapucci (2023)'''
    
    tasks = load_landmine('./')
    P = MTLProblem(tasks)

    pre_S = SingleTaskSolver(P, nu=0)
    W0 = pre_S.solve()
    WUV0 = np.zeros((W0.shape[0],W0.shape[1]*3))
    WUV0[:,:W0.shape[1]] = W0
    WUV0[:,W0.shape[1]:W0.shape[1]*2] = W0
    for nu in [2,0.5,0.1,0.01]:
        print('NU:', nu)
        mtlp = MTLRProblem(2,P,nu=nu)
        mtlp.set_x0(WUV0)

        print('Single task:')
        S = SingleTaskSolver(P, nu=nu)
        S.solve(verbose=False)

        print('PD_inexact:')
        S = PDSolver(problem=mtlp, method='scipy_cg_inexact', multipliers=True, no_eq_mult=False, debug=False, tau_gr=1.3, tau0=1e-3)
        S.solve()

        print('PD_mid:')
        S = PDSolver(problem=mtlp, method='scipy_cg_mid', multipliers=True, no_eq_mult=False, debug=False, tau_gr=1.3, tau0=1e-3)
        S.solve()

        print('PD_accurate:')
        S = PDSolver(problem=mtlp, method='scipy_cg', multipliers=True, no_eq_mult=False, debug=False, tau_gr=1.3, tau0=1e-3)
        S.solve()


        print('ALM_fast:')
        S2 = SpectralSolver(problem=mtlp, debug=False, tau_gr=1.3, m=1, max_sgm_iters=30, gamma_max=1e6, inner_eps=1e-1, sigma=5e-2)
        S2.solve()


        print('ALM_accurate:')
        S2 = SpectralSolver(problem=mtlp, debug=False, tau_gr=1.3, m=4, max_sgm_iters=800, gamma_max=1e9, inner_eps=1e-3, sigma=5e-4)
        S2.solve()


def test_constrained_logistic():
    '''Experiment for Table 6.7 in Kanzow and Lapucci (2023)'''
    
    n_feats = 10
    n_ex = 200
    balance_factor = 1
    nA = 12

    for N in [2,5,10,20,50,100]:
        feasible = False
        print('{} DISJOINT SETS'.format(N))
        while not feasible:
            X = np.random.rand(n_ex,n_feats)
            Y = np.random.randint(2, size=n_ex)
            Y = 2*Y -1
            As = []
            bs = []
            for i in range(N):
                A = np.random.uniform(low=-1,high=1,size=(nA,n_feats))
                b = np.random.uniform(low=-1,high=1,size=(nA))
                As.append(A)
                bs.append(b)

            As = np.array(As)

            coeffs = balance_factor*np.random.rand(n_feats)
            t = 0.1*balance_factor
            
            point = np.random.uniform(low=-0.5,high=0.5,size=n_feats)
            P = NonlinearDisconnectedProblem(As,bs,X,Y,coeffs,t,point)
            def objf(a):
                return P.f(a)

            unconstrained_min = minimize(objf, P.get_x0(), method='CG').x


            P.set_x0(point)
            SE = NonlinearEnumerationSolver(P)
            feasible = SE.solve()
            
        print('Unconstrained minimizer:', objf(unconstrained_min))

        print('Penalty Decomposition')
        S = PDSolver(problem=P, method='scipy_cg', multipliers=True, no_eq_mult=False, debug=False, tau_gr=1.2, tau0=1e-1, inner_eps=1e-2, tau_max=1e12)
        S.solve()


        print('ALM')
        S2 = SpectralSolver(problem=P, debug=False, tau_gr=1.2, m=4, max_sgm_iters=100, gamma_max=1e12, inner_eps=1e-2, sigma=1e-2)
        S2.solve()



def test_difficult_problem():
    '''Experiment for Figure 6.3 in Kanzow and Lapucci (2023)'''
    
    n_feats = 5
    n_ex = 200
    balance_factor = 1
    nA = 7
    ndc = 80
    t_val = 0.1

    count = 0
    while count < 20:
        for N in [50]:
            X = np.random.rand(n_ex,n_feats)
            Y = np.random.randint(2, size=n_ex)
            Y = 2*Y -1
            As = []
            bs = []
            print('{} DISJOINT SETS'.format(N))
            for i in range(N):
                A = np.random.uniform(low=-1,high=1,size=(nA,n_feats))
                b = np.random.uniform(low=-1,high=1,size=(nA))
                As.append(A)
                bs.append(b)

            As = np.array(As)

            coeffs = balance_factor*np.random.rand(ndc,n_feats)
            ts = t_val*np.ones(ndc)
            points = np.random.uniform(low=-0.5,high=0.5,size=(ndc,n_feats))
            P = DifficultNDP(As,bs,X,Y,coeffs,ts,points)
            def objf(a):
                return P.f(a)

            unconstrained_min = minimize(objf, P.get_x0(), method='CG').x
            print('Unconstrained minimizer:', objf(unconstrained_min))


            P.set_x0(unconstrained_min)
            SE = DifficultEnumerationSolver(P)
            feasible = SE.solve()
            if not feasible:
                print('UNFESIBLE')
                continue

            count += 1

            print('Penalty Decomposition')
            S = PDSolver(problem=P, method='scipy_cg', multipliers=True, no_eq_mult=False, debug=False, tau_gr=1.5, tau0=1e-1, inner_eps=2e-2, tau_max=1e12)
            S.solve()


            print('ALM')
            S2 = SpectralSolver(problem=P, debug=False, tau_gr=1.5, m=4, max_sgm_iters=100, gamma_max=1e12, inner_eps=1e-1, sigma=5e-2)
            S2.solve()










test_convergence()
# test_inner_solvers()
# test_portfolio()
# test_SparseQP()
# test_correlation()

# test_landmine()

# test_constrained_logistic()
# test_difficult_problem()




