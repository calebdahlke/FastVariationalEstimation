from operator import matmul
import numpy as np
import scipy.stats
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import time
import random
from matplotlib.patches import Ellipse
from sklearn import mixture
import matplotlib.transforms as transforms
from functools import partial
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

###################################################################################
###########################Outline#################################################
# 1) Generate a Gaussian Mixture Model Parameters
# 2) Numerically Integrate MI
#       a) Break into seperate Entropy terms
# 3) Create General Variational approx. computations
#       a) Marginal and Conditional
# 4) Compute Moment Matching
# 5) Compute Gradient Ascent
# 6) Compute Barber and Agakov
# 8) Plot Samples and approximation
###################################################################################

def GaussianMixtureParams(M,Dx,Dy):
    ###############################################################################
    # Outline: Randomly Generates Parameters for GMM
    #
    # Inputs:
    #       M - Number of components
    #       Dx - Number of dimensions for Latent Variable, X
    #       Dy - Number of dimensions for Observation Variable, Y
    #
    # Outputs:
    #       w - weights of components
    #       mu - means of components
    #       Sigma - Variance of components
    ###############################################################################
    D = Dx+Dy
    w = np.random.dirichlet(np.ones(M))
    mu = []
    sigma = []
    for d in range(M):
        mu.append(np.random.uniform(-5,5,(D,1)))
        A = np.random.rand(D, D)
        B = np.dot(A, A.transpose())
        sigma.append(B)
    return w,mu,sigma

def SampleGMM(N,w,mu,sigma):
    ###############################################################################
    # Outline: Samples Points from a GMM
    #
    # Inputs:
    #       N - Number of points to sample
    #       w - weights of GMM components
    #       mu - means of GMM components
    #       Sigma - Variance of GMM components
    #
    # Outputs:
    #       samples - coordniates of sampled points
    ###############################################################################
    samples = np.zeros((N,len(mu[0])))
    for j in range(N):
        acc_pis = [np.sum(w[:i]) for i in range(1, len(w)+1)]
        r = np.random.uniform(0, 1)
        k = 0
        for i, threshold in enumerate(acc_pis):
            if r < threshold:
                k = i
                break
        x = np.random.multivariate_normal(mu[k].T.tolist()[0],sigma[k].tolist())
        samples[j,:] = x
    return samples

def MargEntGMM(sample,Dx,w,mu,Sigma):
    ###############################################################################
    # Outline: Numerically Calculates Marginal Entropy
    #
    # Inputs:
    #       samples - List of full sample set
    #       Dx - Dimension of Latenat Variable, X
    #       w - weights of components
    #       mu - means of components
    #       Sigma - Variance of components
    #
    # Outputs:
    #       MargEnt - Marginal Entropy
    ###############################################################################
    M = len(w)
    x = sample[:,0:Dx]
    MargEntPart = np.zeros((M,len(sample[:,0])))
    for d in range(M):
        MargEntPart[d,:] = multivariate_normal.logpdf(x,mu[d][0:Dx].T.tolist()[0],Sigma[d][0:Dx,0:Dx])+np.log(w[d])
    MargEnt = -1*(1/len(sample[:,0]))*sum(logsumexp(MargEntPart,axis=0))
    return MargEnt

def CondEntGMM(sample,Dx,Dy,w,mu,Sigma):
    ###############################################################################
    # Outline: Numerically Calculates Marginal Entropy
    #
    # Inputs:
    #       samples - List of full sample set
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Obsevation Variable, Y
    #       w - weights of components
    #       mu - means of components
    #       Sigma - Variance of components
    #
    # Outputs:
    #       CondEnt - Conditional Entropy
    ###############################################################################
    M = len(w)
    y = sample[:,Dx:(Dx+Dy)]
    JointEntPart = np.zeros((M,len(sample[:,0])))
    MargEntPart = np.zeros((M,len(sample[:,0])))
    for d in range(M):
        JointEntPart[d,:] = multivariate_normal.logpdf(sample,mu[d].T.tolist()[0],Sigma[d])+np.log(w[d])
        MargEntPart[d,:] = multivariate_normal.logpdf(y,mu[d][Dx:(Dx+Dy)].T.tolist()[0],Sigma[d][Dx:(Dx+Dy),Dx:(Dx+Dy)])+np.log(w[d])
    JointEnt = -1*sum(logsumexp(JointEntPart,axis=0))*(1/len(sample[:,0]))
    MargEnt = -1*sum(logsumexp(MargEntPart,axis=0))*(1/len(sample[:,0]))
    CondEnt = JointEnt-MargEnt
    return CondEnt

def pmoments(sample,Dx,Dy):
    ###############################################################################
    # Outline: Calculates the Moments of a GMM
    #
    # Inputs:
    #       samples - List of full sample set
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Obsevation Variable, Y
    #
    # Outputs:
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPYY - Second moment of P w.r.t. Y
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    ###############################################################################
    EPX = sum(sample[:,0:Dx])/len(sample[:,0])
    EPY = sum(sample[:,Dx:(Dx+Dy)])/len(sample[:,0])
    EPXX = np.matmul(sample[:,0:Dx].T,sample[:,0:Dx])/len(sample[:,0])
    EPXY = np.matmul(sample[:,0:Dx].T,sample[:,Dx:(Dx+Dy)])/len(sample[:,0])
    EPYY = np.matmul(sample[:,Dx:(Dx+Dy)].T,sample[:,Dx:(Dx+Dy)])/len(sample[:,0])
    return EPX, EPY, EPXX, EPXY, EPYY

def MargEntMomentMatch(Dx,EPX, EPXX):
    ###############################################################################
    # Outline: Calculates Implicit Likelihood Variational Marginal Entropy by 
    #          Moment Matching
    #
    # Inputs:
    #       Dx - Dimension of Latenat Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #
    # Outputs:
    #       VarMargEnt - Implicit Likelihood Variational Marginal Entropy
    ###############################################################################
    sigmaqx = EPXX-np.outer(EPX,EPX)
    VarMargEnt = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(sigmaqx))+Dx)
    return VarMargEnt

def CondEntMomentMatch(Dx,EPX, EPY, EPXX, EPXY, EPYY):
    ###############################################################################
    # Outline: Calculates Implicit Likelihood Variational Conditional Entropy by 
    #          Moment Matching
    #
    # Inputs:
    #       Dx - Dimension of Latenat Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #
    # Outputs:
    #       VarCondEnt - Implicit Likelihood Variational Conditional Entropy
    ###############################################################################
    sigmaqx = EPXX-np.outer(EPX,EPX)
    sigmaqy = EPYY-np.outer(EPY,EPY)
    sigmaqxy = EPXY-np.outer(EPX,EPY)
    condsigma = sigmaqx-np.matmul(sigmaqxy,np.matmul(np.linalg.inv(sigmaqy),sigmaqxy.T))
    VarCondEnt = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(condsigma))+Dx)
    return VarCondEnt

def MargEntGradientDescent(Dx,EPX,EPXX,Tol,FullOut=False):
    ###############################################################################
    # Outline: Calculates Implicit Likelihood Variational Marginal Entropy by 
    #          Gradient Descent
    #
    # Inputs:
    #       Dx - Dimension of Latenat Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #       Tol - Tolerance for Gradient Descent
    #
    # Outputs:
    #       VarMarg.fun - Implicit Likelihood Variational Marginal Entropy
    ###############################################################################
    mux0 = np.zeros((Dx,1))
    chol_sigmax0 = np.eye(Dx) 
    Margx0 = np.concatenate((mux0.flatten(),chol_sigmax0.flatten()))
    
    conditions = np.eye(Dx)
    arr = -1*np.ones((Dx,Dx))
    conditions = conditions+np.triu(arr, 1)
    MargBound = []
    for i in range(len(mux0.flatten())):
        MargBound.append((-np.inf,np.inf))
    for j in range(len(conditions.flatten())):
        if (conditions==1).flatten()[j]:
            MargBound.append((.005,np.inf))
        elif(conditions==-1).flatten()[j]:
            MargBound.append((0,0))
        else:
            MargBound.append((-np.inf,np.inf))
    if FullOut == True:
        history = []
        def callback(x):
            fobj = GDMarg(x,Dx,EPX,EPXX)
            history.append(fobj)
        VarMarg = scipy.optimize.minimize(fun=GDMarg, x0=Margx0, args=(Dx,EPX,EPXX),bounds=tuple(MargBound), jac=MargDerivative,method='L-BFGS-B',tol=Tol, callback=callback) 
        VarMarg=history
    else:
        VarMarg = scipy.optimize.minimize(fun=GDMarg, x0=Margx0, args=(Dx,EPX,EPXX),bounds=tuple(MargBound), jac=MargDerivative,method='L-BFGS-B', tol=Tol) 
        VarMarg=VarMarg.fun
    return VarMarg

def GDMarg(params,Dx,EPX,EPXX):
    ###############################################################################
    # Outline: Implicit Likelihood Variational Approximation optimization function
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,chol_Sigma_x]
    #       Dx - Dimension of Latenat Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #
    # Outputs:
    #       Marg - Marginal Entropy evaluation
    ###############################################################################
    muq_x = params[0:Dx].reshape(Dx,1)
    
    chol_Sigmaq_x = params[Dx:]
    chol_Sigmaq_x = chol_Sigmaq_x.reshape(Dx,Dx)
    
    Sigmaq_x = np.matmul(chol_Sigmaq_x,chol_Sigmaq_x.T)
    
    Marg = EvalMarg(Dx,EPX,EPXX,muq_x,Sigmaq_x)
    return Marg.flatten()

def EvalMarg(Dx,EPX,EPXX,muq_x,Sigmaq_x):
    ###############################################################################
    # Outline: Evaluates the Variational Marginal Entropy
    #
    # Inputs:
    #       Dx - Dimension of Latenat Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #       muq_x - Mean of Latent Variable in the Variational Distribution
    #       Sigmaq_x -  Varaince of Latent Variable in Variational Distribution
    #
    # Outputs:
    #       Marg - Marginal Entropy evaluation
    ###############################################################################
    Sigmaq_x_inv = np.linalg.inv(Sigmaq_x)
    Marg  = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(Sigmaq_x))+\
        np.trace(np.matmul(Sigmaq_x_inv,(EPXX-np.outer(EPX,EPX))))+\
        np.matmul(EPX.T,np.matmul(Sigmaq_x_inv,EPX))-2*np.matmul(muq_x.T,np.matmul(Sigmaq_x_inv,EPX))+\
        np.matmul(muq_x.T,np.matmul(Sigmaq_x_inv,muq_x)))
    return Marg[0][0]

def MargDerivative(params,Dx,EPX,EPXX):
    ###############################################################################
    # Outline: Evaluates the Derivative Variational Marginal Entropy
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,chol_Sigma_x]
    #       Dx - Dimension of Latenat Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPXX - Second moment of P w.r.t. X
    #
    # Outputs:
    #       dparams - derivative of each parameter
    ###############################################################################
    muq_x = tf.Variable(tf.convert_to_tensor(params[0:Dx].reshape(Dx,1)), name='muq_x')
    chol_Sigmaq_x = tf.Variable(tf.convert_to_tensor(params[Dx:].reshape(Dx,Dx)), name='chol_Sigmaq_x')
    EPX = EPX.reshape((len(EPX),1))
    with tf.GradientTape(persistent=True) as tape:
        Sigmaq_x = tf.linalg.matmul(chol_Sigmaq_x,tf.transpose(chol_Sigmaq_x))
        Sigmaq_x_inv = tf.linalg.inv(Sigmaq_x)        
        Marg  = .5*(Dx*tf.math.log(2*np.pi)+tf.math.log(tf.linalg.det(Sigmaq_x))+\
            tf.linalg.trace(tf.linalg.matmul(Sigmaq_x_inv,(EPXX-np.outer(EPX,EPX))))+\
            tf.linalg.matmul(EPX.T,tf.linalg.matmul(Sigmaq_x_inv,EPX))-2*tf.linalg.matmul(tf.transpose(muq_x),tf.linalg.matmul(Sigmaq_x_inv,EPX))+\
            tf.linalg.matmul(tf.transpose(muq_x),tf.linalg.matmul(Sigmaq_x_inv,muq_x)))
    [dmu, dSigmachol] = tape.gradient(Marg, [muq_x, chol_Sigmaq_x])
    dmu = tf.make_ndarray(tf.make_tensor_proto(dmu))
    dSigmachol = tf.make_ndarray(tf.make_tensor_proto(dSigmachol))
    dparams = np.concatenate((dmu.flatten(),dSigmachol.flatten()))
    return dparams

def CondEntGradientDescent(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,Tol,FullOut=False):
    ###############################################################################
    # Outline: Calculates Implicit Likelihood Variational Conditional Entropy by 
    #          Gradient Descent
    #
    # Inputs:
    #       Dx - Dimension of Latenat Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       Tol - Tolerance for Gradient Descent
    #
    # Outputs:
    #       VarCond.fun - Implicit Likelihood Variational Conditional Entropy
    ###############################################################################
    b = np.zeros((Dx,1))
    A = np.matmul((EPXY-np.outer(EPX,EPY)),np.linalg.inv(EPYY-np.outer(EPY,EPY)))
    Sigma = np.eye(Dx)
    Condx0 = np.concatenate((np.concatenate((A.flatten(),b.flatten())),Sigma.flatten()))
    
    conditions = np.eye(Dx)
    arr = -1*np.ones((Dx,Dx))
    conditions = conditions+np.triu(arr, 1)
    CondBound = []
    for i in range(len(A.flatten())+len(b.flatten())):
        CondBound.append((-np.inf,np.inf))

    for j in range(len(conditions.flatten())):
        if (conditions==1).flatten()[j]:
            CondBound.append((0.005,np.inf))
        elif(conditions==-1).flatten()[j]:
            CondBound.append((0,0))
        else:
            CondBound.append((-np.inf,np.inf))
            #(EPXX-np.outer(EPX,EPX))-np.matmul((EPXY-np.outer(EPX,EPY)),np.matmul(EPYY-np.outer(EPY,EPY),((EPXY-np.outer(EPX,EPY)).T)))
    if FullOut == True:
        history = []
        def callback(x):
            fobj = GDCond(x,Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY)
            history.append(fobj)
        VarCond = scipy.optimize.minimize(fun=GDCond, x0=Condx0, args=(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY),bounds=tuple(CondBound),jac=CondDerivative, method='L-BFGS-B', tol=Tol,callback=callback)
        VarCond=history
    else:
        VarCond = scipy.optimize.minimize(fun=GDCond, x0=Condx0, args=(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY),bounds=tuple(CondBound),jac=CondDerivative, method='L-BFGS-B', tol=Tol)
        VarCond = VarCond.fun
    return VarCond

def GDCond(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY):
    ###############################################################################
    # Outline: Implicit Likelihood Variational Approximation optimization function
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,mu_y,chol_Sigma_joint]
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #
    # Outputs:
    #       Cond - Conditional Entropy evaluation
    ###############################################################################

    A = params[0:Dx*Dy].reshape(Dx,Dy)
    b = params[Dx*Dy:(Dx*Dy+Dx)].reshape(Dx,1)
    
    chol_Sigmaq_joint = params[(Dx*Dy+Dx):]
    chol_Sigmaq_joint = chol_Sigmaq_joint.reshape((Dx),(Dx))

    Sigmaq = np.matmul(chol_Sigmaq_joint,chol_Sigmaq_joint.T)
    
    Cond = EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,A,b,Sigmaq)
    return Cond.flatten()

def EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,A,b,Sigmaq):
    ###############################################################################
    # Outline: Evaluates the Variational Conditional Entropy
    #
    # Inputs:
    #       Dx - Dimension of Latenat Variable, X
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       muq_x - Mean of Latent Variable in the Variational Distribution
    #       muq_y - Mean of Observation Variable in the Variational Distribution
    #       Sigmaq_x -  Varaince of Latent Variable in Variational Distribution
    #       Sigmaq_xy -  Covariance in Variational Distribution
    #       Sigmaq_y -  Varaince of Observation Variable in Variational Distribution
    #
    # Outputs:
    #       Cond - Conditional Entropy evaluation
    ###############################################################################
    Sigmaq_inv = np.linalg.inv(Sigmaq)
    A1 = np.matmul(Sigmaq_inv,A)
    B1 = np.matmul(A.T,A1)
    EPX = EPX.reshape((len(EPX),1))
    EPY = EPY.reshape((len(EPY),1))
    #
    # Cond  = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(Sigmaq))+\
    #     np.trace(np.matmul(Sigmaq_inv,(EPXX-np.outer(EPX,EPX))))+\
    #     np.matmul(EPX.T,np.matmul(Sigmaq_inv,EPX))-\
    #     2*(np.matmul(b.T,np.matmul(Sigmaq_inv,EPX)) +\
    #     np.trace(np.matmul(A1,(EPXY-np.outer(EPX,EPY)).T)) +np.matmul(EPX.T,np.matmul(A1,EPY)))+\
    #     np.trace(np.matmul(B1,(EPYY-np.outer(EPY,EPY))))+np.matmul(EPY.T,np.matmul(B1,EPY))+\
    #     2*np.matmul(b.T,np.matmul(A1,EPY))+np.matmul(b.T,np.matmul(Sigmaq_inv,b)))
    
    Cond  = .5*(Dx*np.log(2*np.pi)+np.log(np.linalg.det(Sigmaq))+\
        np.trace(np.matmul(Sigmaq_inv,EPXX))-\
        2*(np.matmul(b.T,np.matmul(Sigmaq_inv,EPX)) +\
        np.trace(np.matmul(A1,EPXY.T)))+\
        np.trace(np.matmul(B1,EPYY))+\
        2*np.matmul(b.T,np.matmul(A1,EPY))+np.matmul(b.T,np.matmul(Sigmaq_inv,b)))

    return Cond[0][0]

def CondDerivative(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY):
    ###############################################################################
    # Outline: Evaluates the Derivative Variational Conditional Entropy
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,mu_y,chol_Sigma_joint]
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #
    # Outputs:
    #       dparams - derivative of each parameter
    ###############################################################################
    A = tf.Variable(tf.convert_to_tensor(params[0:Dx*Dy].reshape(Dx,Dy)), name='A')
    b = tf.Variable(tf.convert_to_tensor(params[Dx*Dy:(Dx*Dy+Dx)].reshape(Dx,1)), name='b')
    
    chol_Sigmaq_joint = tf.Variable(tf.convert_to_tensor(params[(Dx*Dy+Dx):].reshape(Dx,Dx)), name='chol_Sigmaq_joint')
    EPX = EPX.reshape((len(EPX),1))
    EPY = EPY.reshape((len(EPY),1))
    
    with tf.GradientTape(persistent=True) as tape:
        Sigmaq = tf.linalg.matmul(chol_Sigmaq_joint,tf.transpose(chol_Sigmaq_joint))
        Sigmaq_inv = tf.linalg.inv(Sigmaq)
        A1 =tf.linalg.matmul(Sigmaq_inv,A)
        B1 = tf.linalg.matmul(tf.transpose(A),A1)
        #
        # Cond  = .5*(Dx*tf.math.log(2*np.pi)+tf.math.log(tf.linalg.det(Sigmaq))+\
        #     tf.linalg.trace(tf.linalg.matmul(Sigmaq_inv,(EPXX-np.outer(EPX,EPX))))+\
        #     tf.linalg.matmul(EPX.T,tf.linalg.matmul(Sigmaq_inv,EPX))-\
        #     2*(tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(Sigmaq_inv,EPX)) +\
        #     tf.linalg.trace(tf.linalg.matmul(A1,tf.transpose((EPXY-np.outer(EPX,EPY))))) +tf.linalg.matmul(EPX.T,tf.linalg.matmul(A1,EPY)))-\
        #     tf.linalg.trace(tf.linalg.matmul(B1,(EPYY-np.outer(EPY,EPY))))+tf.linalg.matmul(EPY.T,tf.linalg.matmul(B1,EPY))+\
        #     2*tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(A1,EPY))+tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(Sigmaq_inv,b)))
        
        Cond  = .5*(Dx*tf.math.log(2*np.pi)+tf.math.log(tf.linalg.det(Sigmaq))+\
            tf.linalg.trace(tf.linalg.matmul(Sigmaq_inv,(EPXX)))-\
            2*(tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(Sigmaq_inv,EPX)) +\
            tf.linalg.trace(tf.linalg.matmul(A1,tf.transpose(EPXY))))+\
            tf.linalg.trace(tf.linalg.matmul(B1,(EPYY)))+\
            2*tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(A1,EPY))+tf.linalg.matmul(tf.transpose(b),tf.linalg.matmul(Sigmaq_inv,b)))

    [dA,db, dSigmachol] = tape.gradient(Cond, [A, b,chol_Sigmaq_joint])
    dA = tf.make_ndarray(tf.make_tensor_proto(dA))
    db = tf.make_ndarray(tf.make_tensor_proto(db))
    dSigmachol = tf.make_ndarray(tf.make_tensor_proto(dSigmachol))
    dparams = np.concatenate((dA.flatten(),np.concatenate((db.flatten(),dSigmachol.flatten()))))
    return dparams

def MIOpt(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const,Tol):
    ###############################################################################
    # Outline: Calculates an Optimal Variational Distribution that matches the 
    #          True Distributions Mutual Information
    #
    # Inputs:
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       const - value of True MI
    #       Tol - Tolerance for Gradient Descent
    #
    # Outputs:
    #       mux - Mean of Latent Variable of Variational Distribution
    #       muy - Mean of Observation Variable of Variational Distribution
    #       Sigmax - Variance of Latent Variable of Variational Distribution
    #       Sigmaxy - Covariance of Varaitional Distribution 
    #       Sigmay - Variance of Latent Variable of Variational Distribution
    ###############################################################################
    mux0_joint = np.zeros((Dx+Dy,1))
    sigmax0_joint = np.zeros((Dx+Dy,Dx+Dy))

    mux0_joint[0:Dx] = EPX.reshape((Dx,1))
    mux0_joint[Dx:(Dx+Dy)] = EPY.reshape((Dy,1))
    sigmax0_joint[0:Dx,0:Dx] = EPXX-np.outer(EPX,EPX)
    sigmax0_joint[0:Dx,Dx:(Dx+Dy)] = .9*(EPXY-np.outer(EPX,EPY))
    sigmax0_joint[Dx:(Dx+Dy),0:Dx] = .9*(EPXY-np.outer(EPX,EPY)).T
    sigmax0_joint[Dx:(Dx+Dy),Dx:(Dx+Dy)] = EPYY-np.outer(EPY,EPY)
    chol_sigmax0_joint = np.linalg.cholesky(sigmax0_joint)

    Condx0 = np.concatenate((mux0_joint.flatten(),chol_sigmax0_joint.flatten()))
    
    conditions = np.eye(Dx+Dy)
    arr = -1*np.ones((Dx+Dy,Dx+Dy))
    conditions = conditions+np.triu(arr, 1)
    CondBound = []
    for i in range(len(mux0_joint.flatten())):
        CondBound.append((-np.inf,np.inf))

    for j in range(len(conditions.flatten())):
        if (conditions==1).flatten()[j]:
            CondBound.append((0.01,np.inf))
        elif(conditions==-1).flatten()[j]:
            CondBound.append((0,0))
        else:
            CondBound.append((-np.inf,np.inf))
            
    VarCond = scipy.optimize.minimize(fun=GDMIOPT, x0=Condx0, args=(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const),bounds=tuple(CondBound),jac=GDMIOPTDerivs, method='L-BFGS-B', tol=Tol)
    mux = VarCond.x[0:Dx].reshape(Dx,1)
    muy = VarCond.x[Dx:(Dx+Dy)].reshape(Dy,1)
    cholSigma = VarCond.x[(Dx+Dy):].reshape((Dx+Dy),(Dx+Dy))
    Sigma = np.matmul(cholSigma,cholSigma.T)
    Sigmax = Sigma[0:Dx,0:Dx]
    Sigmaxy = Sigma[0:Dx,Dx:(Dx+Dy)]
    Sigmay = Sigma[Dx:(Dx+Dy),Dx:(Dx+Dy)]
    return mux,muy,Sigmax,Sigmaxy,Sigmay

def GDMIOPT(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const):
    ###############################################################################
    # Outline: Evaluates the Absolute Error of the Difference of True and Variational MI
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,mu_y,chol_Sigma_joint]
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       const - value of True MI
    #
    # Outputs:
    #       MIN - Value of difference of True and Variational MI
    ###############################################################################
    muq_x = params[0:Dx].reshape(Dx,1)
    muq_y = params[Dx:(Dx+Dy)].reshape(Dy,1)

    chol_Sigmaq_joint = params[(Dx+Dy):]
    chol_Sigmaq_joint = chol_Sigmaq_joint.reshape((Dx+Dy),(Dx+Dy))
    
    Sigmaq_joint = np.matmul(chol_Sigmaq_joint,chol_Sigmaq_joint.T)
    Sigmaq_x = Sigmaq_joint[0:Dx,0:Dx]
    Sigmaq_xy = Sigmaq_joint[0:Dx,Dx:(Dx+Dy)]
    Sigmaq_y = Sigmaq_joint[Dx:(Dx+Dy),Dx:(Dx+Dy)]
    
    OptMarg = EvalMarg(Dx,EPX,EPXX,muq_x,Sigmaq_x)
    OptCond = EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,muq_x,muq_y,Sigmaq_x,Sigmaq_xy,Sigmaq_y)
    
    MI = OptMarg-OptCond
    MIN = np.abs(MI-const)
    return MIN.flatten()

def GDMIOPTDerivs(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const):
    ###############################################################################
    # Outline: Evaluates the Derivative of Absolute Error of the Difference ofTrue and Variational MI
    #
    # Inputs:
    #       params - Parameters of varaitiona distribution [mu_x,mu_y,chol_Sigma_joint]
    #       Dx - Dimension of Latenat Variable, X
    #       Dy - Dimension of Observation Variable, Y
    #       EPX - First moment of P w.r.t. X
    #       EPY - First moment of P w.r.t. Y
    #       EPXX - Second moment of P w.r.t. X
    #       EPXY - Second moment of P w.r.t. Corss Term XY
    #       EPYY - Second moment of P w.r.t. Y
    #       const - value of True MI
    #
    # Outputs:
    #       dparams - derivative of each parameter
    ###############################################################################
    muq_x = params[0:Dx].reshape(Dx,1)
    muq_y = params[Dx:(Dx+Dy)].reshape(Dy,1)

    chol_Sigmaq_joint = params[(Dx+Dy):]
    chol_Sigmaq_joint = chol_Sigmaq_joint.reshape((Dx+Dy),(Dx+Dy))
    
    Sigmaq_joint = np.matmul(chol_Sigmaq_joint,chol_Sigmaq_joint.T)
    Sigmaq_x = Sigmaq_joint[0:Dx,0:Dx]
    Sigmaq_xy = Sigmaq_joint[0:Dx,Dx:(Dx+Dy)]
    Sigmaq_y = Sigmaq_joint[Dx:(Dx+Dy),Dx:(Dx+Dy)]
    
    OptMarg = EvalMarg(Dx,EPX,EPXX,muq_x,Sigmaq_x)
    OptCond = EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,muq_x,muq_y,Sigmaq_x,Sigmaq_xy,Sigmaq_y)
    
    params1 = np.concatenate((params[0:Dx],chol_Sigmaq_joint[0:Dx,0:Dx].flatten()))
    MargDeriv = MargDerivative(params1,Dx,EPX,EPXX)
    CondDeriv = CondDerivative(params, Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY)
    
    MI = OptMarg-OptCond
    MIN = np.sign(MI-const).flatten()
    dmu_x = MIN*(MargDeriv[0:Dx]-CondDeriv[0:Dx])
    dmu_y = -1*MIN*(CondDeriv[Dx:(Dx+Dy)])
    dSigmachol = -1*MIN*(CondDeriv[(Dx+Dy):]).reshape((Dx+Dy),(Dx+Dy))
    dSigmachol[0:Dx,0:Dx] += MIN*(MargDeriv[Dx:]).reshape((Dx,Dx))
    dparams = np.concatenate((dmu_x.flatten(),np.concatenate((dmu_y.flatten(),dSigmachol.flatten()))))
    return dparams

def MeanAndVariance(N,K,MI):
    ###############################################################################
    # Outline: Calculate the Mean and Variane of a run
    #
    # Inputs:
    #       N - Number of Different Sample Sizes
    #       K - Number of Itterations
    #       MI - Values of Mutual Informations [Storage of Itteration, Storage of Sample Size, 0=MI 1=Run Time]
    #
    # Outputs:
    #       MIMean - Mean at each Sample Size [0=MI 1=Run Time]
    #       MIVariance - Varaince of each Sample Size [0=MI 1=Run Time]
    ###############################################################################
    MIMean = np.zeros((N,2))
    MIVariance = np.zeros((N,2))

    MIMean[:,0] = sum(MI[:,:,0])/K
    MIMean[:,1] = sum(MI[:,:,1])/K

    MIVariance[:,0] = sum((MI[:,:,0]-sum(MI[:,:,0])/K)**2)/(K-1)
    MIVariance[:,1] = sum((MI[:,:,1]-sum(MI[:,:,1])/K)**2)/(K-1)
    return MIMean, MIVariance

def confidence_ellipse(mean, cov, ax, n_std=1, **kwargs):
    ###############################################################################
    # Outline: Adds standard deviation ellipse to plot
    #
    # Inputs:
    #       mean - Mean of Gaussian Distribution to Plot
    #       cov - Covariance of Gaussian Distribution to Plot
    #       ax - Plot to add standard deviation
    #       n_std - number of standard of deviations to plot
    #
    # Outputs:
    #       pearson - correlation to distribution. output is unimportant for future use
    ###############################################################################
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1,1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return pearson

if __name__ == "__main__":

    # random.seed(10)
    PRINTRESULTS = False # Prints all the values of MI calculated
    PLOTEXAMPLE = True # Plots samples of GMM with learned Var MM on top
    PlotPreSaved = False # This is for special plots with the presaved GMM
    Oracle = False # Not used in this paper.

    ###################### Create GMM Model ######################################
    K = 3 # Number of Trials
    Ns = [400,500,1000] #Number of Samples,2000
    Tol = 10**(-8)
    M = 2 # Number of Componentes
    Dx = 1 # Dimension of X
    Dy = 1 # Dimension of Y
    ws,mus,Sigmas = GaussianMixtureParams(M,Dx,Dy) # Generates Parameters for GMM   
    
    ################# Pre-Saved GMM Distributions to Test #########################   
    ws = np.array([.5,.5])
    mus = [np.array([[5],[5]]),np.array([[-5],[-5]])]
    Sigmas = [.4*np.eye(2),.2*np.eye(2)]
    PlotPreSaved = True
    ##############################################################################
    
    SampleMI = np.zeros((K,len(Ns),2))
    MMMI = np.zeros((K,len(Ns),2))
    GDMI = np.zeros((K,len(Ns),2))
    BAMI = np.zeros((K,len(Ns),2))
    VMMI = np.zeros((K,len(Ns),2))
    n=-1
    for N in Ns: # Loop over Number of Samples
        n+=1
        for k in range(K): # Loop over number or itterations
            sample = SampleGMM(N,ws,mus,Sigmas)

            ########################## Sample MI #########################################
            Stime0 = time.time()
            SampleMargEnt = MargEntGMM(sample,Dx,ws,mus,Sigmas)
            SampleCondEnt = CondEntGMM(sample,Dx,Dy,ws,mus,Sigmas)
            Stime1 = time.time()
            
            SampleMI[k,n,0] = SampleMargEnt-SampleCondEnt
            SampleMI[k,n,1] = Stime1-Stime0+.00015
            ##############################################################################


            ########################## Implicit Likelihood MI ###################################
            ILtime0 = time.time()
            EPX, EPY, EPXX, EPXY, EPYY = pmoments(sample,Dx,Dy)

            ILMargEnt = MargEntMomentMatch(Dx, EPX, EPXX)
            ILCondEnt = CondEntMomentMatch(Dx, EPX, EPY, EPXX, EPXY, EPYY)
            ILtime1 = time.time()
            
            MMMI[k,n,0] = ILMargEnt-ILCondEnt
            MMMI[k,n,1] = ILtime1-ILtime0+.00015
            ##############################################################################


            ########################## Gradient Descent MI ###################################
            GDtime0 = time.time()
            EPX, EPY, EPXX, EPXY, EPYY = pmoments(sample,Dx,Dy)

            GDMargEnt = MargEntGradientDescent(Dx,EPX,EPXX,Tol,FullOut=True)
            GDCondEnt = CondEntGradientDescent(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,Tol,FullOut=True)
            GDtime1 = time.time()
            
            GDMI[k,n,0] = GDMargEnt[-1]-GDCondEnt[-1]
            GDMI[k,n,1] = GDtime1-GDtime0+.00015
            ##############################################################################
            
            ######################## Plots GD convergence to MM###########################
            if N==Ns[0] and not PlotPreSaved:
                if k==0:
                    for j in range(len(GDCondEnt)-len(GDMargEnt)):
                        GDMargEnt.append(GDMargEnt[-1])
                    GDMIFull = np.asarray(GDMargEnt)-np.asarray(GDCondEnt)
                    MMMIFull = (ILMargEnt-ILCondEnt)*np.ones((len(GDCondEnt),1))
                    fig2 = go.Figure([
                        go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=MMMIFull.flatten(),
                            line=dict(color='rgb(255,0,0)', width=3),
                            mode='lines',
                            name='MM'
                        ),
                        go.Scatter(
                            x=np.linspace(0,len(GDCondEnt)-1,len(GDCondEnt)),
                            y=GDMIFull.flatten(),
                            line=dict(color='rgb(255,255,0)', width=3),
                            mode='lines',
                            name='GD'
                        )])
                    fig2.update_xaxes(title_text="Gradient Step")
                    fig2.update_yaxes(title_text="MI Approximation")
                    fig2.update_layout(font=dict(size=30),legend=dict(yanchor="top", y=0.82, xanchor="left", x=0.8))
                    # fig2.write_image("GMMGDStep.pdf")
                    fig2.show()
            #################################################################################

            ########################## Barber & Agakov MI ###################################
            BAtime0 = time.time()
            EPX, EPY, EPXX, EPXY, EPYY = pmoments(sample,Dx,Dy)

            BAMargEnt = MargEntGMM(sample,Dx,ws,mus,Sigmas)
            BACondEnt = CondEntMomentMatch(Dx, EPX, EPY, EPXX, EPXY, EPYY)
            BAtime1 = time.time()
            
            BAMI[k,n,0] = BAMargEnt-BACondEnt
            BAMI[k,n,1] = BAtime1-BAtime0+.00015
            ##############################################################################


            ########################## Variational Marginal MI ###################################
            VMtime0 = time.time()
            EPX, EPY, EPXX, EPXY, EPYY = pmoments(sample,Dx,Dy)

            VMMargEnt = MargEntMomentMatch(Dx, EPX, EPXX)
            VMCondEnt = CondEntGMM(sample,Dx,Dy,ws,mus,Sigmas)
            VMtime1 = time.time()
            
            VMMI[k,n,0] = VMMargEnt-VMCondEnt
            VMMI[k,n,1] = VMtime1-VMtime0+.00015
            ##############################################################################

    ########################### Oracle Distribution#######################################
    ### Uses the True MI to find a varaitaional distribution that matches the Mutual
    ### Information Exactly. This is to comapare the best possible variational
    ### distribution to the moment matched distribution. Not used in paper
    ######################################################################################
    if Oracle:

        EPX, EPY, EPXX, EPXY, EPYY = pmoments(sample,Dx,Dy)
        const  = SampleMI[k,n,0]
        mux,muy,Sigmax,Sigmaxy,Sigmay = MIOpt(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,const,10**(-20))
        VarMean = np.append(mux.flatten(),muy.flatten()).reshape(Dx+Dy,1)
        VarVar = np.append(np.append(np.append(Sigmax.flatten(),Sigmaxy.flatten()),(Sigmaxy.T).flatten()),Sigmay.flatten()).reshape(Dx+Dy,Dx+Dy)
        
        CondSigma = Sigmax-np.matmul(Sigmaxy,np.matmul(np.linalg.inv(Sigmay),Sigmaxy.T))
        
        OptMarg = EvalMarg(Dx,EPX,EPXX,mux,Sigmax)
        OptCond = EvalCond(Dx,EPX,EPY,EPXX,EPXY,EPYY,mux,muy,Sigmax,Sigmaxy,Sigmay)
        OptMI = OptMarg-OptCond
        
        MMMean = np.append(EPX.flatten(),EPY.flatten()).reshape(Dx+Dy,1)
        MMVar = np.append(np.append(np.append((EPXX-np.outer(EPX,EPX)).flatten(),(EPXY-np.outer(EPX,EPY)).flatten()),((EPXY-np.outer(EPX,EPY)).T).flatten()),(EPYY-np.outer(EPY,EPY)).flatten()).reshape(Dx+Dy,Dx+Dy)

        if PLOTEXAMPLE:
            fig2, ax = plt.subplots()
            ax.scatter(sample[:,0], sample[:,1], c='black')
            p1 = confidence_ellipse(VarMean, VarVar, ax, n_std=1, facecolor='none', edgecolor='springgreen',label='Oracle')
            p2 = confidence_ellipse(MMMean, MMVar, ax, n_std=1, facecolor='none', edgecolor='red',label='MM')
            ax.legend(fontsize=20)
            plt.show()
    ###############################################################################
    
    if PlotPreSaved:
        n_samples = 300

        # generate spherical data centered on (20, 20)
        S1 = .4*np.eye(2)
        G1 = np.dot(np.random.randn(n_samples, 2), S1) + np.array([5, 5])

        # generate zero centered stretched Gaussian data
        S2 = .2*np.eye(2)
        G2 = np.dot(np.random.randn(n_samples, 2), S2) + np.array([-5, -5])

        # concatenate the two datasets into the final training set
        X_train = np.vstack([G1, G2])

        # fit a Gaussian Mixture Model with two components
        clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
        clf.fit(X_train)
        
        S3 = np.append(np.append(np.append((EPXX-np.outer(EPX,EPX)).flatten(),(EPXY-np.outer(EPX,EPY)).flatten()),((EPXY-np.outer(EPX,EPY)).T).flatten()),(EPYY-np.outer(EPY,EPY)).flatten()).reshape(Dx+Dy,Dx+Dy)
        G3 = np.dot(np.random.randn(n_samples, 2), S3) + np.array([EPX[0], EPY[0]])
        clf1 = mixture.GaussianMixture(n_components=1, covariance_type="full")
        clf1.fit(G3)
        
        MMMean = np.append(EPX.flatten(),EPY.flatten()).reshape(Dx+Dy,1)
        MMVar = np.append(np.append(np.append((EPXX-np.outer(EPX,EPX)).flatten(),(EPXY-np.outer(EPX,EPY)).flatten()),((EPXY-np.outer(EPX,EPY)).T).flatten()),(EPYY-np.outer(EPY,EPY)).flatten()).reshape(Dx+Dy,Dx+Dy)

        # display predicted scores by the model as a contour plot
        x = np.linspace(-8.0, 8.0)
        y = np.linspace(-8.0, 8.0)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -clf.score_samples(XX)
        Z = Z.reshape(X.shape)
        Z1 = -clf1.score_samples(XX)
        Z1 = Z1.reshape(X.shape)
        Levels = [.5,1,1.5,2,2.5,3]
        fig2, ax = plt.subplots()
        CS = plt.contour(
            X, Y, Z,colors='black', levels=np.logspace(.5, 3, 10)
        )
        ax.legend(['GMM'],fontsize=30)
        # CS1 = plt.contour(
        #     X, Y, Z1,colors='red', norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(.5, 3, 10)
        # )
        # CB = plt.colorbar(CS, shrink=0.8, extend="both")
        # plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)
        p2 = confidence_ellipse(MMMean, MMVar, ax, n_std=0, facecolor='none', edgecolor='black')
        p2 = confidence_ellipse(MMMean, MMVar, ax, n_std=1, facecolor='none', edgecolor='red')
        p2 = confidence_ellipse(MMMean, MMVar, ax, n_std=1.5, facecolor='none', edgecolor='red')
        p2 = confidence_ellipse(MMMean, MMVar, ax, n_std=.5, facecolor='none', edgecolor='red')
        ax.legend(['GMM','MM'],fontsize=30,loc='upper left')
        plt.axis("tight")
        plt.savefig("GMMContour.pdf")
        plt.show()
        
        mus =  np.array([[5], [-5]])
        sigmas = np.array([[.4], [.2]]) 
        gmm = mixture.GaussianMixture(2)
        gmm.means_ = mus
        gmm.covars_ = sigmas
        gmm.weights_ = np.array([0.5, 0.5])

        #Fit the GMM with random data from the correspondent gaussians
        gaus_samples_1 = np.random.normal(mus[0], sigmas[0], 100).reshape(100,1)
        gaus_samples_2 = np.random.normal(mus[1], sigmas[1], 100).reshape(100,1)
        fit_samples = np.concatenate((gaus_samples_1, gaus_samples_2))
        gmm.fit(fit_samples)

        mu = EPX
        variance = EPXX-EPX**2
        sigma = np.sqrt(variance)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.linspace(-8, 8, 10000).reshape(10000,1)
        logprob = gmm.score_samples(x)
        pdf = np.exp(logprob)
        #print np.max(pdf) -> 19.8409464401 !?
        ax.plot(x, pdf, '-k')
        ax.plot(x, multivariate_normal.pdf(x, mu, sigma),'-r')
        plt.savefig("GMMpdf.pdf")
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        Methods = ['$I$', '$I_{marg}$', '$I_{post}$', '$I_{m+p}$']
        MIs = [SampleMI[k,n,0],VMMI[k,n,0],BAMI[k,n,0],MMMI[k,n,0]]
        ax.bar(Methods,MIs)
        ax.set_ylabel('Mutual Information',fontsize=30)
        plt.xticks(fontsize=30)
        plt.savefig('GMMbar.pdf', bbox_inches='tight')
        plt.show()

    ########################## Debugging Prints ###################################
    if PRINTRESULTS:
        print("Sample MI:       %s" %SampleMI[k,n,0])
        print("Moment Match MI: %s" %MMMI[k,n,0])
        print("Oracle MI:       %s" %OptMI)
        print("BA MI:           %s" %BAMI[k,n,0])
    ############################################################################### 


    ########################## Plot Result ########################################
    if PLOTEXAMPLE:    
        SampleMIMean, SampleMIVariance = MeanAndVariance(len(Ns),K,SampleMI)
        MMMIMean, MMMIVariance = MeanAndVariance(len(Ns),K,MMMI)
        GDMIMean, GDMIVariance = MeanAndVariance(len(Ns),K,GDMI)
        BAMIMean, BAMIVariance = MeanAndVariance(len(Ns),K,BAMI)
        VMMIMean, VMMIVariance = MeanAndVariance(len(Ns),K,VMMI)
        
        fig = go.Figure([
        go.Scatter(
            x=Ns,
            y=SampleMIMean[:,0],
            line=dict(color='rgb(0,100,80)', width=3),
            mode='lines',
            name='True MI'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((SampleMIMean[:,0]+SampleMIVariance[:,0]),(SampleMIMean[:,0]-SampleMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=MMMIMean[:,0],
            line=dict(color='rgb(255,0,0)', width=3),
            mode='lines',
            name='MM MI'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((MMMIMean[:,0]+MMMIVariance[:,0]),(MMMIMean[:,0]-MMMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=GDMIMean[:,0],
            line=dict(color='rgb(255,255,0)', width=3, dash='dot'),
            mode='lines',
            name='GD MI'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((GDMIMean[:,0]+GDMIVariance[:,0]),(GDMIMean[:,0]-GDMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=BAMIMean[:,0],
            line=dict(color='rgb(0,0,255)', width=3),
            mode='lines',
            name='BA MI'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((BAMIMean[:,0]+BAMIVariance[:,0]),(BAMIMean[:,0]-BAMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=VMMIMean[:,0],
            line=dict(color='rgb(180,0,255)', width=3),
            mode='lines',
            name='VM MI'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((VMMIMean[:,0]+VMMIVariance[:,0]),(VMMIMean[:,0]-VMMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(180,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
        ])
        fig.update_xaxes(title_text="Number of Samples", type="log")
        fig.update_yaxes(title_text="MI Approximation")
        fig.update_layout(font=dict(size=30),showlegend=False)#
        # fig.write_image("LargeGMMConverge.pdf")
        fig.show()
        
        
        fig1 = go.Figure([
        go.Scatter(
            x=Ns,
            y=SampleMIMean[:,1],
            line=dict(color='rgb(0,100,80)', width=3),
            mode='lines',
            name='True'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((SampleMIMean[:,1]+SampleMIVariance[:,1]),(SampleMIMean[:,1]-SampleMIVariance[:,1])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(0,100,80,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
        go.Scatter(
            x=Ns,
            y=MMMIMean[:,1],
            line=dict(color='rgb(255,0,0)', width=3),
            mode='lines',
            name='MM'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((MMMIMean[:,1]+MMMIVariance[:,1]),(MMMIMean[:,1]-MMMIVariance[:,1])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(255,0,0,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
        go.Scatter(
            x=Ns,
            y=GDMIMean[:,1],
            line=dict(color='rgb(255,255,0)', width=3),
            mode='lines',
            name='GD'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((GDMIMean[:,1]+GDMIVariance[:,1]),(GDMIMean[:,1]-GDMIVariance[:,1])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(255,255,0,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
            go.Scatter(
            x=Ns,
            y=BAMIMean[:,1],
            line=dict(color='rgb(0,0,255)', width=3),
            mode='lines',
            name='BA'
        ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((BAMIMean[:,1]+BAMIVariance[:,1]),(BAMIMean[:,1]-BAMIVariance[:,1])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(0,0,255,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        # ),
            go.Scatter(
            x=Ns,
            y=VMMIMean[:,1],
            line=dict(color='rgb(180,0,255)', width=3),
            mode='lines',
            name='VM'
        # ),
        # go.Scatter(
        #     x=Ns+Ns[::-1], # x, then x reversed
        #     y=np.hstack(((VMMIMean[:,1]+VMMIVariance[:,1]),(VMMIMean[:,1]-VMMIVariance[:,1])[::-1])), # upper, then lower reversed
        #     fill='toself',
        #     fillcolor='rgba(180,0,255,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False
        )
        ])
        fig1.update_xaxes(title_text="Number of Samples", type="log")
        fig1.update_yaxes(title_text="Run Time", type="log", dtick = 1)#
        #fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig1.update_layout(font=dict(size=30),legend=dict(yanchor="top", y=0.85, xanchor="left", x=0.01))
        # fig1.write_image("LargeGMMTime.pdf")
        fig1.show()
############################################################################### 
