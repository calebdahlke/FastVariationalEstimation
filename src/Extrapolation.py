import numpy as np
import plotly.graph_objects as go
import time
import pickle
import os
from datetime import datetime
import GMMMutualInfromationExperiment as util
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

def GenerateModel(d):
    X_x = np.array([[1,-.5]])
    X_d = np.array([[-1, d]])
    Sigma_x = 3
    Sigma_d = 1#np.abs(np.sin(d))+1
    return X_x, X_d, Sigma_x, Sigma_d

def SampleJoint(mu_psi, Sigma_psi,X_x, X_d, Sigma_x, Sigma_d,N):
    Sample_psi = np.random.multivariate_normal(mu_psi.flatten(),Sigma_psi,N)
    Sample_x = np.random.normal((np.matmul(X_x,Sample_psi.T)[0].flatten())**2,np.sqrt(Sigma_x)).reshape((N,1))
    Sample_y = np.random.normal((np.matmul(X_d,Sample_psi.T)[0].flatten())**2,np.sqrt(Sigma_d)).reshape((N,1))
    samples = np.hstack((Sample_psi,Sample_x,Sample_y))
    return samples

def TrueMargEnt(samples,X_x):
    N = len(samples)
    MargEnt = 0
    for i in range(N):
        p_xcondpsi = multivariate_normal.logpdf((np.matmul(X_x,samples[np.arange(N)!=i,0:2].T).flatten())**2,samples[i,2],Sigma_x)
        log_alpha = max(p_xcondpsi)
        lpx = logsumexp(p_xcondpsi - log_alpha) + log_alpha -np.log(N-1)
        MargEnt += (-1/N)*lpx
    return MargEnt

def TrueCondEnt(samples,X_x, X_d, Sigma_x, Sigma_d):
    N = len(samples)
    CondEnt = 0
    for i in range(N):
        p_xcondpsi = multivariate_normal.logpdf((np.matmul(X_x,samples[np.arange(N)!=i,0:2].T).flatten())**2,samples[i,2],Sigma_x)
        p_ycondpsi = multivariate_normal.logpdf((np.matmul(X_d,samples[np.arange(N)!=i,0:2].T).flatten())**2,samples[i,3],Sigma_d)
        
        # estimate joint (numerically stable)
        lintermed = p_xcondpsi + p_ycondpsi
        log_alpha = max(lintermed)        
        lpxy = logsumexp(lintermed - log_alpha) + log_alpha - np.log(N-1)
        
        # estimate marginal (numerically stable)
        log_beta = max(p_ycondpsi)
        lpy = logsumexp(p_ycondpsi - log_beta) + log_beta - np.log(N-1)
        
        #update entropy
        CondEnt += (-1/N)*(lpxy-lpy)
    return CondEnt

if __name__ == "__main__":

    # random.seed(10)
    PRINTRESULTS = False # Prints all the values of MI calculated
    PLOTEXAMPLE = True # Plots samples of GMM with learned Var MM on top

    ###################### Create GMM Model ######################################
    K = 3 # Number of Trials
    Ns = [100,200,400,1000,2000] # Number of Samples
    Tol = 10**(-6)
    Dx = 1
    Dy = 1
    D = np.linspace( -3,3,13)
    mu_psi = np.array([[0],[0]])
    Sigma_psi = np.array([[2,1],[1,3]])

    SampleMI = np.zeros((K,len(Ns),len(D),2))
    MMMI = np.zeros((K,len(Ns),len(D),2))
    GDMI = np.zeros((K,len(Ns),len(D),2))
    BAMI = np.zeros((K,len(Ns),len(D),2))
    VMMI = np.zeros((K,len(Ns),len(D),2))
    n=-1
    for N in Ns: # Loop over Number of Samples
        n+=1
        for k in range(K): # Loop over number or itterations
            for d in range(len(D)):
                X_x, X_d, Sigma_x, Sigma_d = GenerateModel(D[d])
                
                sampleStart = time.time()
                sample = SampleJoint(mu_psi, Sigma_psi,X_x, X_d, Sigma_x, Sigma_d,N)
                sampleEnd = time.time()
                sampleTime = sampleEnd-sampleStart
                if sampleTime<.00015:
                    sampleTime=.00015
                    
                ########################## Sample MI #########################################
                Stime0 = time.time()
                SampleMargEnt = TrueMargEnt(sample,X_x)
                SampleCondEnt = TrueCondEnt(sample,X_x, X_d, Sigma_x, Sigma_d)
                Stime1 = time.time()
                
                SampleMI[k,n,d,0] = SampleMargEnt-SampleCondEnt
                if (Stime1-Stime0)<.00015:
                    SampleMI[k,n,d,1] = .00015
                else:
                    SampleMI[k,n,d,1] = Stime1-Stime0+sampleTime
                ##############################################################################


                ########################## Implicit Likelihood MI ###################################
                ILtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                ILMargEnt = util.MargEntMomentMatch(Dx, EPX, EPXX)
                ILCondEnt = util.CondEntMomentMatch(Dx, EPX, EPY, EPXX, EPXY, EPYY)
                ILtime1 = time.time()
                
                MMMI[k,n,d,0] = ILMargEnt-ILCondEnt
                MMMI[k,n,d,1] = ILtime1-ILtime0+sampleTime
                ##############################################################################


                ########################## Gradient Descent MI ###################################
                GDtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                GDMargEnt = util.MargEntGradientDescent(Dx,EPX,EPXX,Tol)
                GDCondEnt = util.CondEntGradientDescent(Dx,Dy,EPX,EPY,EPXX,EPXY,EPYY,Tol)
                GDtime1 = time.time()
                
                GDMI[k,n,d,0] = GDMargEnt-GDCondEnt
                GDMI[k,n,d,1] = GDtime1-GDtime0+sampleTime
                ##############################################################################
                

                ########################## Barber & Agakov MI ###################################
                BAtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                BAMargEnt = TrueMargEnt(sample,X_x)
                BACondEnt = util.CondEntMomentMatch(Dx, EPX, EPY, EPXX, EPXY, EPYY)
                BAtime1 = time.time()
                
                BAMI[k,n,d,0] = BAMargEnt-BACondEnt
                BAMI[k,n,d,1] = BAtime1-BAtime0+sampleTime
                ##############################################################################


                ########################## Variational Marginal MI ###################################
                VMtime0 = time.time()
                EPX, EPY, EPXX, EPXY, EPYY = util.pmoments(sample[:,2:],Dx,Dy)

                VMMargEnt = util.MargEntMomentMatch(Dx, EPX, EPXX)
                VMCondEnt = TrueCondEnt(sample,X_x, X_d, Sigma_x, Sigma_d)
                VMtime1 = time.time()
                
                VMMI[k,n,d,0] = VMMargEnt-VMCondEnt
                VMMI[k,n,d,1] = VMtime1-VMtime0+sampleTime
                ##############################################################################

########################## Debugging Prints ###################################
    if PRINTRESULTS:
        print("Sample MI:       %s" %SampleMI[k,n,5,0])
        print("Moment Match MI: %s" %MMMI[k,n,5,0])
        print("BA MI:           %s" %BAMI[k,n,5,0])
    ############################################################################### 

    ###############################################################################
    ############################# MI By Decision###################################
    i=0
    for N in Ns:
        SampleMIMean, SampleMIVariance = util.MeanAndVariance(len(D),K,SampleMI[:,i,:,:])
        MMMIMean, MMMIVariance = util.MeanAndVariance(len(D),K,MMMI[:,i,:,:])
        GDMIMean, GDMIVariance = util.MeanAndVariance(len(D),K,GDMI[:,i,:,:])
        BAMIMean, BAMIVariance = util.MeanAndVariance(len(D),K,BAMI[:,i,:,:])
        VMMIMean, VMMIVariance = util.MeanAndVariance(len(D),K,VMMI[:,i,:,:])
        xs = D.tolist()
        fig2 = go.Figure([
        go.Scatter(
            x=xs,
            y=SampleMIMean[:,0],
            line=dict(color='rgb(0,100,80)', width=3),
            mode='lines',
            name='NMC'
        ),
        go.Scatter(
            x=xs+xs[::-1], # x, then x reversed
            y=np.hstack(((SampleMIMean[:,0]+SampleMIVariance[:,0]),(SampleMIMean[:,0]-SampleMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=xs,
            y=MMMIMean[:,0],
            line=dict(color='rgb(255,0,0)', width=3),
            mode='lines',
            name='MM'
        ),
        go.Scatter(
            x=xs+xs[::-1], # x, then x reversed
            y=np.hstack(((MMMIMean[:,0]+MMMIVariance[:,0]),(MMMIMean[:,0]-MMMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=xs,
            y=GDMIMean[:,0],
            line=dict(color='rgb(255,255,0)', width=3, dash='dot'),
            mode='lines',
            name='GD'
        ),
        go.Scatter(
            x=xs+xs[::-1], # x, then x reversed
            y=np.hstack(((GDMIMean[:,0]+GDMIVariance[:,0]),(GDMIMean[:,0]-GDMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=xs,
            y=BAMIMean[:,0],
            line=dict(color='rgb(0,0,255)', width=3),
            mode='lines',
            name='BA+NMC'
        ),
        go.Scatter(
            x=xs+xs[::-1], # x, then x reversed
            y=np.hstack(((BAMIMean[:,0]+BAMIVariance[:,0]),(BAMIMean[:,0]-BAMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=xs,
            y=VMMIMean[:,0],
            line=dict(color='rgb(180,0,255)', width=3),
            mode='lines',
            name='VM+NMC'
        ),
        go.Scatter(
            x=xs+xs[::-1], # x, then x reversed
            y=np.hstack(((VMMIMean[:,0]+VMMIVariance[:,0]),(VMMIMean[:,0]-VMMIVariance[:,0])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(180,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
        ])
        fig2.update_xaxes(title_text="Decision")
        fig2.update_yaxes(title_text="MI Approximation")
        #fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig2.update_layout(font=dict(size=30),legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        if N == Ns[0]:
            fig2.write_image("ExtrapolationMImin.pdf")
        else:
            fig2.write_image("ExtrapolationMI.pdf")
        fig2.show()
        i+=1


    ########################## Plot Result ########################################
    if PLOTEXAMPLE:    
        SampleMIMean, SampleMIVariance = util.MeanAndVariance(len(Ns),K,SampleMI[:,:,7,:])
        MMMIMean, MMMIVariance = util.MeanAndVariance(len(Ns),K,MMMI[:,:,7,:])
        GDMIMean, GDMIVariance = util.MeanAndVariance(len(Ns),K,GDMI[:,:,7,:])
        BAMIMean, BAMIVariance = util.MeanAndVariance(len(Ns),K,BAMI[:,:,7,:])
        VMMIMean, VMMIVariance = util.MeanAndVariance(len(Ns),K,VMMI[:,:,7,:])
        
        fig = go.Figure([
        go.Scatter(
            x=Ns,
            y=SampleMIMean[:,0],
            line=dict(color='rgb(0,100,80)', width=3),
            mode='lines',
            name='NMC'
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
            name='MM'
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
            name='GD'
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
            name='BA+NMC'
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
            name='VM+NMC'
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
        fig.update_xaxes(title_text="Number of Samples", type="log", dtick = 1)
        fig.update_yaxes(title_text="MI Approximation")#, type="log", dtick = "L1"
        #fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(showlegend=False)
        fig.update_layout(font=dict(size=30))#,legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        fig.write_image("ExtrapolationConverge.pdf")
        fig.show()
        
        
        fig1 = go.Figure([
        go.Scatter(
            x=Ns,
            y=SampleMIMean[:,1],
            line=dict(color='rgb(0,100,80)', width=3),
            mode='lines',
            name='NMC'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((SampleMIMean[:,1]+SampleMIVariance[:,1]),(SampleMIMean[:,1]-SampleMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=MMMIMean[:,1],
            line=dict(color='rgb(255,0,0)', width=3),
            mode='lines',
            name='MM'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((MMMIMean[:,1]+MMMIVariance[:,1]),(MMMIMean[:,1]-MMMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
        go.Scatter(
            x=Ns,
            y=GDMIMean[:,1],
            line=dict(color='rgb(255,255,0)', width=3),
            mode='lines',
            name='GD'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((GDMIMean[:,1]+GDMIVariance[:,1]),(GDMIMean[:,1]-GDMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(255,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=BAMIMean[:,1],
            line=dict(color='rgb(0,0,255)', width=3),
            mode='lines',
            name='BA+NMC'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((BAMIMean[:,1]+BAMIVariance[:,1]),(BAMIMean[:,1]-BAMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ),
            go.Scatter(
            x=Ns,
            y=VMMIMean[:,1],
            line=dict(color='rgb(180,0,255)', width=3),
            mode='lines',
            name='VM+NMC'
        ),
        go.Scatter(
            x=Ns+Ns[::-1], # x, then x reversed
            y=np.hstack(((VMMIMean[:,1]+VMMIVariance[:,1]),(VMMIMean[:,1]-VMMIVariance[:,1])[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(180,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
        ])
        fig1.update_xaxes(title_text="Number of Samples", type="log", dtick = 1)
        fig1.update_yaxes(title_text="Run Time", type="log", dtick = 1)#
        #fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig1.update_layout(showlegend=False)
        fig1.update_layout(font=dict(size=30))#,legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        fig1.write_image("ExtrapolationTime.pdf")
        fig1.show()
###############################################################################