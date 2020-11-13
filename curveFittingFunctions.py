#!/usr/bin/env python
# coding: utf-8

# ### Nonlinear Leasts squares curve fitting with B-Splines

# In[1]:


def bspline_curve_fit(data, n, p, refine):
    """
    Algorithm from Borges and Pastva, Total least squares fitting of Bezier and B-spline curves to ordered data. 
    Specifically, the B-Spline method. Knot vector and parameterization
   (not specified from this paper) are determined by the methods recommended
   in Piegl the NURBS Book p 412, i.e. the parameterization is the
    "centripetal method" and the knots are the  from eq 9.69 which guarantees
    every knot span contains atleast one parameteric point.

 B-Spline curve is optimized line search using scipy fminbound and the knot 
 vectors are updated each time the curve is re-parameterized
 Inputs:
-------
 data        - the data to be fitted
 n           - numer of control points in B-spline curve
 p           - degree of curve
 refine      - boolean flag to add points to refine the data

 Outputs:
--------
 t     - parameterization
 knots - knot vector
 CP   - control points B-Spline Curve

 Written by DeAnna Sewell Gilchrist """
    import numpy as np
    import scipy.optimize as sciopt
    from numpy import linalg as LA
    from scipy import linalg 
    import sys

    
    #--------------------------------------------------------------------------
    # Initialize variables before Gauss-Newton iterating
    #--------------------------------------------------------------------------
    if refine == False:
        U = data[:,0]
        V = data[:,1]

    else:
        u = 0
        v = 0
        for ind in range(len(data)-1):
            u = np.append(u, np.linspace(data[ind,0], data[ind+1,0],num = 5))
            v = np.append(v, np.linspace(data[ind,1], data[ind+1,1],num = 5))
        u = u[1::]
        v = v[1::]
        u = u.T
        v = v.T
        U = u
        V = v

    m = len(V); # number of data points
    toler = 1e-12; # tolerance for Gauss-Newton iteration
    Data = np.column_stack((U, V))

    #--------------------------------------------------------------------------
    # Initialize t-vector and compute knots
    #--------------------------------------------------------------------------
    d = 0;
    for k in range(1,m): # not inclusive of m
        d += LA.norm( Data[k,:]-Data[k-1,:] )**(1/2)
    	
    # Generate parameterization- page 365 in The NURBS book
    ubar = np.zeros(m)
    for k in range(1,m-1):
        ubar[k] = ubar[k-1] + (LA.norm( Data[k,:]-Data[k-1,:] )**(1/2))/d    
    ubar[-1] = 1
    t = ubar[1:-1]
    tau = [0]
    tau = np.append(tau[:],t)
    tau = np.append(tau[:],1)

    knots = computeKnots(ubar, m-1, n-1, p)

    #--------------------------------------------------------------------------
    #  B-SPLINE MATRIX and Q-R factorization
    #--------------------------------------------------------------------------

    Nn = splvander(tau, 3, knots,0)
    Q,R,E = linalg.qr(Nn,pivoting=True)
    Q2 = Q[:,p+1:]


    #--------------------------------------------------------------------------
    #  Initialize the iterating variables
    #--------------------------------------------------------------------------

    resid = np.matmul(Q2.T, np.column_stack((U, V)) )
    err = np.matmul(resid.flatten('F').T, resid.flatten('F'))  # variable projection functional (thing minimizing)

    for iter in range(1):
    
        # Finish computing the residual
        resid = np.matmul(Q2,resid[:])

        # Decompose other components of the QR factorization
        R = R[:p+1,:]
        Q1 = Q[:,:p+1] 
        oldErr = err      # Update the error
        
        #----------------------------------------------------------------------
        # Compute Bernstein derivative (dBn), projection matrix (P), and
        # Jacobian (J)
        #----------------------------------------------------------------------

        dNn = splvander(t, 3, knots,1)
        P = np.matmul( dNn[:,E], LA.lstsq(R,Q1.T, rcond=None)[0] ) #LA.lstsq solving linear system via least squares bc of matrix sizes
        Q2p = Q2[1:m-1,:].T 
        aa = np.matmul( Q2p, np.diag(np.matmul(P,U) ) )
        J1 = np.matmul(Q2,  np.matmul( Q2p, np.diag(np.matmul(P,U) ) ) ) +             np.matmul( P.T, np.diag(resid[1:m-1,0]))
        J2 = np.matmul(Q2,  np.matmul( Q2p, np.diag(np.matmul(P,V) ) ) ) +             np.matmul( P.T, np.diag(resid[1:m-1,1]))
        J = np.vstack( (J1,J2) )
        
        # step for Gauss-Newton
        #----------------------------------------------------------------------
        delT = LA.lstsq(J, np.matrix.flatten(resid,'F') ,rcond=None)[0];
        
        # Force point ordering
        #----------------------------------------------------------------------
        ords = np.divide( (np.append(delT, 0) - np.append(0, delT) ), (np.append(t,1) - np.append(0,t) ) )
        if min(ords) <= -1:
            delT = delT[:]/(-min(ords)*1.1)
        
        # take new step
        #----------------------------------------------------------------------
        tnew = t[:] + delT[:]

        
        # update  the iterating variables
        #----------------------------------------------------------------------

        Nn = splvander(np.append(np.append(0, tnew[:]),1), 3, knots,0)
        Q,R,E = linalg.qr(Nn,pivoting=True)
        Q2 = Q[:,p+1:]
        resid = np.matmul(Q2.T, np.column_stack((U, V)) )
        err = np.matmul(resid.flatten('F').T, resid.flatten('F'))  # variable projection functional (thing minimizing)
        
        # Make sure that the residual is getting smaller
        #----------------------------------------------------------------------
        if err < oldErr:
            t = tnew[:]
        else:
            # Use something to perform line search
            #------------------------------------------------------------------
            alpha = sciopt.fminbound(to_min,0,1,args=(delT,knots,t,p,U,V,m,n))

            # Update everything again
            t = t[:] + alpha*delT[:]
            Nn = splvander(np.append(np.append(0, t[:]),1), 3, knots,0)
            Q,R,E = linalg.qr(Nn,pivoting=True)
            Q2 = Q[:,p+1:]
            
            resid = np.matmul(Q2.T, np.column_stack((U, V)) )
            err = np.matmul(resid.flatten('F').T, resid.flatten('F'))
            
        # Continue

        #relErr = LA.norm(crv - np.column_stack((U, V) ))
        relErr = np.abs(oldErr - err)/oldErr
        
        if relErr < 10^-6:
            break
            
        # End of newton loop
        
    # finalize output
    #--------------------------------------------------------------------------
    t = np.append(np.append(0, t[:]),1)
    Nn = splvander(t, 3, knots,0)
    CP = LA.lstsq(Nn, np.column_stack((U, V)), rcond=None)[0]
    return(t, knots, CP, err)
    


# #--------------------------------------------------------------------------
# #--------------------------------------------------------------------------
# ###       SUPPLEMENTARY FUNCTIONS
# #--------------------------------------------------------------------------
# #--------------------------------------------------------------------------



def computeKnots(locParams, m, n,p): 
    """computes knots using equation eq 9.69 in The NURBS Book; u initializes the knot vector"""
    import numpy as np    
    dd = (m+1)/(n-p+1)# Compute parameters for computing knots
    
    # Compute internal knots
    u = np.zeros(n-p)
    for jj in range(1,n-p+1): # +1 to be inclusive of last point
#         ii = int(np.floor(jj*dd))
        ii = int(round(jj*dd))
        alpha = jj*dd - ii
#         
        u[jj-1] = (1-alpha)*locParams[ii-1] + alpha*locParams[ii]
    
    kts = np.zeros(p+1)
    kts = np.append(kts,u)
    kts = np.append(kts,np.ones(p+1))
    return kts




def to_min(alpha,delT,knots,t,p,U,V,m,n):
    from scipy import linalg
    import numpy as np

    # for line search using fminbound
    #--------------------------------------------------------------------------
    nt = t[:] + alpha*delT[:]
    Nn = splvander(np.append(np.append(0, nt[:]),1), 3, knots,0)
    Q,R,E = linalg.qr(Nn,pivoting=True)
    Q2 = Q[:,p+1:]
    resid = np.matmul(Q2.T, np.column_stack((U, V)) )
    err = np.matmul(resid.flatten('F').T, resid.flatten('F'))
    return err


# In[ ]:


def splvander(x, deg, knots, deriv):
    """ original source: https://mail.python.org/pipermail/scipy-user/2012-July/032677.html
    
    Vandermonde type matrix for splines.

    Returns a matrix whose columns are the values of the b-splines of deg
    `deg` associated with the knot sequence `knots` evaluated at the points
    `x`.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the b-splines.
    deg : int
        Degree of the splines.
    knots : array_like
        List of knots. The convention here is that the interior knots have
        been extended at both ends by ``deg + 1`` extra knots.

    Returns
    -------
    vander : ndarray
        Vandermonde like matrix of shape (m,n), where ``m = len(x)`` and
        ``m = len(knots) - deg - 1``

    Notes
    -----
    The knots exending the interior points are usually taken to be the same
    as the endpoints of the interval on which the spline will be evaluated.

    """
    from scipy.interpolate import fitpack as spl
    import numpy as np
  
    m = len(knots) - deg - 1
    v = np.zeros((m, len(x)))
    d = np.eye(m, len(knots))
    for i in range(m):
        v[i] = spl.splev(x, (knots, d[i], deg), der = deriv)
    return v.T

