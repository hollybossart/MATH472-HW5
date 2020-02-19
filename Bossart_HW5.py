import numpy as np
 

nc = 85
ni = 196
nt = 341
pi0 = 1/3
pc0 = 1/3
pt0 = 1/3
n = nc + ni + nt


def ncc(pc, pi, pt):
    return (nc*(pc**2)/(pc**2 + 2*pc*pi + 2*pc*pt))

def nci(pc, pi, pt):
    return ((2*nc*pc*pi)/(pc**2 + 2*pc*pi + 2*pc*pt))

def nct(pc, pi, pt):
    return ((2*nc*pc*pt)/(pc**2 + 2*pc*pi + 2*pc*pt))

def nii(pc, pi, pt):
    return ((ni*(pi**2))/(pi**2 + 2*pi*pt))

def nit(pc, pi, pt):
    return (2*ni*pi*pt)/(pi**2 + 2*pi*pt)

def update_pc(ncc, nci, nct):
    return ((2*ncc + nci + nct)/(2*n))

def update_pi(nii, nit, nci):
    return ((2*nii + nit + nci)/(2*n))
    
def update_pt(ntt, nct, nit):
    return ((2*ntt + nct + nit)/(2*n))


def r(pc1, pi1, pc2, pi2):
    ptemp1 = [pc1, pi1]
    ptemp1 = np.asarray(ptemp1)
    ptemp2 = [pc2, pi2]
    ptemp2 = np.asarray(ptemp2)
    num = np.linalg.norm(ptemp2 - ptemp1)
    denom = np.linalg.norm(ptemp1)
    return (num / denom)

def dc(pct, pc_hat, pct_1):
    return (pct - pc_hat)/(pct_1 - pc_hat)

def di(pit, pi_hat, pit_1):
    return (pit - pi_hat)/(pit_1 - pi_hat)


pi_vals = []
pc_vals = []
pt_vals = []
def run_em():
    
    for i in range(10):
        
    