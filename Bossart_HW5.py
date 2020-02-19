import numpy as np
 

nc = 85
ni = 196
nt = 341
pi0 = 1/3
pc0 = 1/3
pt0 = 1/3
n = nc + ni + nt


def exp_ncc(pc, pi, pt):
    return (nc*(pc**2)/(pc**2 + 2*pc*pi + 2*pc*pt))

def exp_nci(pc, pi, pt):
    return ((2*nc*pc*pi)/(pc**2 + 2*pc*pi + 2*pc*pt))

def exp_nct(pc, pi, pt):
    return ((2*nc*pc*pt)/(pc**2 + 2*pc*pi + 2*pc*pt))

def exp_nii(pc, pi, pt):
    return ((ni*(pi**2))/(pi**2 + 2*pi*pt))

def exp_nit(pc, pi, pt):
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

tol = 1e-5
pi_vals = [pi0]
pc_vals = [pc0]
pt_vals = [pt0]

def run_em():
    i = 0
    converged = False
    
    while not converged:
        ncc_val = exp_ncc(pc_vals[i], pi_vals[i], pt_vals[i])
        nci_val = exp_nci(pc_vals[i], pi_vals[i], pt_vals[i])
        nct_val = exp_nct(pc_vals[i], pi_vals[i], pt_vals[i])
        nii_val = exp_nii(pc_vals[i], pi_vals[i], pt_vals[i])
        nit_val = exp_nit(pc_vals[i], pi_vals[i], pt_vals[i])
        
        current_pi = update_pi(nii_val, nit_val, nci_val)
        current_pc = update_pc(ncc_val, nci_val, nct_val)
        current_pt = update_pt(nt, nct_val, nit_val)
        
        pi_vals.append(current_pi)
        pc_vals.append(current_pc)
        pt_vals.append(current_pt)
        i += 1
        
        if (r(pc_vals[i-1], pi_vals[i-1], pc_vals[i], pi_vals[i]) <= tol):
            converged = True

run_em()       
        