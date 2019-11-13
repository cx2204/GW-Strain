import numpy as np
import astropy
import matplotlib.pyplot as plt

# common function shortcuts
log10 = np.log10
pi = np.pi
sqrt = np.sqrt

def strain(Mass, freq, q, Dist):

    # physical constants for natural units c = G = 1
    c = 2.99792458 * (10 ** 8)
    G = 6.67428 * (10 ** (-11))
    s_mass = G * (1.98892 * 10 ** (30)) / (c ** 3)

    h = (2 * (Mass * s_mass) ** (5/3) * (q / (1+q) ** 2) * (pi * freq) ** (2/3)) / (Dist * 9.93 * 1e13) # (Dist * 1e6 * 3.31 * 3 * 1e7)
    return h

def GW_strain(parameter):
    M_vir, freq, D_l = parameter[0],parameter[1],parameter[2]
    # physical constants for natural units c = G = 1
    c=2.99792458*(10**8)
    G=6.67428*(10**(-11))
    s_mass=G*(1.98892*10**(30))/(c**3)

    # M_BHB = M_vir * s_mass
    M_BHB = 10 ** M_vir

    mass_dist = np.random.normal(M_BHB, 1e6 , 5000)
    # freq_dist =freq # computed from period, no distribution.
    freq_dist = np.random.normal(10 ** -8, 10 ** -9, 5000)
    q_dist = np.random.uniform(0.026, 1, 5000)
    D_l_dist = np.random.normal(D_l, 10, 5000)

    h_dist = []
    for i in range(5000):
        M = np.random.choice(mass_dist)
        f = np.random.choice(freq_dist)
        qq = np.random.choice(q_dist)
        D_L = np.random.choice(D_l_dist)
        h_dist.append(strain(M, f, qq, D_L))

    from scipy import stats
    skewness, h, h_Err = stats.skewnorm.fit(h_dist)
    List_result = [h,h_Err,skewness]

    # return skewness, h, h_Err
    return List_result

def run_in_parallel(parameter):
    from multiprocessing import Pool
    pool = Pool(processes=6)
    P = pool.map(GW_strain, parameter)
    return P

Mass = np.loadtxt('SaveData/GW_Calc/QSOIdMilogM_BH_1376.txt')
# Mass = np.loadtxt('Data/QSOIdMilogM_BH_1376.txt')
QSOId, log_M_BH = Mass[:,0], Mass[:,2]

Data = np.loadtxt('SaveData/GW_Calc/qsoid_z_Dl.txt')
# Data = np.loadtxt('Data/qsoid_z_Dl.txt')
z, D_l = Data[:,1], Data[:,2]

LC = np.load('SaveData/GW_Calc/QSOId_CRTS_NUV_1315.npz',allow_pickle=True)
qsoid, mjd1, mag1, magErr1 = LC['a'], LC['b'], LC['c'], LC['d']

Id, ind1, ind2 = np.intersect1d(QSOId,qsoid,return_indices=True)

M_vir, redshift, D_L = log_M_BH[ind1], z[ind1], D_l[ind1]

List_data = np.stack([M_vir[:5], redshift[:5], D_L[:5]],axis=1)
#
import time
start_time = time.time()
Output = run_in_parallel(List_data)
end_time = time.time()
print(end_time - start_time)

print(Output)

# h, h_Err, a = [],[],[]
# for i in range(len(QSOId)):
#     parameter = M_vir[i], redshift[i], D_L[i]
#     h1, hErr1,a1 = GW_strain(parameter)
#     h.append(h1)
#     h_Err.append(hErr1)
#     a.append(a1)

# np.savez('SaveData/GW_Calc/h_Err_a.npz',a=Output)

# np.savez('Data/h_Err_a.npz',a=h,b=h_Err,c=a)

