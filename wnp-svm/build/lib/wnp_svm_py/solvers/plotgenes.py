import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import math
from kneed import KneeLocator
import matplotlib.transforms as mtransforms
import tensorflow as tf

def plot_top_genes(meta,Kingry):
    Genes = meta['oligo_id']
    Features1 = meta['Samples'].iloc[0:54]
    Features2 = meta['Samples'].iloc[54:108]

    Lungs = Kingry['A']
    Spleen = Kingry['B']
    za = Kingry['ya'].flatten()
    zb = Kingry['yb'].flatten()
    ya = za/np.linalg.norm(za)
    yb = zb/np.linalg.norm(zb)

    Aproj = np.abs(np.dot(Lungs,yb))
    Lproj = np.abs(np.dot(Spleen,ya))
    ind1 = Aproj.argsort()[::-1]
    ind2 = Lproj.argsort()[::-1]

    ALB = Lungs[ind1,:]    
    ALN = Spleen[ind2,:]
    ALLBgen = Genes[ind1]
    ALLNgen = Genes[ind2]

    N = 50
    ALLB = ALB[0:N,:]
    ALLN = ALN[0:N,:]
    ALLBgenes = ALLBgen[0:N].replace(np.nan,'',regex=True)
    ALLNgenes = ALLNgen[0:N].replace(np.nan,'',regex=True)

    f = set(ALLBgenes)
    g = set(ALLNgenes)
    h = f & g
    # print(f.difference(h))
    # print(g.difference(h))
    #print(len(h)) 

    # print(max(Aproj))
    # print(max(Lproj))

    fig, ax = plt.subplots(1,1,figsize = (11,7))
    sns.heatmap(np.concatenate((ALLB[:,0:6],ALLB[:,24:72]),axis=1),yticklabels=ALLBgenes,xticklabels=Features1,cmap='jet',robust = True)
    plt.ylabel('Genes',fontsize=10)
    plt.xlabel('Samples',fontsize=10)
    plt.title('Genes highly significant in Lungs partitioned for time points (L-R)',fontsize=10)
    plt.axvline(x = 6, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 12, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 18, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 24, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 30, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 36, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 42, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 48, color = 'r', linestyle = '-.',linewidth=6)
    #plt.savefig('Lungs.jpg')

    # GeneB1 = a['ID_REF'][ind1][0:N].replace(np.nan,'',regex=True)
    # GeneB2 = a['oligo_id'][ind1][0:N].replace(np.nan,'',regex=True)
    # GeneB3 = a['descrip'][ind1][0:N].replace(np.nan,'',regex=True)
    # H2_S = pd.DataFrame(np.concatenate((ALLB[:,0:6],ALLB[:,24:72]),axis=1),columns=Features1)
    # H2_S.insert(0,'ID_REF',np.array(GeneB1))
    # H2_S.insert(1,'oligo_id',np.array(GeneB2))
    # H2_S.insert(2,'descrip',np.array(GeneB3))
    # H2_S.to_csv(r'C:\Users\ugoob\Google Drive\KIRBY PAPERS\GSVD final Analysis\Tularensis\Lungs_NEW.csv')

    fig, ax = plt.subplots(1,1,figsize = (11,7))
    sns.heatmap(np.concatenate((ALLN[:,0:6],ALLN[:,24:72]),axis=1),yticklabels=ALLNgenes,xticklabels=Features2,cmap='jet',robust = True)
    plt.ylabel('Genes',fontsize=10)
    plt.xlabel('Samples',fontsize=10)
    plt.title('Genes highly significant in Spleen partitioned for time points (L-R)',fontsize=10)
    plt.axvline(x = 6, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 12, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 18, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 24, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 30, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 36, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 42, color = 'r', linestyle = '-.',linewidth=6)
    plt.axvline(x = 48, color = 'r', linestyle = '-.',linewidth=6)
    #plt.savefig('Spleen.jpg')

    # GeneB1 = a['ID_REF'][ind2][0:N].replace(np.nan,'',regex=True)
    # GeneB2 = a['oligo_id'][ind2][0:N].replace(np.nan,'',regex=True)
    # GeneB3 = a['descrip'][ind2][0:N].replace(np.nan,'',regex=True)
    # H2_S = pd.DataFrame(np.concatenate((ALLN[:,0:6],ALLN[:,24:72]),axis=1),columns=Features2)
    # H2_S.insert(0,'ID_REF',np.array(GeneB1))
    # H2_S.insert(1,'oligo_id',np.array(GeneB2))
    # H2_S.insert(2,'descrip',np.array(GeneB3))
    # H2_S.to_csv(r'C:\Users\ugoob\Google Drive\KIRBY PAPERS\GSVD final Analysis\Tularensis\Spleen_NEW.csv')


def plot_weights(Ap,Bp,methods,c,the,thee):

    fig, axs = plt.subplots(1, 2, figsize=(24, 8))

    # Plot for Ap
    for i in range(len(c)):
        axs[0].plot(Ap[:,i], '--', lw=4, label=methods[i], color=c[i])
    axs[0].axhline(y=0, color='k', linestyle='-.', linewidth=3)
    y = Ap[:, len(c)-1]
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    axs[0].plot(kn.knee, [y[kn.knee]], "or", ms=8)
    trans_offset = mtransforms.offset_copy(axs[0].transData, fig=fig, x=0.05, y=0.1, units='inches')
    for a, b in zip([kn.knee], [y[kn.knee]]):
        axs[0].text(a, b, str((a, round(b, 2))), color="k", fontsize=12, transform=trans_offset)
    axs[0].legend(fontsize=16,title = the[0])
    axs[0].set_ylabel('Sorted Projection Scores', fontsize=20)
    axs[0].set_xlabel('Gene Index', fontsize=20)

    # Plot for Bp
    for i in range(len(c)):
        axs[1].plot(Bp[:,i], '--', lw=4, label=methods[i], color=c[i])
    axs[1].axhline(y=0, color='k', linestyle='-.', linewidth=4)
    y = Bp[:, len(c)-1]
    x = range(1, len(y)+1)
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    axs[1].plot(kn.knee, [y[kn.knee]], "or", ms=8)
    trans_offset = mtransforms.offset_copy(axs[1].transData, fig=fig, x=0.08, y=0.1, units='inches')
    for a, b in zip([kn.knee], [y[kn.knee]]):
        axs[1].text(a, b, str((a, round(b, 2))), color="k", fontsize=12, transform=trans_offset)
    axs[1].legend(fontsize=16,title = the[1])
    axs[1].set_ylabel('Sorted Projection Scores', fontsize=20)
    axs[1].set_xlabel('Gene Index', fontsize=20)

    plt.savefig('{}.jpg'.format(thee))
    plt.tight_layout()
    plt.show()
