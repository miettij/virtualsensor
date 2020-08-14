import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

def tresdeplot(arrpath, flag):
    arr = np.loadtxt(arrpath)
    plt.rcParams.update({'font.size': 12})
    print(arr.shape)
    Xs = arr[0,:] #vel?
    Ys = arr[1,:] #frq?
    Zs = arr[2,:] #amplitude
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')




    if flag == 'oir':
        Zs = Zs*1000
        #Zs = np.log(Zs)
        surf = ax.plot_trisurf(Xs[:], Ys[:], Zs[:], cmap=cm.jet, linewidth=0, vmin = 0,vmax = 150)
        fig.colorbar(surf)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    if flag == 'oex':
        Zs = Zs*1000
        surf = ax.plot_trisurf(Xs[:], Ys[:], Zs[:], cmap=cm.jet, linewidth=0, vmin = 0,vmax = 150)
        fig.colorbar(surf)
        ax.xaxis.set_major_locator(MaxNLocator(6))

    majors = [i for i in range(0,41,8)]

    ax.yaxis.set_major_locator(ticker.FixedLocator(majors))
    #ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.set_zlim(0,620)
    ax.zaxis.set_major_locator(ticker.FixedLocator([0,200,400,600], nbins = 4))
    #ax.zaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=1.0, numdecs=4, numticks=None))
    #ax.zaxis.set_major_locator(ticker.IndexLocator(80,80))
    #ax.zaxis.set_major_locator(ticker.LinearLocator(presets = (80,320)))
    #zloc = ax.zaxis.set_major_locator(ticker.MultipleLocator([80,160,240,320]))
    ax.invert_yaxis()
    ax.set_xlabel('Rotating speed (Hz)')
    ax.set_ylabel('Frequency (Hz)')


    fig.tight_layout()
    ax.view_init(25, 205)
    ax.set_zlabel('Amplitude (μm)',rotation=90)
    savepath = get_savepath(arrpath)
    print(savepath)
    plt.savefig(savepath,format = 'pdf')
    #plt.show() # or:

def errorplot(approx,true,flag):
    approxarr = np.loadtxt(approx)
    truearr = np.loadtxt(true)

    approx_Xs = approxarr[0,:]
    approx_Ys = approxarr[1,:]
    approx_Zs = approxarr[2,:]
    #mean = np.mean(approx_Zs)
    #approx_Zs -= mean

    true_Xs = truearr[0,:]
    true_Ys = truearr[1,:]
    true_Zs = truearr[2,:]
    percentages = 0
    for i in range(len(approx_Zs)):
        if approx_Ys[i]>1:
            approx_Zs[i] = approx_Zs[i]-true_Zs[i]
        else:
            approx_Zs[i] = 0
        #print(approx_Zs[i],"-",true_Zs[i],"/",true_Zs[i], "=|",(approx_Zs[i]-true_Zs[i])/true_Zs[i],"|=",np.abs((approx_Zs[i]-true_Zs[i])/true_Zs[i]))
        errorprop = np.abs((approx_Zs[i]-true_Zs[i])/true_Zs[i])
        if errorprop > percentages:
            print("At ", true_Ys[i], "and ", true_Xs[i])
            print(approx_Zs[i],"-",true_Zs[i],"/",true_Zs[i], "=|",(approx_Zs[i]-true_Zs[i])/true_Zs[i],"|=",np.abs((approx_Zs[i]-true_Zs[i])/true_Zs[i]))

            percentages = errorprop
            print(percentages)


    approx_Zs = approx_Zs*1000
    print(max(np.abs(approx_Zs)))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_zlim(-10, 50)
    # plt.rcParams.update({'font.size': 12})
    #
    #
    #
    # if flag == 'oir':
    #     surf = ax.plot_trisurf(approx_Xs[:], approx_Ys[:], approx_Zs[:], cmap=cm.jet, linewidth=0,vmin = -10, vmax = 15)
    #     fig.colorbar(surf)
    #     ax.xaxis.set_major_locator(MaxNLocator(5))
    # if flag == 'oex':
    #     surf = ax.plot_trisurf(approx_Xs[:], approx_Ys[:], approx_Zs[:], cmap=cm.jet, linewidth=0,vmin = -10, vmax = 15)
    #     fig.colorbar(surf)
    #     ax.xaxis.set_major_locator(MaxNLocator(6))
    #
    # majors = [i for i in range(0,41,8)]
    #
    # ax.yaxis.set_major_locator(ticker.FixedLocator(majors))
    # #ax.yaxis.set_major_locator(MaxNLocator(6))
    # ax.zaxis.set_major_locator(ticker.FixedLocator([i for i in range(-15,51,5)]))
    # ax.invert_yaxis()
    # ax.set_xlabel('Rotating speed (Hz)')
    # ax.set_ylabel('Frequency (Hz)')
    #
    # #ax.set_clim(-10,10)
    # fig.tight_layout()
    # ax.view_init(25, 205)
    # ax.set_zlabel('Amplitude (μm)',rotation=90)
    # savepath = get_savepath_err(approx)
    # plt.savefig(savepath,format = 'pdf')
    # #plt.show() # or:

def get_savepath_err(arrpath):
    folderpath = '../../../paper/elsarticle/'
    filename = None
    if 'HOR' in arrpath:
        filename = 'herror2.pdf'
    else:
        filename = 'verror2.pdf'
    return folderpath+filename

def get_savepath(arrpath):
    folderpath = '../../../paper/elsarticle/'
    filename = None
    if 'HOR' in arrpath:
        if 'pred' in arrpath:
            filename = 'horpred.pdf'
        if 'true' in arrpath:
            filename = 'hortrue.png'
    if 'VER' in arrpath:
        if 'pred' in arrpath:
            filename = 'verpred.png'
        if 'true' in arrpath:
            filename = 'vertrue.png'
    return folderpath+filename

if __name__ == '__main__':
    # print(get_savepath('../../scratch/bd_lstm/teststats/HORpred.csv'))
    # print(get_savepath('../../scratch/bd_lstm/teststats/HORtrue.csv'))
    # print(get_savepath('../../scratch/bd_lstm/teststats/VERpred.csv'))
    # print(get_savepath('../../scratch/bd_lstm/teststats/VERtrue.csv'))
    #tresdeplot('../../scratch/bd_lstm/teststats/HORpred.csv','oir')
    #tresdeplot('../../scratch/bd_lstm/teststats/HORtrue.csv','oir')
    errorplot('../../scratch/bd_lstm/teststats/HORpred.csv','../../scratch/bd_lstm/teststats/HORtrue.csv','oir')

    #tresdeplot('../../scratch/bd_lstm/teststats/VERpred.csv','oir')
    #tresdeplot('../../scratch/bd_lstm/teststats/VERtrue.csv','oir')
    errorplot('../../scratch/bd_lstm/teststats/VERpred.csv','../../scratch/bd_lstm/teststats/VERtrue.csv','oir')
