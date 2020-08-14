import seaborn
import numpy as np
import matplotlib.pyplot as plt


def read_names(filenamepath):
    f = open(filenamepath,'r')
    lines = f.readlines()
    filenamelist = []
    for line in lines:
        if '.DS_Store' not in line:
            line = line.split('_')
            speeds = line[1].strip('Hz').split(',')
            #print(speeds)
            speeds = speeds[0]+'.'+speeds[1]
            filename = float(speeds),int(line[3])
            filenamelist.append(filename)

    return filenamelist

def get_stiffnesses(data):
    dicti = {}
    temp = []
    for i in range(len(data)):
        print(data[i][0])
        if data[i][0] in dicti.keys():
            print("  :",data[i][1])
            dicti[data[i][0]].append(data[i][1])
            None
        else:
            dicti[data[i][0]]=[data[i][1]]


    for key in dicti.keys():
        dicti[key].sort()
    return dicti


def main():
    testdata = read_names('../../scratch/bd_lstm/filenames/testfiles.txt')
    traindata = read_names('../../scratch/bd_lstm/filenames/trainfiles.txt')
    min_stiffness = -250
    print(get_stiffnesses(testdata)[10.0])
    print(get_stiffnesses(traindata)[10.05])
    test_stiff = get_stiffnesses(testdata)
    train_stiff = get_stiffnesses(traindata)
    test = 0
    train = 1
    no = 2
    speedindices = [i for i in range(0,281,1)]
    speeds = [i/100 for i in range(400,1805,5)]
    stiffindices = [i for i in range(0,26,1)]
    stiffs = [i for i in range(-250,10,10)]

    def indices(i,j):
        return speeds[i], stiffs[j]

    heat_values = np.zeros((len(range(400,1805,5)),len(range(-250,10,10))))
    print(heat_values.shape)
    for i in speedindices:
        for j in stiffindices:
            #print(i,j)
            speed, stiff = indices(i,j)
            if speed in test_stiff.keys() and stiff in test_stiff[speed]:
                heat_values[i,j] = test
            elif speed in train_stiff.keys() and stiff in train_stiff[speed]:
                heat_values[i,j] = train
            else:
                heat_values[i,j] = no
    print(heat_values.shape)
    yticks = [x for x in range(-250,10,10)]
    xticks = [y/100 for y in range(400,1805,5)]

    ax= seaborn.heatmap(heat_values.T,yticklabels=yticks,xticklabels = xticks,cbar = False,annot_kws={"size": 16})
    new_labels = np.arange(4,19)
    old_ticks = ax.get_xticks()
    new_ticks = np.linspace(np.min(old_ticks), np.max(old_ticks), len(new_labels))
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_labels)
    #seaborn.heatmap(heat_values.T,xticklabels=[x for x in range(-250,10,10)],yticklabels = [y/100 for y in range(1000,1805,5)])
    ax.set(xlabel = "Rotating speed (Hz)",
    ylabel = "Stiffness control position (mm)")
    plt.show()
    #plt.savefig('../../../paper/elsarticle/heatmap.pdf',format='pdf')
    print(heat_values[0,:])

main()
