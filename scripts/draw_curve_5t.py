import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    LINE_OTHERS = 2.5
    LINE_OURS = 4.5

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    fig, ax = plt.subplots()
    ax.set_ylabel('Accuracy (Top-1)')
    ax.set_xlabel('Number of classes')
    plt.xlim(left=50, right=100)
    plt.ylim(bottom=30, top=100)

    x = np.array([50, 60, 70, 80, 90, 100])
    # LwF
    y1 = np.array([86, 60.7, 50.2, 45.6, 42.1, 40.1])
    # iCaRL
    y2 = np.array([85.4, 74.7, 65.1, 60.6, 55.5, 53.6])
    # LUCIR
    y3 = np.array([86, 78.1, 71.1, 68.2, 64.5, 60])
    # PODNet
    y4 = np.array([86.4, 80.6, 76.3, 74.7, 69.8, 67.6])
    # POD-AANets
    y5 = np.array([86.1, 81.5, 77.7, 75.5, 71.6, 69.4])
    # FtC
    y6 = np.array([88.40, 84.07, 80.14, 77.78, 73.89, 71.48])

    plt.plot(x, y1, '--', label="LwF (54.12)", linewidth=LINE_OTHERS)
    plt.plot(x, y2, '--', label="iCaRL (65.82)", linewidth=LINE_OTHERS)
    plt.plot(x, y3, '--', label="LUCIR (71.32)", linewidth=LINE_OTHERS)
    plt.plot(x, y4, '--', label="PODNet (75.90)", linewidth=LINE_OTHERS)
    plt.plot(x, y5, '--', label="AANet (76.97)", linewidth=LINE_OTHERS)
    plt.plot(x, y6, '-', label="FtC (Ours) (79.29)", linewidth=LINE_OURS)
    plt.grid()

    plt.legend(loc='lower left')
    plt.savefig("./5t.pdf")
