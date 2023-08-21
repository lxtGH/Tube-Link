import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

if __name__ == '__main__':
    SMALL_SIZE = 6
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14

    LINE_OTHERS = 2.5
    LINE_OURS = 4.5

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    fig, ax = plt.subplots()
    ax.set_ylabel('FPS')
    ax.set_xlabel('Input Subclip Size')
    plt.xlim(left=0, right=9)
    plt.ylim(bottom=1, top=16)

    x = np.array([1, 2, 4, 6, 8])
    # tb-link r101
    # 1 frames: 2
    y1 = np.array([5.8, 9.2, 9.6, 10.8, 10.4])
    # tb-link r50
    y2 = np.array([6.8, 10.0, 11.0, 12.5, 11.5])
    # tb-link stdcv2
    # 1 frames: 9.5  2frames: 12.0 4frames: 12.2 6frames: 13.9 8frames: 15.2
    y3 = np.array([9.5, 12.0, 12.2, 15.0, 15.2])
    # dialted deeplabv3+ r101
    y4 = np.array([3.5, 3.5, 3.5, 3.5, 3.5])
    # Mask2Former
    y5 = np.array([7.0, 7.0, 7.0, 7.0, 7.0 ])

    plt.plot(x, y1, '--', label="Tube-Link (R-101)", linewidth=LINE_OTHERS)
    plt.plot(x, y2, '--', label="Tube-Link (R-50)", linewidth=LINE_OTHERS)
    plt.plot(x, y3, '--', label="Tube-Link (STDCv2)", linewidth=LINE_OTHERS)
    plt.plot(x, y4, '--', label="Deeplabv3+ (R101)", linewidth=LINE_OTHERS)
    plt.plot(x, y5, '--', label="Mask2Former (R50)", linewidth=LINE_OTHERS)

    plt.grid()

    plt.legend(loc='lower left')
    plt.savefig("D:\CVPR23_logs/tb_link/fps.pdf")
    # plt.show()