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
    ax.set_ylabel('mIoU(%)')
    ax.set_xlabel('Input Subclip Size')
    plt.xlim(left=0, right=9)
    plt.ylim(bottom=30, top=46)

    x = np.array([1, 2, 4, 6, 8])
    # tb-link r101
    y1 = np.array([42.0, 43.0, 44.2, 45.0, 44.6])
    # tb-link r50
    y2 = np.array([40.2, 41.4, 42.8, 43.4, 43.0])
    # tb-link stdcv2
    y3 = np.array([33.8, 34.5, 35.1, 35.8, 35.5])
    # deeplabv3+ r101
    y4 = np.array([35.7, 35.7, 35.7, 35.7, 35.7])
    # Mask2Former
    y5 = np.array([38.4, 38.4, 38.4, 38.4, 38.4 ])

    plt.plot(x, y1, '--', label="Tube-Link (R-101)", linewidth=LINE_OTHERS)
    plt.plot(x, y2, '--', label="Tube-Link (R-50)", linewidth=LINE_OTHERS)
    plt.plot(x, y3, '--', label="Tube-Link (STDCv2)", linewidth=LINE_OTHERS)
    plt.plot(x, y4, '--', label="Deeplabv3+ (R101)", linewidth=LINE_OTHERS)
    plt.plot(x, y5, '--', label="Mask2Former (R50)", linewidth=LINE_OTHERS)

    plt.grid()

    plt.legend(loc='lower left')
    plt.savefig("D:\CVPR23_logs/tb_link/mIoU.pdf")
    # plt.show()