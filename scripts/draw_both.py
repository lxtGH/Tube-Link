import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (10, 5)

if __name__ == '__main__':
    SMALL_SIZE = 6
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 22

    LINE_OTHERS = 2.5
    LINE_OURS = 4.5

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=15)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    fig, axs = plt.subplots(ncols=2)
    ax = axs[0]
    ax.set_ylabel('mIoU(%)')
    ax.set_xlabel('Input Subclip Size')
    ax.set_xlim(left=0.5, right=8)
    ax.set_ylim(bottom=30, top=46)

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
    y5 = np.array([38.4, 38.4, 38.4, 38.4, 38.4])

    ax.plot(x, y1, '--', label="Tube-Link (R-101)", linewidth=LINE_OTHERS)
    ax.plot(x, y2, '--', label="Tube-Link (R-50)", linewidth=LINE_OTHERS)
    ax.plot(x, y3, '--', label="Tube-Link (STDCv2)", linewidth=LINE_OTHERS)
    ax.plot(x, y4, '--', label="Deeplabv3+ (R-101)", linewidth=LINE_OTHERS)
    ax.plot(x, y5, '--', label="Mask2Former (R-50)", linewidth=LINE_OTHERS)

    ax.grid()

    ax.legend(loc='lower left')

    ax = axs[1]
    ax.set_ylabel('FPS')
    ax.set_xlabel('Input Subclip Size')
    ax.set_xlim(left=0.5, right=8)
    ax.set_ylim(bottom=1, top=16)

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
    y5 = np.array([7.0, 7.0, 7.0, 7.0, 7.0])

    ax.plot(x, y1, '--', label="Tube-Link (R-101)", linewidth=LINE_OTHERS)
    ax.plot(x, y2, '--', label="Tube-Link (R-50)", linewidth=LINE_OTHERS)
    ax.plot(x, y3, '--', label="Tube-Link (STDCv2)", linewidth=LINE_OTHERS)
    ax.plot(x, y4, '--', label="Deeplabv3+ (R-101)", linewidth=LINE_OTHERS)
    ax.plot(x, y5, '--', label="Mask2Former (R-50)", linewidth=LINE_OTHERS)

    ax.grid()

    ax.legend(loc='lower left')

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("./both.pdf")
    plt.show()
