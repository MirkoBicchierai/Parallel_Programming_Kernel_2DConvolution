import numpy as np
from matplotlib import pyplot as plt


def do_plot(l, values, ty):
    x = np.arange(len(l))
    width = 0.35
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects)
        multiplier += 1

    ax.set_ylabel('Speed Up')
    ax.set_title('Speed Up vectorization analysis ' + ty)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(l)
    ax.legend(loc='upper left', ncols=2)

    plt.savefig('Img/plots/barplot' + ty + '.pdf')
    plt.savefig('Img/plots/barplot' + ty + '.png')


if __name__ == "__main__":
    l = ("3x3", "5x5", "7x7")
    values = {
        'Without vectorization': (5.18, 5.22, 5.14),
        'With vectorization': (10.27, 17.76, 34.14),
    }
    ty = "4K"
    do_plot(l, values, ty)

    values = {
        'Without vectorization': (5.22, 5.28, 5.22),
        'With vectorization': (10.49, 17.87, 34.25),
    }
    ty = "2K"
    do_plot(l, values, ty)
