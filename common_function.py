from matplotlib import pyplot as plt


def speed_up(seq, par, par_s, x, labels, ty, kernel_name):
    colors = ["#ff6347", "#4682b4", "#32cd32", "#ff8c00", "#9370db"]

    x.insert(0, 1)
    for j in range(len(seq)):
        su = [1]
        total_sum = sum(seq[j])
        seq_avg = total_sum / len(seq[j])
        sum_times_threads =[sum(y) for y in zip(*par[j])]
        for t in sum_times_threads:
            par_avg = t / len(par[j])
            su.append(seq_avg / par_avg)

        plt.plot(x, su, label=labels[j], color=colors[j])

    plt.title('Speed Up - ' + kernel_name)
    plt.xlabel('Threads')
    plt.ylabel('speed up')
    plt.legend()
    plt.grid(True)
    plt.savefig('Img/plots/speed-up_' + ty + '_' + kernel_name + '.png')
    plt.savefig('Img/plots/speed-up_' + ty + '_' + kernel_name + '.pdf')
    plt.close()

    for j in range(len(seq)):
        su = [1]
        total_sum = sum(seq[j])
        seq_avg = total_sum / len(seq[j])
        sum_times_threads =[sum(y) for y in zip(*par_s[j])]
        for t in sum_times_threads:
            par_avg = t / len(par_s[j])
            su.append(seq_avg / par_avg)

        plt.plot(x, su, label=labels[j], color=colors[j], alpha=0.65)

    plt.title('Speed Up (Shared Memory)- ' + kernel_name)
    plt.xlabel('Threads')
    plt.ylabel('speed up')
    plt.grid(True)
    plt.legend()
    plt.savefig('Img/plots/speed-up_' + ty + '_shared_' + kernel_name + '.png')
    plt.savefig('Img/plots/speed-up_' + ty + '_shared_' + kernel_name + '.pdf')
