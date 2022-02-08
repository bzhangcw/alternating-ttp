import matplotlib.pyplot as plt


def plot_fig(train_list, station_list, miles, D_var, A_var, save_dir, TimeSpan=1080,
             TimeStart=360):
    color_value = {
        '0': 'midnightblue',
        '1': 'mediumblue',
        '2': 'c',
        '3': 'orangered',
        '4': 'm',
        '5': 'fuchsia',
        '6': 'olive'
    }

    plt.figure(figsize=(20, 10))
    for i in range(len(train_list)):
        train = train_list[i]
        time_list = []
        location_list = []
        for station in train.staList:
            time_list.append(A_var[train][station].x)
            location_list.append(miles[station_list.index(station)])
            time_list.append(D_var[train][station].x)
            location_list.append(miles[station_list.index(station)])
        plt.plot(time_list, location_list, color=color_value[str(i % 7)], linewidth=1.5)
        plt.text(time_list[0] + 0.8, location_list[0] + 4, train.traNo, ha='center', va='bottom',
                 color=color_value[str(i % 7)], weight='bold', family='Times new roman', fontsize=9)

    plt.grid(True)  # show the grid
    plt.ylim(0, miles[-1])  # y range
    plt.xlim(TimeStart, TimeStart + TimeSpan)  # x range
    plt.yticks(miles, station_list, family='Times new roman')
    plt.xlabel('Time (min)', family='Times new roman')
    plt.ylabel('Space (km)', family='Times new roman')
    plt.savefig(save_dir)
    plt.show()


def plot_high_level_train(train_list, station_list, miles, D_var, A_var, save_dir, TimeSpan=1080, TimeStart=360):
    color_value = {
        '0': 'midnightblue',
        '1': 'mediumblue',
        '2': 'c',
        '3': 'orangered',
        '4': 'm',
        '5': 'fuchsia',
        '6': 'olive'
    }

    plt.figure(figsize=(20, 10))
    for i in range(len(train_list)):
        train = train_list[i]
        time_list = []
        location_list = []
        for station in train.staList:
            time_list.append(A_var[train][station])
            location_list.append(miles[station_list.index(station)])
            time_list.append(D_var[train][station])
            location_list.append(miles[station_list.index(station)])
        plt.plot(time_list, location_list, color=color_value[str(i % 7)], linewidth=1.5)
        plt.text(time_list[0] + 0.8, location_list[0] + 4, train.traNo, ha='center', va='bottom',
                 color=color_value[str(i % 7)], weight='bold', family='Times new roman', fontsize=9)

    plt.grid(True)  # show the grid
    plt.ylim(0, miles[-1])  # y range
    plt.xlim(TimeStart, TimeStart + TimeSpan)  # x range
    plt.yticks(miles, station_list, family='Times new roman')
    plt.xlabel('Time (min)', family='Times new roman')
    plt.ylabel('Space (km)', family='Times new roman')
    plt.savefig(save_dir)
    plt.show()
