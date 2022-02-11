import matplotlib.pyplot as plt


def plot_fig(train_list, station_list, miles, time_table, save_dir, show = True, TimeSpan=1080, TimeStart=360):
    color_value = {
        '0': '#EA0437',
        '1': '#87D300',
        '2': '#FFD100',
        '3': '#4F1F91',
        '4': '#A24CC8',
        '5': '#D71671',
        '6': '#FF7200',
        '7': '#009EDB',
        '8': '#78C7EB',
        '9': '#BC87E6',
        '10': '#7C2230',
        '11': '#007B63',
        '12': '#F293D1',
        '13': '#7F7800',
        '14': '#BBA786',
        '15': '#32D4CB',
        '16': '#B67770',
        '17': '#D6A461'
    }

    plt.figure(figsize=(20, 10))
    for i in range(len(train_list)):
        train = train_list[i]
        time_list = []
        location_list = []
        for station in train.staList:
            time_list.append(time_table[train.traNo][station]['arr'])
            location_list.append(miles[station_list.index(station)])
            time_list.append(time_table[train.traNo][station]['dep'])
            location_list.append(miles[station_list.index(station)])
        plt.plot(time_list, location_list, color=color_value[str(i % len(color_value))], linewidth=1.5)
        plt.text(time_list[0] + 0.8, location_list[0] + 4, train.traNo, ha='center', va='bottom',
                 color=color_value[str(i % len(color_value))], weight='bold', family='Times new roman', fontsize=9)

    plt.grid(True)  # show the grid
    plt.ylim(0, miles[-1])  # y range
    plt.xlim(TimeStart, TimeStart + TimeSpan)  # x range
    plt.yticks(miles, station_list)
    plt.xlabel('Time (min)')
    plt.ylabel('Space (km)')
    plt.savefig(save_dir)

    if show:
        plt.show()
