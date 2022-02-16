import random

import matplotlib.pyplot as plt


def plot_fig(train_list, station_list, miles, time_table, save_dir='', TimeSpan=1080, TimeStart=360):
    color_value = ['#EA0437', '#87D300', '#FFD100', '#4F1F91', '#A24CC8', '#D71671', '#FF7200', '#009EDB', '#78C7EB',
                   '#BC87E6', '#7C2230', '#007B63', '#F293D1', '#7F7800', '#BBA786', '#32D4CB', '#B67770', '#D6A461']
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
        if train.standard == 0:
            train_color = color_value[i % len(color_value)]
        else:
            train_color = 'black'
        plt.plot(time_list, location_list, color=train_color, linewidth=1.5)
        rand_shift = random.randint(5, 30)
        plt.text(time_list[0] + rand_shift / 5, location_list[0] + rand_shift, train.traNo, ha='center', va='bottom',
                 color=train_color, weight='bold', fontsize=4)

    plt.grid(True)  # show the grid
    plt.ylim(0, miles[-1])  # y range
    plt.xlim(TimeStart, TimeStart + TimeSpan)  # x range
    plt.yticks(miles, station_list)
    plt.xlabel('Time (min)')
    plt.ylabel('Space (km)')
    if save_dir != '':
        plt.savefig(save_dir)
    plt.show()
