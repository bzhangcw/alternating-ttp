export train_size=$1
export station_size=$2
export time_span=$3

nohup python -u main_slim.py &> lagrangian.$train_size.$station_size.$time_span.log