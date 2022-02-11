export train_size=$1
export station_size=$2
export time_span=$3
export iter_max=$4
export dual=$5 # lagrange or pdhg
export primal=$6 # seq or jsp

nohup python -u main_slim.py &> $primal-$dual.$train_size.$station_size.$time_span.$iter_max.log