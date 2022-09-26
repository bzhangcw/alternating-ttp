export station_size=$2
export time_span=$3

#for nt in 10 20 50; do
#  for ns in 5 10; do
#    for time_span in 200 300 500; do
#      export train_size=$nt
#      export station_size=$ns
#      export time_span=$time_span
#      python main_create_models.py
#    done
#  done
#done
#
#for nt in 50 100 200; do
#  for ns in 15 20 29; do
#    for time_span in 600 720 1080; do
#      export train_size=$nt
#      export station_size=$ns
#      export time_span=$time_span
#      python main_create_models.py
#    done
#  done
#done


for nt in 100 200; do
  for ns in 15 20 29; do
    for time_span in 720; do
      export train_size=$nt
      export station_size=$ns
      export time_span=$time_span
      python main_create_models.py
    done
  done
done
