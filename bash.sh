#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Mon 16 Mar 14:46:04 GMT 2026      
date_start=`date +%s`
echo $SLURM_JOB_NUM_NODES nodes \( $SLURM_CPUS_ON_NODE processes per node \)        
echo $SLURM_JOB_NUM_NODES hosts used: $SLURM_JOB_NODELIST      
# Set this otherwise a different transport gets selected on some nodes and things break in strange ways
export OMPI_MCA_pml=^cm
echo Job output begins                                           
echo -----------------                                           
echo   
#hostname

# Need to set the max locked memory very high otherwise IB can't allocate enough and fails with "UCX  ERROR Failed to allocate memory pool chunk: Input/output error"
ulimit -l unlimited

export OMP_NUM_THEADS=1
 nice -n 10 /bin/bash -c /usr/bin/python3 src/bpe_tokenizer.py --shards_dir data/datasets/cylinder_graph_hmm/shards --vocab_size 128 --save_path data/bpe_tokenizer_128.json && /usr/bin/python3 src/prepare_bpe_data.py --tokenizer_path data/bpe_tokenizer_128.json --shards_dir data/datasets/cylinder_graph_hmm/shards --seq_length 16
  echo ---------------                                           
  echo Job output ends                                           

  date_end=`date +%s`
  seconds=$((date_end-date_start))
  minutes=$((seconds/60))
  seconds=$((seconds-60*minutes))
  hours=$((minutes/60))
  minutes=$((minutes-60*hours))
  echo =========================================================   
  echo PBS job: finished   date = `date`   
  echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
  echo =========================================================
