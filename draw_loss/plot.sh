caffe_root='/home/smiles/hz/caffe-ssd'

#$caffe_root/tools/extra/parse_log.sh caffe.log  
$caffe_root/tools/extra/plot_training_log.py.example 6 loc_flow.png caffe.log  

#    0: Test accuracy  vs. Iters  
#    1: Test accuracy  vs. Seconds  
#    2: Test loss  vs. Iters  
#    3: Test loss  vs. Seconds  
#    4: Train learning rate  vs. Iters  
#    5: Train learning rate  vs. Seconds  
#    6: Train loss  vs. Iters  
#    7: Train loss  vs. Seconds  
