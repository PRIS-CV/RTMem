# RTMem

# Environment

numpy, torch, torchvision

six, h5py, Pillow, scipy,

scikit-learn, metric-learn, 

faiss_gpu


# Datasets

Market-1501, MSMT17

# Train

<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/RTMem_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --eps 0.5 --num-instances 16
</code></pre>


# Acknowledgements

Thanks to [ClusterContrast](https://github.com/alibaba/cluster-contrast-reid) and [SpCL](https://github.com/yxgeee/SpCL) for the opening source
