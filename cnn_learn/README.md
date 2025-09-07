# cnn_utils

## Introduction:  
- Repo for CNN learning and some personal, useful utils for Pytorch and Matplotlib.
- Will add some other utils like dirs reading (os lib), training process plotting, metrics plotting, etc.
  
## Env setup  
- Make a venv:
```
python -m venv cnn_venv  
"./cnn_venv/Scripts/activate.bat"  
pip install -r "./cnn_learn/requirements.txt"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  
```  
- Now pray that the version is compatible for your usecase.  
## Structure of the Repo:  
<pre>
cnn_learn/  
│  
├── cam/                     # Class activation map  
│   ├── __init__.py  
│   ├── core.py              # Core of CAM, Grad-CAM, Grad-CAM++, etc  
│   ├── utils.py             # Other utils  
│   ├── visualize.py         # Visualization of CAM  
│   ├── example.py           # Example  
│   └── cli.py               # Command line support  
│  
├── dataset_utils/           # File manipulation for dataset  
│   ├── __init__.py  
│   ├── ds_visualize.py      # Visualization of dataset's thingy  
│   ├── ds_dir.py            # Get loader, renaming utils  
│   ├── self_sup_anno.py     # Self-supervised annotation  
│   └── k_fold.py            # Use K-fold cross-validation  
│  
├── kernels/                 # Basic Kernels  
│   ├── __init__.py          # __init__ file  
│   ├── blur.py              # Kernel for bluring  
│   ├── edge_detection.py    # Kernel for edge detection  
│   ├── sharpen.py           # Kernel for sharpening  
│   └── utils.py             # Other utilization (forward, tensor4plt, normalize_tensor, plot_tensors)  
│  
├── validation_report/       # Report of loss, acc, etc.  
│   ├── __init__.py          # __init__ file  
│   ├── loss_acc_graph.py    # Loss and accuracy visualization right after training  
│   ├── confusion_mat.py  
│   ├── compute_metrics.py   # Latency, throughput and FLOP  
│   └── model_comparison.py  # Save result JSON, comparison calculation
│  
├── main.ipynb               # Main file  
│  
├── requirements.txt         # Requirements for the CNN part  
├── .gitignore               # Git ignore file  
└── README.md                # Info for the cnn  
</pre>
