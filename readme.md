



# AutoML Project Setup



## Prerequisites



* Python 3.11

* Conda (Miniconda or Anaconda)

* NVIDIA GPU (optional, for GPU acceleration)

* CUDA-compatible drivers installed on your system



## 1. Create and Activate a Conda Environment



```bash

conda create -n automl python=3.11 cudatoolkit=11.3 -y

conda activate automl

```



## 2. Install PyTorch



Install the CUDA-enabled version of PyTorch:



```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

```



> **Note:** Ensure the selected PyTorch/CUDA version is compatible with your GPU drivers.



## 3. Upgrade Packaging Tools



```bash

pip install --upgrade pip setuptools wheel

```



## 4. Install AutoGluon



```bash

pip install autogluon

```



## 5. Install Project Dependencies



```bash

pip install -r requirements.txt

```



## 6. Launch the Application



Start the Streamlit application:



```bash

streamlit run app.py

```



## Verify Installation



To confirm that PyTorch can access your GPU:



```bash

python -c "import torch; print(torch.cuda.is_available())"

```



Expected output:



```text

True

```



---



### Quick Setup



```bash

conda create -n automl python=3.11 cudatoolkit=11.3 -y

conda activate automl



pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install autogluon

pip install -r requirements.txt



streamlit run app.py

```


