```bash
conda create -n quipsharp python=3.11
conda activate quipsharp
pip install -r requirements.txt
cd quiptools
pip install -e .
cd ..
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .
cd ..
```