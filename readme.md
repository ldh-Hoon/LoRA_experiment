
pip install -r requirements.txt

cd peft

pip install .

cd ..

---

Torch version: 2.8.0+cu128

CUDA available: True

CUDA version: 12.8

conda activate 후



---

python train.py --config_path configs/boolq_llama3_2_1b.yaml
python train.py --config_path configs/pclora_boolq_llama3_2_1b.yaml


python test.py --config_path configs/pclora_boolq_llama3_2_1b_test.yaml --model_path ./results/
