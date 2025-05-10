# x2ct_gan

# 📦 X2CT-GAN: Reconstructing CT from X-rays using GAN

本專案為 X2CT-GAN 的實作，實現從多視角 X-ray 影像重建 3D CT volume。  
包含完整的資料處理、模型訓練、測試與可視化流程。

---

## 🔧 Requirement

- Python ≥ **3.10**
- 建議使用虛擬環境（如 `conda`, `venv`）

### ⚙️ Torch 安裝建議

若有 GPU（請依實際 CUDA 版本選擇）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

若無 GPU：

```bash
pip install torch torchvision torchaudio
```

---

## 📥 安裝方式

請先安裝依賴套件：

```bash
pip install -r requirements.txt
```

---

## 📁 資料集與模型下載

本專案使用的訓練資料與預訓練模型可直接從原始論文作者提供的 GitHub 專案下載：

🔗 **原始專案網址**：https://github.com/kylekma/X2CT/tree/master

- 預處理後的資料位於：
  ```
  ./data/LIDC-HDF5-256/
  ```

- 訓練與測試列表檔案為：
  ```
  ./data/train.txt
  ./data/test.txt
  ```

- 預訓練模型下載後請放至：
  ```
  ./save_models/multiView_CTGAN/LIDC256/d2_multiview2500/checkpoints/
  ```

---

## 🚀 Demo 使用方式

### ▶️ 訓練模型

```bash
python train.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml \
                --gpu=0 --dataroot=./data/LIDC-HDF5-256 \
                --dataset=train --tag=d2_multiview2500 \
                --data=LIDC256 --dataset_class=align_ct_xray_views_std \
                --datasetfile=./data/train.txt --valid_datasetfile=./data/test.txt \
                --valid_dataset=test --model_class=MultiViewCTGAN
```

---

### 🧪 測試模型（輸出數值指標）

```bash
python test.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml \
               --gpu=0 --dataroot=./data/LIDC-HDF5-256 \
               --dataset=test --tag=d2_multiview2500 \
               --data=LIDC256 --dataset_class=align_ct_xray_views_std \
               --datasetfile=./data/test.txt --resultdir=./multiview \
               --check_point=90 --how_many=3 --model_class=MultiViewCTGAN
```

---

### 🖼 可視化結果（輸出 CT volume 圖片）

```bash
python visual.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml \
                 --gpu=0 --dataroot=./data/LIDC-HDF5-256 \
                 --dataset=test --tag=d2_multiview2500 \
                 --data=LIDC256 --dataset_class=align_ct_xray_views_std \
                 --datasetfile=./data/test.txt --resultdir=./multiview \
                 --check_point=90 --how_many=3 --model_class=MultiViewCTGAN
```

---

如需更多設定項目，請參考各 `.py` 檔案與 `configs/` 中的 YAML 設定檔說明。