@echo off
echo ========================================
echo Car Detection System - Auto Setup
echo ========================================
echo.

echo [1/5] Creating virtual environment...
conda create --name mmdet_car python=3.8 -y
conda activate mmdet_car

echo [2/5] Installing PyTorch...
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

echo [3/5] Cloning and installing MMDetection...
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..

echo [4/5] Installing requirements...
pip install -r requirements.txt

echo [5/5] Setup complete!
echo.
echo Next steps:
echo 1. Download dataset from Kaggle
echo 2. Run: python tools/fix_yolo_validation.py
echo 3. Run: python app_gui/car_detection_app.py
pause