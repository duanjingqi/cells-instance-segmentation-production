# cells-instance-segmentation-production

## Install Cell Segger
git clone https://github.com/duanjingqi/cells-instance-segmentation-production.git
cd ./cells-instance-segmentation-production
pip install -r requirements.txt

## Predict with Cell Segger
### To see help message
python cells-instance-segmentation-production/predict.py -h

### Predict
python cells-instance-segmentation-production/predict.py -image image.png -dest output_dir

