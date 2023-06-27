# Cells Segger (cells-instance-segmentation)
# the production repo

## Installation

```console
git clone https://github.com/duanjingqi/cells-instance-segmentation-production.git
cd ./cells-instance-segmentation-production
pip install -r requirements.txt
echo "export \$SEGGER_DIR=$(pwd)" >> $HOME/.bashrc
```

## Predict with Cells Segger
1. Display the predict.py help message
```console
python $SEGGER_DIR/predict.py -h
```
2. Predict for a single image 
```console
python $SEGGER_DIR/predict.py -image foo.png -dest output_dir
```
3. Predict in batch
```console
python $SEGGER_DIR/predict.py -files foo.txt -dest output_dir
```
Feed single image to '_image' argument. For batch prediction, add file path of the images to be used to a txt file, e.g. foo.txt, feed the file to '_files' argument. Predicted cell masks are in a png image for each input image in 'output_dir'.

