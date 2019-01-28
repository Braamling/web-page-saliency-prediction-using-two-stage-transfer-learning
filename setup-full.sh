# Create all the required folders
mkdir storage
mkdir storage/salicon
mkdir storage/weights
mkdir storage/inference
mkdir storage/inference/images
mkdir storage/inference/output
mkdir storage/logs
mkdir storage/tmp
mkdir storage/FiWi
mkdir storage/salicon/images
mkdir storage/salicon/heatmaps
mkdir storage/salicon/heatmaps/train
mkdir storage/salicon/heatmaps/val
mkdir storage/FiWi/heatmaps
mkdir storage/FiWi/heatmaps/val
mkdir storage/FiWi/heatmaps/train
mkdir storage/FiWi/images
mkdir storage/FiWi/images/val
mkdir storage/FiWi/images/train

# install the salicon and mscoco api's
cd lib/cocoapi/PythonAPI/
python setup.py build_ext install
cd ../../salicon
python setup.py build_ext install
cd ../../

# Install all required packages
pip3 install -r requirements.txt

# Download all the training files
python3 downloader.py --type full

# Extract all required folders and remove the zip files
unzip storage/salicon/val_images.zip -d storage/salicon/images
unzip storage/salicon/train_images.zip -d storage/salicon/images
unzip storage/FiWi/dataset.zip -d storage/FiWi

rm storage/salicon/val_images.zip
rm storage/salicon/train_images.zip
rm storage/FiWi/dataset.zip

# Create the fixation data.
python3 createFixationImages.py