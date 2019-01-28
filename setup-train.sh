# Create all the required folders
mkdir storage
mkdir storage/weights
mkdir storage/inference
mkdir storage/inference/images
mkdir storage/inference/output
mkdir storage/logs
mkdir storage/tmp


# Install all required packages
pip3 install --user -r requirements.txt

cd libraries/saliconeval/
python3 setup.py build_ext install --user
cd ../..

# Download all the training files
python3 downloader.py --type train

# Extract all required folders and remove the zip files
unzip storage/dataset.zip -d storage/
rm storage/dataset.zip
