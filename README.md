## Two-Stage Transfer Learning of End-to-End Convolutional Neural Networks for Webpage Saliency Prediction
Implementation by *Bram van den Akker*.
Original paper by *Shan, Wei and Sun, Guangling and Zhou, Xiaofei and Liu, Zhi*

**Note: Before starting, make sure unzip, python3 and pip3 are installed.**
#### Setting up saliency predictor.
Depending on your usage, there are several setups possible. Please select the one that is applicable for your use case. Note: please install the correct pytorch version for your systems on [here](http://pytorch.org/).
##### Inference setup (recommended)
If your application only requires a pre-trained inference model, the inference setup should be enough. This setup will install the requirements, setup the folder structure and download the pre-trained model weights. All required steps can be performed using:
```
sh setup-infer.sh
```
##### Training setup (less recommended)
If you want to retrain an existing model or create a new model, the training setup is required. This will download all the processed images and heatmaps from salicon and FiWi datasets, create the required folders and install all python packages. All required steps can be performed using:
```
sh setup-train.sh
```
##### Full setup (not recommended)
If you want to change anything to the preprocessing of the Salicon and FiWi datasets (ie. gaussian episilon or using the raw dataset) you will need to run the full setup. Note: this setup takes quite a while. This will download the raw Salicon and FiWi dataset, install the salicon and msCOCO libaries, creates the heatmaps, makes the FiWi training/validation split, create the required folders and install all python packages.
```
sh setup-full.sh
```
##### Manual setup (not recommended)
TODO, but for now please check the content of the bash scripts above.


#### Training the model
Training the model can be as easy as running `train.py`. By default, the model starts on stage one of the transfer learning process. 

  | argument | Description | default value |
  | :-------------: |:-------------:| :-----| 
  | heatmap_path | The location of the salicon heatmaps data for training. | 'storage/salicon/heatmaps/' |
  | image_path | The location of the salicon images for training. | 'storage/salicon/images/' |
  | weights_path | The location to store the model weights. *Note: naming is done with the description argument.* | 'storage/weights/{}.pth' |
  | checkpoint | The location to store the model intermediate checkpoint weights.  *Note: naming is done with the description argument.* | 'storage/weights/{}_checkpoint.pth' |
  | batch_size | The batch size used for training. | 10 |
  | model_type | The pretrained vgg model to start from. (if training from loaded weights, the same models has to be used.). | "vgg16" |
  | epochs | The amount of epochs used to train. | 10 |
  | from_weights | The model to start training from, if None it will start from scratch (pretrained vgg).  | None |
  | log_dir | The location to place the tensorboard logs.  *Note: naming is done with the description argument.* | 'storage/logs/{}' |
  | tmp_dir | The location to place temporary files that are generated during training. | 'storage/tmp/' |
  | phase | The transfer learning phase to start | 1 |
  | description | The description of the run, for logging, output and weights naming. | 'example_run' |
  | learning_rate | The learning rate to use for the experiment | '0.0001' |

Stage one can be started by running 

```
python train.py --heatmap_path storage/salicon/heatmaps/ --image_path storage/salicon/images --phase 1 --description stage_one`
```

After the model is trained, stage two can be initalized by running 
```
python train.py --heatmap_path storage/FiWi/heatmaps/ --image_path storage/FiWi/images --phase 2 --description stage_two --from_weights storage/weights/stage_one.pth
```

#### Inference
Inference can be performed in large batches using `infer.py`. Make sure to specify as source folder with images to perform inference on and a target for the saliency heatmaps. Each saliency heatmap will be stored under the same name as its source in the specified location.

  | argument | Description | default value |
  | :-------------: |:-------------:| -----:| 
| image_path | The location to store the stage one model weights. | 'storage/inference/images' | 
| batch_size | Batch size to use during inference. | 10 | 
| target_path | The location to store the stage one model weights. | 'storage/inference/output' | 
| model_type | The model type to use for inference (vgg16 or vgg16_bn). | 'vgg16' | 
| weights_path | The location to store the model weights. | 'storage/weights/s1_weights.pth' | 

```
python train.py --image_path storage/inference/images --target_path storage/inference/output --weights_path storage/weights/stage_two.pth
```

Credits:

- SALICON-api has been forked from https://github.com/NUS-VIP/salicon-api and is modified to work with python3.
- COCO-api has been forked from https://github.com/cocodataset/cocoapi
- Salicon-evaluation has been forked and modified from https://github.com/NUS-VIP/salicon-evaluation
- Drive-download code has been taken from Stackoverflow user `turdus-merula` on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

>@inproceedings{shan2017two,
>  title={Two-Stage Transfer Learning of End-to-End Convolutional Neural Networks for Webpage Saliency Prediction},
>  author={Shan, Wei and Sun, Guangling and Zhou, Xiaofei and Liu, Zhi},
>  booktitle={International Conference on Intelligent Science and Big Data Engineering},
>  pages={316--324},
>  year={2017},
>  organization={Springer}
>}	