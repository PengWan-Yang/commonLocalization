# Localizing the Common Action Among a Few Videos(ECCV20)

This paper strives to localize the temporal extent of an action in a long untrimmed video. Where existing work leverages many examples with their start, their ending, and/or the class of the action during training time, we propose few-shot common action localization. The start and end of an action in a long untrimmed video is determined based on just a hand-full of trimmed video examples containing the same action, without knowing their common class label. To address this task, we introduce a new 3D convolutional network architecture able to align representations from the support videos with the relevant query video segments. The network contains: (i) a mutual enhancement module to simultaneously complement the representation of the few trimmed support videos and the untrimmed query video; (ii) a progressive alignment module that iteratively fuses the support videos into the query branch; and (iii) a pairwise matching module to weigh the importance of different support videos. Evaluation of few-shot common action localization in untrimmed videos containing a single or multiple action instances demonstrates the effectiveness and general applicability of our proposal.

For more details, please check our [paper](https://arxiv.org/abs/2008.05826).



### System Requirements

   * cuda=9.0
   * python 3.6
   * gcc=5.5.0
   * torch=0.4(currently doesn't support torch0.4.1, for a smooth installation of NMS, see https://github.com/jwyang/faster-rcnn.pytorch/issues/235#issuecomment-409493006)

### Package Requirements

```
conda create -n focal  python=3.6
pip install torch==0.4
pip install --no-cache --upgrade git+https://github.com/dongzhuoyao/pytorchgo.git
pip install -r requirements.txt
```


Compile the cuda dependencies using following simple commands:
```
cd lib
sh make.sh
```


### Preparation

Download the initial backbone weight from [Onedrive](https://1drv.ms/u/s!AjFGSP2CJWCwgXE38zYaQRR3a6L9?e=8FNHFg), and put it in the directory data/pretrained_model



Extract ActivityNet1.3 frames by FPS=3 following [R-C3D](https://github.com/VisionLearningGroup/R-C3D/blob/master/preprocess/activityNet/generate_frames.py), after that please put them in the directory ```dataset/activitynet13/train_val_frames_3/```, it ought to contain two folders: ```training, validation```.


The detail structure of the dataset is already splitted in our pickle file in ```./preprocess```. If you want to create your own dataset, you can follow [here](https://github.com/sunnyxiaohu/R-C3D.pytorch/blob/master/preprocess/activitynet/generate_roidb_training.py#L137) to create your own pickle file.



### Training

```
python main.py --bs 1 --gpus 0
```


### Evaluate our trained weight 

Firstly, download our trained weight from [Onedrive](https://1drv.ms/u/s!ArycXAIEda_Kcexadq6DPu0AF5o?e=fID2NN), and put the trained weight file ```best_model.pth``` in ```train_log/main```, then do the evaluation following the command:

```
python main.py --test
```

### Email for QA

Any question related to the repo, please send email to us: ```yangpengwan2016@gmail.com```.


### Acknowledgement

This repo is developed based on [https://github.com/sunnyxiaohu/R-C3D.pytorch](https://github.com/sunnyxiaohu/R-C3D.pytorch), thanks for their contribution.

### Citation

If you think our work is useful, please kindly cite our work.

```
@INPROCEEDINGS{YangECCV20,
        author = {Pengwan Yang and Vincent Tao Hu and Pascal Mettes and Cees G. M. Snoek},
        title = {Localizing the Common Action Among a Few Videos},
        booktitle = {European Conference on Computer Vision},
        month = {August},
        year = {2020},
        address = {Glasgow, UK},
      }
```




