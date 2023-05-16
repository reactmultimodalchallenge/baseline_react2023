# REACT 2023 Multimodal Challenge
[[Homepage]](https://sites.google.com/cam.ac.uk/react2023/home)  [[Reference Paper]](https://arxiv.org/abs/2302.06514) [[Code]](https://github.com/lingjivoo/React2023)


This repository provides baseline methods for the [REACT 2023 Multimodal Challenge](https://sites.google.com/cam.ac.uk/react2023/home)

### Challenge Description
Human behavioural responses are stimulated by their environment (or context), and people will inductively process the stimulus and modify their interactions to produce an appropriate response. When facing the same stimulus, different facial reactions could be triggered across not only different subjects but also the same subjects under different contexts. The Multimodal Multiple Appropriate Facial Reaction Generation Challenge (REACT 2023) is a satellite event of ACM MM 2023, (Ottawa, Canada, October 2023), which aims at comparison of multimedia processing and machine learning methods for automatic human facial reaction generation under different dyadic interaction scenarios. The goal of the Challenge is to provide the first benchmark test set for multimodal information processing and to bring together the audio, visual and audio-visual affective computing communities, to compare the relative merits of the approaches to automatic appropriate facial reaction generation under well-defined conditions. 


#### Task 1 - Offline Appropriate Facial Reaction Generation
This task aims to develop a machine learning model that takes the entire speaker behaviour sequence as the input, and generates multiple appropriate and realistic / naturalistic spatio-temporal facial reactions, consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour. 


#### Task 2 - Online Appropriate Facial Reaction Generation
This task aims to develop a machine learning model that estimates each frame, rather than taking all frames into consideration. The model is expected to gradually generate all facial reaction frames to form multiple appropriate and realistic / naturalistic spatio-temporal facial reactions consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour. 


## 🛠️ Installation

### Basic requirements

- Python 3.8+ 
- PyTorch 1.9+
- CUDA 11.1+ 

### Install Python dependencies

```shell
conda create -n react python=3.8
conda activate react
pip install -r requirements.txt
```


## 👨‍🏫 Get Started 

<details><summary> <b> Data </b> </summary>
<p>
 
**Challenge Data Description:**
- The REACT 2023 Multimodal Challenge Dataset is a compilation of recordings from the following three publicly available datasets for studying dyadic interactions: [NOXI](https://dl.acm.org/doi/10.1145/3136755.3136780), [RECOLA](https://ieeexplore.ieee.org/document/6553805) and [UDIVA](https://www.computer.org/csdl/proceedings-article/wacvw/2021/196700a001/1sZ3sn1GBxe). 

- Participants can apply for the data at our [Homepage](https://sites.google.com/cam.ac.uk/react2023/home).
   
**Data organization (`data/`) is listed below:**
```
data
├── test
├── val
├── train
   ├── Video_files
       ├── NoXI
           ├── 010_2016-03-25_Paris
               ├── Expert_video
               ├── Novice_video
                   ├── 1
                       ├── 1.png
                       ├── ....
                       ├── 751.png
                   ├── ....
           ├── ....
       ├── RECOLA
       ├── UDIVA
   ├── Audio_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.wav
                   ├── ....
           ├── group-2
           ├── group-3
       ├── UDIVA
   ├── Emotion
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.csv
                   ├── ....
           ├── group-2
           ├── group-3
       ├── UDIVA
   ├── 3D_FV_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.npy
                   ├── ....
           ├── group-2
           ├── group-3
       ├── UDIVA
            
```
</p>
</details>


<details><summary> <b> External Tool Preparation </b> </summary>
<p>

In this baseline, we leverage [3DMM model](https://github.com/LizhenWangT/FaceVerse) to extract 3DMM coefficients and render the 3D listener and 

then we use a 3D-to-2D [PIRender](https://github.com/RenYurui/PIRender) to render final 2D frames of listener.
   
- If you use our prepared 3DMM coefficients, you need to download the FaceVerse version 2 model at this [page](https://github.com/LizhenWangT/FaceVerse) 
 
  and put in the folder (`external/FaceVerse/data/`).
 
  We provide extracted 3DMM coefficients for downloading at [Google Drive]. 

  We also provide the mean_face and std_face of 3DMM coefficients at [Google Drive]. Please put them at the folder (`external/FaceVerse/`).

 
- We re-train the PIRender and provide the [checkpoint]. Please put it at the folder (`external/PIRender/`).
   
</p>
</details>


<details><summary> <b> Training </b>  </summary>
<p>
 
- Running the following shell can start training:
 ```shell
 python train.py --batch-size 8  --gpu_ids 0  -lr 0.00002  -e 50  -j 12  --outdir results/train_offline
 ```
 or 
 
  ```shell
 python train.py --batch-size 8  --gpu_ids 0  -lr 0.00002  -e 50  -j 12  --online --outdir results/train_online
 ```
 
</p>
</details>


<details><summary> <b> Validation </b>  </summary>
<p>
 
- Before validation, run the following to get the martix (defining appropriate neighbours in val set):
 ```shell
 cd tool
 python val_matrix.py --dataset-path ./data
 ```
 Please put files(data_indices.csv, neighbour_emotion_1_7.0921.npy and val.csv) in the folder `./data/`
 
 Then, evaluate a trained model on val set and run:
```shell
python val.py  --resume xxx/best_checkpoint.pth  --gpu-ids 1  --outdir val_online --online
```
 or
 ```shell
python val.py  --resume xxx/best_checkpoint.pth  --gpu-ids 1  --outdir val_offline
```
 
</p>
</details>




## 🖊️ Citation

```BibTeX
@misc{song2023multiple,
  title={Multiple Appropriate Facial Reaction Generation in Dyadic Interaction Settings: What, Why and How?},
  author={Song, Siyang and Spitale, Micol and Luo, Yiming and Bal, Batuhan and Gunes, Hatice},
  journal={arXiv e-prints},
  pages={arXiv--2302},
  year={2023}
}


@inproceedings{palmero2021context,
  title={Context-aware personality inference in dyadic scenarios: Introducing the udiva dataset},
  author={Palmero, Cristina and Selva, Javier and Smeureanu, Sorina and Junior, Julio and Jacques, CS and Clap{\'e}s, Albert and Mosegu{\'\i}, Alexa and Zhang, Zejian and Gallardo, David and Guilera, Georgina and others},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1--12},
  year={2021}
}

@inproceedings{ringeval2013introducing,
  title={Introducing the RECOLA multimodal corpus of remote collaborative and affective interactions},
  author={Ringeval, Fabien and Sonderegger, Andreas and Sauer, Juergen and Lalanne, Denis},
  booktitle={2013 10th IEEE international conference and workshops on automatic face and gesture recognition (FG)},
  pages={1--8},
  year={2013},
  organization={IEEE}
}

@inproceedings{cafaro2017noxi,
  title={The NoXi database: multimodal recordings of mediated novice-expert interactions},
  author={Cafaro, Angelo and Wagner, Johannes and Baur, Tobias and Dermouche, Soumia and Torres Torres, Mercedes and Pelachaud, Catherine and Andr{\'e}, Elisabeth and Valstar, Michel},
  booktitle={Proceedings of the 19th ACM International Conference on Multimodal Interaction},
  pages={350--359},
  year={2017}
}


```

## 🤝 Acknowledgement
Thanks to the open source of the following projects:

- [FaceVerse](https://github.com/LizhenWangT/FaceVerse) &#8194;

- [PIRender](https://github.com/RenYurui/PIRender) &#8194;
