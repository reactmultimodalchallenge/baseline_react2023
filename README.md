# This is the official baseline code for REACT 2023 Multimodal Challenge
[[Homepage]](https://sites.google.com/cam.ac.uk/react2023/home)  [[Reference Paper]](https://arxiv.org/abs/2302.06514) [[Code]](https://github.com/reactmultimodalchallenge/baseline_react2023)

This repository provides baseline methods for the [REACT 2023 Multimodal Challenge](https://sites.google.com/cam.ac.uk/react2023/home)

### Baseline paper:
To be released.

### Challenge Description
Human behavioural responses are stimulated by their environment (or context), and people will inductively process the stimulus and modify their interactions to produce an appropriate response. When facing the same stimulus, different facial reactions could be triggered across not only different subjects but also the same subjects under different contexts. The Multimodal Multiple Appropriate Facial Reaction Generation Challenge (REACT 2023) is a satellite event of ACM MM 2023, (Ottawa, Canada, October 2023), which aims at comparison of multimedia processing and machine learning methods for automatic human facial reaction generation under different dyadic interaction scenarios. The goal of the Challenge is to provide the first benchmark test set for multimodal information processing and to bring together the audio, visual and audio-visual affective computing communities, to compare the relative merits of the approaches to automatic appropriate facial reaction generation under well-defined conditions. 


#### Task 1 - Offline Appropriate Facial Reaction Generation
This task aims to develop a machine learning model that takes the entire speaker behaviour sequence as the input, and generates multiple appropriate and realistic / naturalistic spatio-temporal facial reactions, consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour. 


#### Task 2 - Online Appropriate Facial Reaction Generation
This task aims to develop a machine learning model that estimates each frame, rather than taking all frames into consideration. The model is expected to gradually generate all facial reaction frames to form multiple appropriate and realistic / naturalistic spatio-temporal facial reactions consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour. 


https://github.com/reactmultimodalchallenge/baseline_react2023/assets/35754447/8c7e7f92-d991-4741-80ec-a5112532460b

## 🛠️ Installation

### Basic requirements

- Python 3.8+ 
- PyTorch 1.9+
- CUDA 11.1+ 

### Install Python dependencies (all included in 'requirements.txt')

```shell
conda create -n react python=3.8
conda activate react
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


## 👨‍🏫 Get Started 

<details><summary> <b> Data </b> </summary>
<p>
 
**Challenge Data Description:**
- The REACT 2023 Multimodal Challenge Dataset is a compilation of recordings from the following three publicly available datasets for studying dyadic interactions: [NOXI](https://dl.acm.org/doi/10.1145/3136755.3136780), [RECOLA](https://ieeexplore.ieee.org/document/6553805) and [UDIVA](https://www.computer.org/csdl/proceedings-article/wacvw/2021/196700a001/1sZ3sn1GBxe). 

- Participants can apply for the data at our [Homepage](https://sites.google.com/cam.ac.uk/react2023/home).
   
**Data organization (`data/`) is listed below:**
```data/partition/modality/site/chat_index/person_index/clip_index/actual_data_files```
The example of data structure.
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
 
- The task is to predict one role's reaction ('Expert' or 'Novice',  'P25' or 'P26'....) to the other ('Novice' or 'Expert',  'P26' or 'P25'....).
- 3D_FV_files involve extracted 3DMM coefficients (including expression (52 dim), angle (3 dim) and translation (3 dim) coefficients.
- The frame rate of processed videos in each site is 25 (fps = 25), height = 256, width = 256. And each video clip has 751 frames. 
 
 
</p>
</details>



<details><summary> <b> External Tool Preparation </b> </summary>
<p>

We use 3DMM coefficients to represent a 3D listener or speaker, and for further 3D-to-2D frame rendering. 
 
The baselines leverage [3DMM model](https://github.com/LizhenWangT/FaceVerse) to extract 3DMM coefficients, and render 3D facial reactions.  

- You should first download 3DMM (FaceVerse version 2 model) at this [page](https://github.com/LizhenWangT/FaceVerse) 
 
  and then put it in the folder (`external/FaceVerse/data/`).
 
  We provide our extracted 3DMM coefficients (which are used for our baseline visualisation) at [Google Drive] (https://drive.google.com/drive/folders/1RrTytDkkq520qUUAjTuNdmS6tCHQnqFu). 

  We also provide the mean_face, std_face and reference_full of 3DMM coefficients at [Google Drive](https://drive.google.com/drive/folders/1uVOOJzY3p2XjDESwH4FCjGO8epO7miK4). Please put them in the folder (`external/FaceVerse/`).

 
Then, we use a 3D-to-2D tool [PIRender](https://github.com/RenYurui/PIRender) to render final 2D facial reaction frames.
 
- We re-trained the PIRender, and the well-trained model is provided at the [checkpoint](https://drive.google.com/drive/folders/1Ys1u0jxVBxrmQZrcrQbm8tagOPNxrTUA). Please put it in the folder (`external/PIRender/`).
   
</p>
</details>


<details><summary> <b> Training </b>  </summary>
<p>
 
- Running the following shell can start training:
 ```shell
 python train.py --batch-size 8  --gpu-ids 0  -lr 0.00002  -e 50  -j 12  --outdir results/train_offline
 ```
 &nbsp; or 
 
  ```shell
 python train.py --batch-size 8  --gpu-ids 0  -lr 0.00002  -e 50  -j 12  --online --outdir results/train_online
 ```
 
</p>
</details>


<details><summary> <b> Validation </b>  </summary>
<p>
 
- Before validation, run the following script to get the martix (defining appropriate neighbours in val set):
 ```shell
 cd tool
 python matrix_split.py --dataset-path ./data --partition val
 ```
&nbsp;  Please put files (`data_indices.csv`, `Approprirate_facial_reaction.npy` and `val.csv`) in the folder `./data/`.
  
- Then, evaluate a trained model on val set and run:

 ```shell
python val.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/val_offline
```
 
&nbsp; or
 
```shell
python val.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/val_online 
```
 
- For computing FID (FRRea), run the following script:

```
python -m pytorch_fid  ./results/val_offline/fid/real  ./results/val_offline/fid/fake
```
</p>
</details>




<details><summary> <b> Test </b>  </summary>
<p>
 
- Before testing, run the following script to get the martix (defining appropriate neighbours in val set):
 ```shell
 cd tool
 python matrix_split.py --dataset-path ./data --partition test
 ```
&nbsp;  Please put files (`data_indices.csv`, `Approprirate_facial_reaction.npy` and `test.csv`) in the folder `./data/`.
  
- Then, evaluate a trained model on val set and run:

 ```shell
python test.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/test_offline
```
 
&nbsp; or
 
```shell
python test.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/test_online 
```
 
- For computing FID (FRRea), run the following script:

```
python -m pytorch_fid  ./results/test_offline/fid/real  ./results/test_offline/fid/fake
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
