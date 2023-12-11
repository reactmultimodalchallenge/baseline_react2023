# This is the official baseline code for REACT 2023 Multimodal Challenge
[[Homepage]](https://sites.google.com/cam.ac.uk/react2023/home)  [[Reference Paper]](https://arxiv.org/abs/2302.06514) [[Code]](https://github.com/reactmultimodalchallenge/baseline_react2023)

This repository provides baseline methods for the [REACT 2023 Multimodal Challenge](https://sites.google.com/cam.ac.uk/react2023/home)

### Baseline paper:
https://arxiv.org/abs/2306.06583

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
- The frame rate of processed videos in each site is 25 (fps = 25), height = 256, width = 256. And each video clip has 751 frames (about 30s), The samping rate of audio files is 44100. 
- The csv files for baseline training and validation dataloader are now avaliable at 'data/train.csv' and 'data/val.csv'
 
 
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
 
 <b>Trans-VAE</b>
- Running the following shell can start training Trans-VAE baseline:
 ```shell
 python train.py --batch-size 4  --gpu-ids 0  -lr 0.00001  --kl-p 0.00001 -e 50  -j 12  --outdir results/train_offline 
 ```
 &nbsp; or 
 
  ```shell
 python train.py --batch-size 4  --gpu-ids 0  -lr 0.00001  --kl-p 0.00001 -e 50  -j 12 --online  --window-size 16 --outdir results/train_online  
 ```
 
 <b>BeLFusion</b>
 - First train the variational autoencoder (VAE):
```shell
python train_belfusion.py config=config/1_belfusion_vae.yaml name=All_VAEv2_W50
```
 
 - Once finished, you will be able to train the offline/online variants of BeLFusion with the desired value for k:
```shell
python train_belfusion.py config=config/2_belfusion_ldm.yaml name=<NAME> arch.args.k=<INT (1 or 10)> arch.args.online=<BOOL>
```

 
</p>
</details>

<details><summary> <b> Pretrained weights </b>  </summary>
 If you would rather skip training, download the following checkpoints and put them inside the folder './results'.
<p>
 
 <b>Trans-VAE</b>: [download](https://drive.google.com/drive/folders/1tyLQnQj1e2SMArBkc3gHDZVHwSr_GEod?usp=share_link)
 
 <b>BeLFusion</b>: [download](https://ubarcelona-my.sharepoint.com/:f:/g/personal/germanbarquero_ub_edu/EvF9K27g_DFPp2MS_8OqkmwBYGzUKs7J3QmkidbRLVSt6Q?e=WCJ2JU)
 
</details>

<details><summary> <b> Validation </b>  </summary>
<p>
 Follow this to evaluate Trans-VAE or BeLFusion after training, or downloading the pretrained weights.
 
- Before validation, run the following script to get the martix (defining appropriate neighbours in val set):
 ```shell
 cd tool
 python matrix_split.py --dataset-path ./data --partition val
 ```
&nbsp;  Please put files (`data_indices.csv`, `Approprirate_facial_reaction.npy` and `val.csv`) in the folder `./data/`.
  
- Then, evaluate a trained model on val set and run:

 ```shell
python evaluate.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/val_offline --split val
```
 
&nbsp; or
 
```shell
python evaluate.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/val_online --split val
```
 
- For computing FID (FRRea), run the following script:

```
python -m pytorch_fid  ./results/val_offline/fid/real  ./results/val_offline/fid/fake
```
</p>
</details>




<details><summary> <b> Test </b>  </summary>
<p>
 Follow this to evaluate Trans-VAE or BeLFusion after training, or downloading the pretrained weights.
 
- Before testing, run the following script to get the martix (defining appropriate neighbours in test set):
 ```shell
 cd tool
 python matrix_split.py --dataset-path ./data --partition test
 ```
&nbsp;  Please put files (`data_indices.csv`, `Approprirate_facial_reaction.npy` and `test.csv`) in the folder `./data/`.
  
- Then, evaluate a trained model on test set and run:

 ```shell
python evaluate.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/test_offline --split test
```
 
&nbsp; or
 
```shell
python evaluate.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/test_online --split test
```

 
- For computing FID (FRRea), run the following script:

```
python -m pytorch_fid  ./results/test_offline/fid/real  ./results/test_offline/fid/fake
```
</p>
</details>



<details><summary> <b> Other baselines </b>  </summary>
<p>
 
- Run the following script to sequentially evaluate the naive baselines presented in the paper:
 ```shell
 python run_baselines.py --split SPLIT
 ```
 SPLIT can be `val` or `test`.
</p>
</details>



## 🖊️ Citation

### Submissions should cite the following papers:

#### Theory paper and baseline paper:

[1] Song, Siyang, Micol Spitale, Yiming Luo, Batuhan Bal, and Hatice Gunes. "Multiple Appropriate Facial Reaction Generation in Dyadic Interaction Settings: What, Why and How?." arXiv preprint arXiv:2302.06514 (2023).

[2] Song, Siyang, Micol Spitale, Cheng Luo, German Barquero, Cristina Palmero, Sergio Escalera, Michel Valstar et al. "REACT2023: the first Multi-modal Multiple Appropriate Facial Reaction Generation Challenge." arXiv preprint arXiv:2306.06583 (2023).

#### Dataset papers:

[3] Palmero, C., Selva, J., Smeureanu, S., Junior, J., Jacques, C. S., Clapés, A., ... & Escalera, S. (2021). Context-aware personality inference in dyadic scenarios: Introducing the udiva dataset. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1-12).

[4] Ringeval, F., Sonderegger, A., Sauer, J., & Lalanne, D. (2013, April). Introducing the RECOLA multimodal corpus of remote collaborative and affective interactions. In 2013 10th IEEE international conference and workshops on automatic face and gesture recognition (FG) (pp. 1-8). IEEE.

[5] Cafaro, A., Wagner, J., Baur, T., Dermouche, S., Torres Torres, M., Pelachaud, C., ... & Valstar, M. (2017, November). The NoXi database: multimodal recordings of mediated novice-expert interactions. In Proceedings of the 19th ACM International Conference on Multimodal Interaction (pp. 350-359).

#### Annotation, basic feature extraction tools and baselines:

[6] Song, Siyang, Yuxin Song, Cheng Luo, Zhiyuan Song, Selim Kuzucu, Xi Jia, Zhijiang Guo, Weicheng Xie, Linlin Shen, and Hatice Gunes. "GRATIS: Deep Learning Graph Representation with Task-specific Topology and Multi-dimensional Edge Features." arXiv preprint arXiv:2211.12482 (2022).

[7] Luo, Cheng, Siyang Song, Weicheng Xie, Linlin Shen, and Hatice Gunes. (2022, July) "Learning multi-dimensional edge feature-based au relation graph for facial action unit recognition." Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (pp. 1239-1246).

[8] Toisoul, Antoine, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, and Maja Pantic. "Estimation of continuous valence and arousal levels from faces in naturalistic conditions." Nature Machine Intelligence 3, no. 1 (2021): 42-50.

[9] Eyben, Florian, Martin Wöllmer, and Björn Schuller. "Opensmile: the munich versatile and fast open-source audio feature extractor." In Proceedings of the 18th ACM international conference on Multimedia, pp. 1459-1462. 2010.

[10] Barquero, German, Sergio Escalera, and Cristina Palmero. "BeLFusion: Latent Diffusion for Behavior-Driven Human Motion Prediction." arXiv preprint arXiv:2211.14304 (2022). 


### Submissions are encouraged to cite previous facial reaction generation papers:

[1] Huang, Yuchi, and Saad M. Khan. "Dyadgan: Generating facial expressions in dyadic interactions." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 11-18. 2017.

[2] Huang, Yuchi, and Saad Khan. "A generative approach for dynamically varying photorealistic facial expressions in human-agent interactions." In Proceedings of the 20th ACM International Conference on Multimodal Interaction, pp. 437-445. 2018.

[3] Shao, Zilong, Siyang Song, Shashank Jaiswal, Linlin Shen, Michel Valstar, and Hatice Gunes. "Personality recognition by modelling person-specific cognitive processes using graph representation." In proceedings of the 29th ACM international conference on multimedia, pp. 357-366. 2021.

[4] Song, Siyang, Zilong Shao, Shashank Jaiswal, Linlin Shen, Michel Valstar, and Hatice Gunes. "Learning Person-specific Cognition from Facial Reactions for Automatic Personality Recognition." IEEE Transactions on Affective Computing (2022).

[5] Ng, Evonne, Hanbyul Joo, Liwen Hu, Hao Li, Trevor Darrell, Angjoo Kanazawa, and Shiry Ginosar. "Learning to listen: Modeling non-deterministic dyadic facial motion." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20395-20405. 2022.

[6] Zhou, Mohan, Yalong Bai, Wei Zhang, Ting Yao, Tiejun Zhao, and Tao Mei. "Responsive listening head generation: a benchmark dataset and baseline." In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVIII, pp. 124-142. Cham: Springer Nature Switzerland, 2022.

[7] Luo, Cheng, Siyang Song, Weicheng Xie, Micol Spitale, Linlin Shen, and Hatice Gunes. "ReactFace: Multiple Appropriate Facial Reaction Generation in Dyadic Interactions." arXiv preprint arXiv:2305.15748 (2023).

[8] Xu, Tong, Micol Spitale, Hao Tang, Lu Liu, Hatice Gunes, and Siyang Song. "Reversible Graph Neural Network-based Reaction Distribution Learning for Multiple Appropriate Facial Reactions Generation." arXiv preprint arXiv:2305.15270 (2023).

## 🤝 Acknowledgement
Thanks to the open source of the following projects:

- [FaceVerse](https://github.com/LizhenWangT/FaceVerse) &#8194;

- [PIRender](https://github.com/RenYurui/PIRender) &#8194;
