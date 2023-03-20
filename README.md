# Feature Learning

This repository is a modified implementation of:

DeTone, Daniel, Tomasz Malisiewicz, and Andrew Rabinovich. "Superpoint: Self-supervised interest point detection and description." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*. 2018. [CVF Open Access Link](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html).

It is based on another implementation by  You-Yi Jau and Rui Zhu: https://github.com/eric-yyjau/pytorch-superpoint

The code has been developed as part of the european project [EndoMapper](https://cordis.europa.eu/project/id/863146).

## Feature extraction in endoscopy videos

Consecutive frames | Frames 1 second apart
-------|-------
![v33_s14_31_32](doc/original_v33_s14_31_32.png) | ![v33_s14_31_70](doc/original_v33_s14_31_70.png)
![v33_s19_1_2](doc/original_v33_s19_1_2.png) | ![v33_s19_1_40](doc/original_v33_s19_1_40.png)
<!---
![v33_s19_36_37](doc/original_v33_s19_36_37.png) | ![v33_s19_36_75](doc/original_v33_s19_36_75.png)
![v89_s39_41_42](doc/original_v89_s39_41_42.png) | ![v89_s39_41_80](doc/original_v89_s39_41_80.png)
![v34_s3_15_16](doc/original_v34_s3_15_16.png) | ![v34_s3_15_54](doc/original_v34_s3_15_54.png)
--->

## Results

### Well-known local feature detectors

SIFT | ORB | SuperPoint
-------|-------|-------
![SIFT](doc/sift_v33_s14_31_32.png) | ![ORB](doc/orb_v33_s14_31_32.png) | ![SuperPoint](doc/sp_v33_s14_31_32.png)
![SIFT](doc/sift_v33_s14_31_70.png) | ![ORB](doc/orb_v33_s14_31_70.png) | ![SuperPoint](doc/sp_v33_s14_31_70.png)
![SIFT](doc/sift_v33_s19_1_2.png) | ![ORB](doc/orb_v33_s19_1_2.png) | ![SuperPoint](doc/sp_v33_s19_1_2.png)
![SIFT](doc/sift_v33_s19_1_40.png) | ![ORB](doc/orb_v33_s19_1_40.png) | ![SuperPoint](doc/sp_v33_s19_1_40.png)

### Our models

E-SuperPoint | E-SuperPoint+S
-------|-------
![E-SuperPoint](doc/spft_v33_s14_31_32.png) | ![E-SuperPoint+S](doc/spspec_v33_s14_31_32.png)
![E-SuperPoint](doc/spft_v33_s14_31_70.png) | ![E-SuperPoint+S](doc/spspec_v33_s14_31_70.png)
![E-SuperPoint](doc/spft_v33_s19_1_2.png) | ![E-SuperPoint+S](doc/spspec_v33_s19_1_2.png)
![E-SuperPoint](doc/spft_v33_s19_1_40.png) | ![E-SuperPoint+S](doc/spspec_v33_s19_1_40.png)


## Authors

Developed by [Óscar León Barbed](https://github.com/leonbp), [Ana C. Murillo](https://github.com/anacmurillo) & [François Chadebecq](https://github.com/FChadebecq).
