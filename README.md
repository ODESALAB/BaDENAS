BaDENAS: Bayesian Based Neural Architecture Search for Retinal Vessel Segmentation
===
Abstract: Retinal vessel segmentation is an important task for analyzing retinal images and is an effective tool used in diagnosing and treating eye diseases. Although deep learning methods like U-Net that automate vessel segmentation have shown promising results in this field, they have many hyper-parameters that need to be optimized. Neural architecture search (NAS) is commonly used to optimize these hyper-parameters. This study proposes a new neural architecture search method for U-shaped networks by combining the advantages of BANANAS and the Differential Evolution (DE) algorithm: BaDENAS. Comparisons made with various neural architecture search studies show that BaDENAS improves convergence, segmentation performance, and model complexity results. Additionally, the proposed method produces the least complex model and achieves highly competitive results, with a model having up to 152 times fewer parameters than other neural architecture search methods.

## Paper Information
- Title:  [BaDENAS: Bayesian Based Neural Architecture Search for Retinal Vessel Segmentation](https://doi.org/10.1109/SIU59756.2023.10223862)
- Authors:  `Zeki Kuş`,`Berna Kiraz`

## Dataset

DRIVE: [Link](https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0)<br>
You should put the DRIVE folder into the ./DataSets/.

## Use
- for short-term evaluation
  ```
  please look into ./bananas.py
  please look into ./badenas.py
  please look into ./de.py
  ```
- for long-term evaluation
  ```
  please look into ./eval_model.py
  ```
- for F1, ACC, SEN, SPE, IOU Evaluation
  ```
  please look into ./figures.py
  ```
  
To cite the paper or code:
```bibtex
@INPROCEEDINGS{10223862,
  author={Kuş, Zeki and Kiraz, Berna},
  booktitle={2023 31st Signal Processing and Communications Applications Conference (SIU)}, 
  title={BaDENAS: Bayesian Based Neural Architecture Search for Retinal Vessel Segmentation}, 
  year={2023},
  volume={},
  number={},
  pages={1-4},
  doi={10.1109/SIU59756.2023.10223862}}
```
