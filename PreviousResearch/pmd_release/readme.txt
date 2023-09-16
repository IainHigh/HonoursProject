# Progressive Mirror Detection
This is the code for Progressive Mirror Detection. Visit our project homepage for any updates. (https://jiaying.link/cvpr2020-pgd/)

download link for `pmd.pth` for MSD dataset: 
https://portland-my.sharepoint.com/:u:/g/personal/jiayinlin5-c_my_cityu_edu_hk/ESQAea7Y5qxAsy5MITlR27wBe8sjuFw0yTU_MOPwDLpCUw?e=p9OdLq

download link for `pmd_split.pth` for the new splited PMD dataset: 
https://portland-my.sharepoint.com/:u:/g/personal/jiayinlin5-c_my_cityu_edu_hk/ESzmXRPp7ZROtm3s_iSQkbsBfFuWEE0ien7aGHyCIrEv7A?e=figJB6

- Some evaluation results on splited PMD:

| Methods   | F_beta | MAE   |
|-----------|--------|-------|
| EGNet     | 0.672  | 0.087 |
| MirrorNet | 0.748  | 0.061 |
| Ours      | 0.790  | 0.032 |


- Test 
download `*.pth` and then run `infer.py`.

- Contact
jiayinlin5-c@my.cityu.edu.hk

Please cite our paper if you use our code or benchmark dataset:

@INPROCEEDINGS{PMD:2020,
   Author = {Jiaying Lin and Guodong Wang and Rynson W.H. Lau},
   Title = {Progressive Mirror Detection},
   Booktitle = {Proc. CVPR},
   Year = {2020}
}
