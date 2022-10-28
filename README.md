# LocatorModel

Data and Source code for Paper accepted in COLING 2022.

[Where to Attack: A Dynamic Locator Model for Backdoor Attack in Text Classifications](https://aclanthology.org/2022.coling-1.82/)

## Abstract

Nowadays, deep-learning based NLP models are usually trained with large-scale third-party data which can be easily injected with malicious backdoors. Thus, BackDoor Attack (BDA) study has become a trending research to help promote the robustness of an NLP system. Text-based BDA aims to train a poisoned model with both clean and poisoned texts to perform normally on clean inputs while being misled to predict those trigger-embedded texts as target labels set by attackers. Previous works usually choose fixed Positions-to-Poison (P2P) first, then add triggers upon those positions such as letter insertion or deletion. However, considering the positions of words with important semantics may vary in different contexts, fixed P2P models are severely limited in flexibility and performance. We study the text-based BDA from the perspective of automatically and dynamically selecting P2P from contexts. We design a novel Locator model which can predict P2P dynamically without human intervention. Based on the predicted P2P, four effective strategies are introduced to show the BDA performance. Experiments on two public datasets show both tinier test accuracy gap on clean data and higher attack success rate on poisoned ones. Human evaluation with volunteers also shows the P2P predicted by our model are important for classification.

## 2022-10-26

We have uploaded our Data in Dataset/

The codes will be uploaded ASAP ...

## 2022-10-28

We have uploaded the early version of the code.

We are still optimizing a better version, and will updated later.

Thank you.

...

# Cite information
```
@inproceedings{lu-etal-2022-attack,
    title = "Where to Attack: A Dynamic Locator Model for Backdoor Attack in Text Classifications",
    author = "Lu, Heng-yang  and
      Fan, Chenyou  and
      Yang, Jun  and
      Hu, Cong  and
      Fang, Wei  and
      Wu, Xiao-jun",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.82",
    pages = "984--993",
}
```
