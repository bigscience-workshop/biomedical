
---
language: 
- en
- zh
- ja
bigbio_language: 
- English
- Chinese
- Japanese
license: cc-by-4.0
multilinguality: multilingual
bigbio_license_shortname: CC_BY_4p0
pretty_name: NTCIR-13 MedWeb
homepage: http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- TRANSLATION
- TEXT_CLASSIFICATION
---


# Dataset Card for NTCIR-13 MedWeb

## Dataset Description

- **Homepage:** http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html
- **Pubmed:** False
- **Public:** False
- **Tasks:** TRANSL,TXTCLASS


NTCIR-13 MedWeb (Medical Natural Language Processing for Web Document) task requires
to perform a multi-label classification that labels for eight diseases/symptoms must
be assigned to each tweet. Given pseudo-tweets, the output are Positive:p or Negative:n
labels for eight diseases/symptoms. The achievements of this task can almost be
directly applied to a fundamental engine for actual applications.

This task provides pseudo-Twitter messages in a cross-language and multi-label corpus,
covering three languages (Japanese, English, and Chinese), and annotated with eight
labels such as influenza, diarrhea/stomachache, hay fever, cough/sore throat, headache,
fever, runny nose, and cold.

For more information, see:
http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html

As this dataset also provides a parallel corpus of pseudo-tweets for english,
japanese and chinese it can also be used to train translation models between
these three languages.



## Citation Information

```
@article{wakamiya2017overview,
  author    = {Shoko Wakamiya, Mizuki Morita, Yoshinobu Kano, Tomoko Ohkuma and Eiji Aramaki},
  title     = {Overview of the NTCIR-13 MedWeb Task},
  journal   = {Proceedings of the 13th NTCIR Conference on Evaluation of Information Access Technologies (NTCIR-13)},
  year      = {2017},
  url       = {
    http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings13/pdf/ntcir/01-NTCIR13-OV-MEDWEB-WakamiyaS.pdf
  },
}

```
