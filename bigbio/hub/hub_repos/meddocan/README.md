
---
language: 
- es
bigbio_language: 
- Spanish
license: cc-by-4.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_4p0
pretty_name: MEDDOCAN
homepage: https://temu.bsc.es/meddocan/
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for MEDDOCAN

## Dataset Description

- **Homepage:** https://temu.bsc.es/meddocan/
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER


MEDDOCAN: Medical Document Anonymization Track

This dataset is designed for the MEDDOCAN task, sponsored by Plan de Impulso de las Tecnologías del Lenguaje.

It is a manually classified collection of 1,000 clinical case reports derived from the Spanish Clinical Case Corpus (SPACCC), enriched with PHI expressions.

The annotation of the entire set of entity mentions was carried out by experts annotatorsand it includes 29 entity types relevant for the annonymiation of medical documents.22 of these annotation types are actually present in the corpus: TERRITORIO, FECHAS, EDAD_SUJETO_ASISTENCIA, NOMBRE_SUJETO_ASISTENCIA, NOMBRE_PERSONAL_SANITARIO, SEXO_SUJETO_ASISTENCIA, CALLE, PAIS, ID_SUJETO_ASISTENCIA, CORREO, ID_TITULACION_PERSONAL_SANITARIO,ID_ASEGURAMIENTO, HOSPITAL, FAMILIARES_SUJETO_ASISTENCIA, INSTITUCION, ID_CONTACTO ASISTENCIAL,NUMERO_TELEFONO, PROFESION, NUMERO_FAX, OTROS_SUJETO_ASISTENCIA, CENTRO_SALUD, ID_EMPLEO_PERSONAL_SANITARIO
    
For further information, please visit https://temu.bsc.es/meddocan/ or send an email to encargo-pln-life@bsc.es



## Citation Information

```
@inproceedings{marimon2019automatic,
  title={Automatic De-identification of Medical Texts in Spanish: the MEDDOCAN Track, Corpus, Guidelines, Methods and Evaluation of Results.},
  author={Marimon, Montserrat and Gonzalez-Agirre, Aitor and Intxaurrondo, Ander and Rodriguez, Heidy and Martin, Jose Lopez and Villegas, Marta and Krallinger, Martin},
  booktitle={IberLEF@ SEPLN},
  pages={618--638},
  year={2019}
}

```
