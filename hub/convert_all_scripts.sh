#!/usr/bin/env bash

SOURCEDIR="/home/natasha/Projects/hfbiomed/biomedical/bigbio/biodatasets"
SAVEDIR="/home/natasha/Projects/hfbiomed/biomedical/hub"

for dset in anat_em an_em ask_a_patient bc5cdr bc7_litcovid bioasq_2021_mesinesp bioasq_task_b bioasq_task_c_2017 bioinfer biology_how_why_corpus biomrc bionlp_shared_task_2009 bionlp_st_2011_epi bionlp_st_2011_ge bionlp_st_2011_id bionlp_st_2011_rel bionlp_st_2013_cg bionlp_st_2013_ge bionlp_st_2013_gro bionlp_st_2013_pc bionlp_st_2019_bb biored biorelex bioscope bio_simlex bio_sim_verb biosses blurb cadec cantemist cas cellfinder chebi_nactem chemdner chemprot chia citation_gia_test_collection codiesp cord_ner ctebmsp ddi_corpus diann_iber_eval distemist ebm_pico ehr_rel essai euadr evidence_inference gad genetag genia_ptm_event_corpus genia_relation_corpus genia_term_corpus geokhoj_v1 gnormplus hallmarks_of_cancer hprd50 iepa jnlpba linnaeus lll mantra_gsc mayosrs medal meddialog meddocan medhop medical_data mediqa_nli mediqa_qa mediqa_rqe medmentions mednli med_qa meqsum minimayosrs mirna mlee mqp msh_wsd muchmore multi_xscience mutation_finder n2c2_2006_deid n2c2_2006_smokers n2c2_2008 n2c2_2009 n2c2_2010 n2c2_2011 n2c2_2014_deid n2c2_2014_risk_factors n2c2_2018_track1 n2c2_2018_track2 nagel ncbi_disease nlmchem nlm_gene nlm_wsd ntcir_13_medweb osiris paramed pcr pdr pharmaconer pho_ner pico_extraction pmc_patients progene psytar pubhealth pubmed_qa pubtator_central quaero scai_chemical scai_disease scicite scielo scifact sciq scitail seth_corpus spl_adr_200db swedish_medical_ner thomas2011 tmvar_v1 tmvar_v2 tmvar_v3 twadrl umnsrs verspoor_2013; do
    
    TARGET=$SOURCEDIR"/"$dset"/"$dset".py"
    SAVENAME=$SAVEDIR"/"$dset"_hub.py"
    echo $dset
    python make_hub_script.py --script $TARGET --savename
    echo "------------"
done