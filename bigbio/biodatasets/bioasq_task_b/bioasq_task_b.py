# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
BioASQ Task B On Biomedical Semantic QA (Involves IR, QA, Summarization qnd
More). This task uses benchmark datasets containing development and test
questions, in English, along with gold standard (reference) answers constructed
by a team of biomedical experts. The participants have to respond with relevant
concepts, articles, snippets and RDF triples, from designated resources, as well
as exact and 'ideal' answers.

Fore more information about the challenge, the organisers and the relevant
publications please visit: http://bioasq.org/
"""
import glob
import json
import os
import re

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [
    Tags.YESNO,
    Tags.FACTOID,
    Tags.FACTOID_LIST,
    Tags.ABSTRACTIVE,
    Tags.EXTRACTIVE,
]
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = True
_CITATION = """\
@article{tsatsaronis2015overview,
	title        = {
		An overview of the BIOASQ large-scale biomedical semantic indexing and
		question answering competition
	},
	author       = {
		Tsatsaronis, George and Balikas, Georgios and Malakasiotis, Prodromos
        and Partalas, Ioannis and Zschunke, Matthias and Alvers, Michael R and
		Weissenborn, Dirk and Krithara, Anastasia and Petridis, Sergios and
		Polychronopoulos, Dimitris and others
	},
	year         = 2015,
	journal      = {BMC bioinformatics},
	publisher    = {BioMed Central Ltd},
	volume       = 16,
	number       = 1,
	pages        = 138
}
"""

_DATASETNAME = "bioasq"

_BIOASQ_10B_DESCRIPTION = """\
The data are intended to be used as training and development data for BioASQ
10, which will take place during 2022. There is one file containing the data:
 - training10b.json

The file contains the data of the first nine editions of the challenge: 4234
questions [1] with their relevant documents, snippets, concepts and RDF
triples, exact and ideal answers.

Differences with BioASQ-training9b.json
- 492 new questions added from BioASQ9
    - The question with id 56c1f01eef6e394741000046 had identical body with
    602498cb1cb411341a00009e. All relevant elements from both questions
    are available in the merged question with id 602498cb1cb411341a00009e.
    - The question with id 5c7039207c78d69471000065 had identical body with
    601c317a1cb411341a000014. All relevant elements from both questions
    are available in the merged question with id 601c317a1cb411341a000014.
	- The question with id 5e4b540b6d0a27794100001c had identical body with
    602828b11cb411341a0000fc. All relevant elements from both questions
    are available in the merged question with id 602828b11cb411341a0000fc.
    - The question with id 5fdb42fba43ad31278000027 had identical body with
    5d35eb01b3a638076300000f. All relevant elements from both questions
    are available in the merged question with id 5d35eb01b3a638076300000f.
    - The question with id 601d76311cb411341a000045 had identical body with
    6060732b94d57fd87900003d. All relevant elements from both questions
    are available in the merged question with id 6060732b94d57fd87900003d.

[1] 4234 questions : 1252 factoid, 1148 yesno, 1018 summary, 816 list
"""

_BIOASQ_9B_DESCRIPTION = """\
The data are intended to be used as training and development data for BioASQ 9,
which will take place during 2021. There is one file containing the data:
 - training9b.json

The file contains the data of the first seven editions of the challenge: 3742
questions [1] with their relevant documents, snippets, concepts and RDF triples,
exact and ideal answers.

Differences with BioASQ-training8b.json
- 499 new questions added from BioASQ8
    - The question with id 5e30e689fbd6abf43b00003a had identical body with
    5880e417713cbdfd3d000001. All relevant elements from both questions
    are available in the merged question with id 5880e417713cbdfd3d000001.

[1] 3742 questions : 1091 factoid, 1033 yesno, 899 summary, 719 list
"""

_BIOASQ_8B_DESCRIPTION = """\
The data are intended to be used as training and development data for BioASQ 8,
which will take place during 2020. There is one file containing the data:
 - training8b.json

The file contains the data of the first seven editions of the challenge: 3243
questions [1] with their relevant documents, snippets, concepts and RDF triples,
exact and ideal answers.

Differences with BioASQ-training7b.json
- 500 new questions added from BioASQ7
- 4 questions were removed
    - The question with id 5717fb557de986d80d000009 had identical body with
    571e06447de986d80d000016. All relevant elements from both questions
    are available in the merged question with id 571e06447de986d80d000016.
    - The question with id 5c589ddb86df2b917400000b had identical body with
    5c6b7a9e7c78d69471000029. All relevant elements from both questions
    are available in the merged question with id 5c6b7a9e7c78d69471000029.
    - The question with id 52ffb5d12059c6d71c00007c had identical body with
    52e7870a98d023950500001a. All relevant elements from both questions
    are available in the merged question with id 52e7870a98d023950500001a.
    - The question with id 53359338d6d3ac6a3400004f had identical body with
    589a246878275d0c4a000030. All relevant elements from both questions
    are available in the merged question with id 589a246878275d0c4a000030.

**** UPDATE 25/02/2020 *****
The previous version of the dataset contained an inconsistency on question with
id "5c9904eaecadf2e73f00002e", where the "ideal_answer" field was missing.
This has been fixed.
"""

_BIOASQ_7B_DESCRIPTION = """\
The data are intended to be used as training and development data for BioASQ 7,
which will take place during 2019. There is one file containing the data:
 - BioASQ-trainingDataset7b.json

The file contains the data of the first six editions of the challenge: 2747
questions [1] with their relevant documents, snippets, concepts and RDF triples,
exact and ideal answers.

Differences with BioASQ-trainingDataset6b.json
- 500 new questions added from BioASQ6
    - 4 questions were removed
        - The question with id 569ed752ceceede94d000004 had identical body with
        a new question from BioASQ6. All relevant elements from both questions
        are available in the merged question with id 5abd31e0fcf456587200002c
        - 3 questions were removed as incomplete: 54d643023706e89528000007,
        532819afd6d3ac6a3400000f, 517545168ed59a060a00002b
    - 4 questions were revised for various confusions that have been identified
        - In 2 questions the ideal answer has been revised :
        51406e6223fec90375000009, 5172f8118ed59a060a000019
        - In 4 questions the snippets and documents list has been revised :
        51406e6223fec90375000009, 5172f8118ed59a060a000019,
        51593dc8d24251bc05000099, 5158a5b8d24251bc05000097
        - In 198 questions the documents list has updated with missing
        documents from the relevant snippets list. [2]

[1] 2747 questions : 779 factoid, 745 yesno, 667 summary, 556 list
[2] 55031181e9bde69634000014, 51406e6223fec90375000009, 54d643023706e89528000007,
    52bf1b0a03868f1b06000009, 52bf19c503868f1b06000001, 51593dc8d24251bc05000099,
    530a5117970c65fa6b000007, 553a8d78f321868558000003, 531a3fe3b166e2b806000038,
    532819afd6d3ac6a3400000f, 5158a5b8d24251bc05000097, 553653a5bc4f83e828000007,
    535d2cf09a4572de6f000004, 53386282d6d3ac6a3400005a, 517a8ce98ed59a060a000045,
    55391ce8bc4f83e828000018, 5547d700f35db75526000007, 5713bf261174fb1755000011,
    6f15c5a2ac5ed1459000012, 52b2e498f828ad283c000010, 570a7594cf1c325851000026,
    530cefaaad0bf1360c000012, 530f685c329f5fcf1e000002, 550c4011a103b78016000009,
    552faababc4f83e828000005, 54cf48acf693c3b16b00000b, 550313aae9bde6963400001f,
    551177626a8cde6b72000005, 54eded8c94afd6150400000c, 550c3754a103b78016000007,
    56f555b609dd18d46b000007, 54c26e29f693c3b16b000003, 54da0c524b1fd0d33c00000b,
    52bf1d3c03868f1b0600000d, 5343bdd6aeec6fbd07000001, 52cb9b9b03868f1b0600002d,
    55423875ec76f5e50c000002, 571366ba1174fb1755000005, 56c4d14ab04e159d0e000003,
    550c44d1a103b7801600000a, 5547a01cf35db75526000005, 55422640ccca0ce74b000004,
    54ecb66d445c3b5a5f000002, 553656c4bc4f83e828000009, 5172f8118ed59a060a000019,
    513711055274a5fb0700000e, 54d892ee014675820d000005, 52e6c92598d0239505000019,
    5353aedb288f4dae47000006, 52bf1f1303868f1b06000014, 5519113b622b19434500000f,
    52b2f1724003448f5500000b, 5525317687ecba3764000007, 554a0cadf35db7552600000f,
    55152bd246478f2f2c000002, 516c3960298dcd4e51000073, 571e417bbb137a4b0c00000a,
    551910d3622b194345000008, 54dc8ed6c0bb8dce23000002, 511a4ec01159fa8212000004,
    54d8ea2c4b1fd0d33c000002, 5148e1d6d24251bc0500003a, 515dbb3b298dcd4e51000018,
    56f7c15a09dd18d46b000012, 51475d5cd24251bc0500001b, 54db7c4ac0bb8dce23000001,
    57152ebbcb4ef8864c000002, 57134d511174fb1755000002, 55149f156a8cde6b72000013,
    56bcd422d36b5da378000005, 54ede5c394afd61504000006, 517545168ed59a060a00002b,
    5710ed19a5ed216440000003, 53442472aeec6fbd07000008, 55088e412e93f0133a000001,
    54d762653706e89528000014, 550aef0ec2af5d5b7000000a, 552435602c8b63434a000009,
    552446612c8b63434a00000c, 54d901ec4b1fd0d33c000006, 54cf45e7f693c3b16b00000a,
    52fc8b772059c6d71c00006e, 5314d05adae131f84700000d, 5512c91b6a8cde6b7200000b,
    56c5a7605795f9a73e000002, 55030a6ce9bde6963400000f, 553fac39c6a5098552000001,
    531a3a58b166e2b806000037, 5509bd6a1180f13250000002, 54f9c40ddd3fc62544000001,
    553c8fd1f32186855800000a, 56bce51cd36b5da37800000a, 550316a6e9bde69634000029,
    55031286e9bde6963400001b, 536e46f27d100faa09000012, 5502abd1e9bde69634000008,
    551af9106b348bb82c000002, 54edeb4394afd6150400000b, 5717cdd2070aa3d072000001,
    56c5ade15795f9a73e000003, 531464a6e3eabad021000014, 58a0d87a78275d0c4a000053,
    58a3160d60087bc10a00000a, 58a5d54860087bc10a000025, 58a0da5278275d0c4a000054,
    58a3264e60087bc10a00000d, 589c8ef878275d0c4a000042, 58a3428d60087bc10a00001b,
    58a3196360087bc10a00000b, 58a341eb60087bc10a000018, 58a3275960087bc10a00000f,
    58a342e760087bc10a00001c, 58bd645702b8c60953000010, 58bc8e5002b8c60953000006,
    58bc8e7a02b8c60953000007, 58a1da4e78275d0c4a000059, 58bcb83d02b8c6095300000f,
    58bc9a5002b8c60953000008, 589dee3778275d0c4a000050, 58a32efe60087bc10a000013,
    58a327bf60087bc10a000011, 58bca08702b8c6095300000a, 58bc9dbb02b8c60953000009,
    58c99fcc02b8c60953000029, 58bca2f302b8c6095300000c, 58cbf1f402b8c60953000036,
    58cdb41302b8c60953000042, 58cdb80302b8c60953000043, 58cdbaf302b8c60953000044,
    58cb305c02b8c60953000032, 58caf86f02b8c60953000030, 58c1b2f702b8c6095300001e,
    58bde18b02b8c60953000014, 58eb7898eda5a57672000006, 58caf88c02b8c60953000031,
    58e11bf76fddd3e83e00000c, 58cdbbd102b8c60953000045, 58df779d6fddd3e83e000001,
    58dbb4f08acda3452900001a, 58dbb8968acda3452900001b, 58add7699ef3c34033000009,
    58dbbbf08acda3452900001d, 58dbba438acda3452900001c, 58dd2cb08acda34529000029,
    58eb9542eda5a57672000007, 58f3ca5c70f9fc6f0f00000d, 58e9e7aa3e8b6dc87c00000d,
    58e3d9ab3e8b6dc87c000002, 58eb4ce7eda5a57672000004, 58f3c8f470f9fc6f0f00000c,
    58f3c62970f9fc6f0f00000b, 58adca6d9ef3c34033000007, 58f4b3ee70f9fc6f0f000013,
    593ff22b70f9fc6f0f000023, 5a679875b750ff4455000004, 5a774585faa1ab7d2e000005,
    5a6f7245b750ff4455000050, 5a787544faa1ab7d2e00000b, 5a74d9980384be9551000008,
    5a6a02a3b750ff4455000021, 5a6e47b1b750ff4455000049, 5a87124561bb38fb24000001,
    5a6e42f1b750ff4455000046, 5a8b1264fcd1d6a10c00001d, 5a981e66fcd1d6a10c00002f,
    5a8718c861bb38fb24000008, 5a7615af83b0d9ea6600001f, 5a87140a61bb38fb24000003,
    5a77072c9e632bc06600000a, 5a897601fcd1d6a10c000008, 5a871a6861bb38fb24000009,
    5a74e9ad0384be955100000a, 5a79d25dfaa1ab7d2e00000f, 5a6900ebb750ff445500001d,
    5a87145861bb38fb24000004, 5a871b8d61bb38fb2400000a, 5a897a06fcd1d6a10c00000b,
    5a8dc6b4fcd1d6a10c000026, 5a8712af61bb38fb24000002, 5a8714e261bb38fb24000005,
    5aa304f1d6d6b54f79000004, 5a981bcffcd1d6a10c00002d, 5aa3fa73d6d6b54f79000008,
    5aa55b45d6d6b54f7900000d, 5a981dd0fcd1d6a10c00002e, 5a9700adfcd1d6a10c00002c,
    5a9d8ffe1d1251d03b000022, 5a96c74cfcd1d6a10c000029, 5aa50086d6d6b54f7900000c,
    5a95765bfcd1d6a10c000028, 5a96f40cfcd1d6a10c00002b, 5ab144fefcf4565872000012,
    5aa67b4fd6d6b54f7900000f, 5abd5a62fcf4565872000031, 5abbe429fcf456587200001c,
    5aaef38dfcf456587200000f, 5abce6acfcf4565872000022, 5aae6499fcf456587200000c
"""

_BIOASQ_6B_DESCRIPTION = """\
The data are intended to be used as training and development data for BioASQ 6,
which will take place during 2018. There is one file containing the data:
 - BioASQ-trainingDataset6b.json

Differences with BioASQ-trainingDataset5b.json
- 500 new questions added from BioASQ5
    - 48 pairs of questions with identical bodies have been merged into one
    question having only one question-id, but all the documents, snippets,
    concepts, RDF triples and answers of both questions of the pair.
        - This normalization lead to the removal of 48 deprecated question
        ids [2] from the dataset and to the update of the 48 remaining
        questions [3].
        - In cases where a pair of questions with identical bodies had some
        inconsistency (e.g. different question type), the inconsistency has
        been solved merging the pair manually consulting the BioASQ expert team.
    - 12 questions were revised for various confusions that have been
    identified
        - In 8 questions the question type has been changed to better suit to
        the question body. The change of type lead to corresponding changes
        in exact answers existence and format : 54fc4e2e6ea36a810c000003,
        530b01a6970c65fa6b000008, 530cf54dab4de4de0c000009,
        531b2fc3b166e2b80600003c, 532819afd6d3ac6a3400000f,
        532aad53d6d3ac6a34000010, 5710ade4cf1c32585100002c,
        52f65f372059c6d71c000027
		- In 6 questions the ideal answer has been revised :
        532aad53d6d3ac6a34000010, 5710ade4cf1c32585100002c,
        53147b52e3eabad021000015, 5147c8a6d24251bc05000027,
        5509bd6a1180f13250000002, 58bbb71f22d3005309000016
		- In 5 questions the exact answer has been revised :
        5314bd7ddae131f847000006, 53130a77e3eabad02100000f,
        53148a07dae131f847000002, 53147b52e3eabad021000015,
        5147c8a6d24251bc05000027
		- In 2 questions the question body has been revised :
        52f65f372059c6d71c000027, 5503145ee9bde69634000022
    - In lists of ideal answers, documents, snippets, concepts and RDF triples
    any duplicate identical elements have been removed.
    - Ideal answers in format of one string have been converted to a list with
    one element for consistency with cases where more than one golden ideal
    answers are available. (i.e. "ideal_ans1" converted to ["ideal_ans1"])
    - For yesno questions: All exact answers have been normalized to "yes" or
    "no" (replacing "Yes", "YES" and "No")
    - For factoid questions: The format of the exact answer was normalized to a
    list of strings for each question, representing a set of synonyms
    answering the question (i.e. [`ans1`, `syn11`, ... ]).
    - For list questions: The format of the exact answer was normalized to a
    list of lists. Each internal list represents one element of the answer
    as a set of synonyms
    (i.e. [[`ans1`, `syn11`, `syn12`], [`ans2`], [`ans3`, `syn31`] ...]).
    - Empty elements, e.g. empty lists of documents have been removed.

[1] 2251 questions : 619 factoid, 616 yesno, 531 summary, 485 list
[2] The 48 deprecated question ids are : 52f8b2902059c6d71c000053,
    52f11bf22059c6d71c000005, 52f77edb2059c6d71c000028, 52ed795098d0239505000032,
    56d1a9baab2fed4a47000002, 52f7d3472059c6d71c00002f, 52fbe2bf2059c6d71c00006c,
    52ec961098d023950500002a, 52e8e98298d0239505000020, 56cae5125795f9a73e000024,
    530cefaaad0bf1360c000007, 530cefaaad0bf1360c000005, 52d63b2803868f1b0600003a,
    530cefaaad0bf1360c00000a, 516425ff298dcd4e51000051, 55191149622b194345000010,
    52fa70142059c6d71c000056, 52f77f4d2059c6d71c00002a, 52efc016c8da89891000001a,
    52efc001c8da898910000019, 52f896ae2059c6d71c000045, 52eceada98d023950500002d,
    52efc05cc8da89891000001c, 515e078e298dcd4e51000031, 52fe54252059c6d71c000079,
    514217a6d24251bc05000005, 52d1389303868f1b06000032, 530cf4d5e2bfff940c000003,
    52fc946d2059c6d71c000071, 52e8e99e98d0239505000021, 52ef7786c8da898910000015,
    52d8494698d0239505000007, 530cf51d5610acba0c000001, 52f637972059c6d71c000025,
    52e9f99798d0239505000025, 515de572298dcd4e51000021, 52fe4ad52059c6d71c000077,
    52f65bf02059c6d71c000026, 52e8e9d298d0239505000022, 52fa74052059c6d71c00005a,
    52ffbddf2059c6d71c00007d, 56bc932aac7ad1001900001c, 56c02883ef6e394741000017,
    52d2b75403868f1b06000035, 52f118aa2059c6d71c000003, 52e929eb98d0239505000023,
    532c12f2d6d3ac6a3400001d, 52d8466298d0239505000006'
[3] The 48 questions resulting from merging with their pair have the
    following ids: 5149aafcd24251bc05000045, 515db020298dcd4e51000011,
    515db54c298dcd4e51000016, 51680a49298dcd4e51000062, 52b06a68f828ad283c000005,
    52bf1aa503868f1b06000006, 52bf1af803868f1b06000008, 52bf1d6003868f1b0600000e,
    52cb9b9b03868f1b0600002d, 52d2818403868f1b06000033, 52df887498d023950500000c,
    52e0c9a298d0239505000010, 52e203bc98d0239505000011, 52e62bae98d0239505000015,
    52e6c92598d0239505000019, 52e7bbf698d023950500001d, 52ea605098d0239505000028,
    52ece29f98d023950500002c, 52ecf2dd98d023950500002e, 52ef7754c8da898910000014,
    52f112bb2059c6d71c000002, 52f65f372059c6d71c000027, 52f77f752059c6d71c00002b,
    52f77f892059c6d71c00002c, 52f89ee42059c6d71c00004d, 52f89f4f2059c6d71c00004e,
    52f89fba2059c6d71c00004f, 52f89fc62059c6d71c000050, 52f89fd32059c6d71c000051,
    52fa6ac72059c6d71c000055, 52fa73c62059c6d71c000058, 52fa73e82059c6d71c000059,
    52fa74252059c6d71c00005b, 52fc8b772059c6d71c00006e, 52fc94572059c6d71c000070,
    52fc94ae2059c6d71c000073, 52fc94db2059c6d71c000074, 52fe52702059c6d71c000078,
    52fe58f82059c6d71c00007a, 530cefaaad0bf1360c000008, 530cefaaad0bf1360c000010,
    533ba218fd9a95ea0d000007, 534bb147aeec6fbd07000014, 55167dec46478f2f2c00000a,
    56c04412ef6e39474100001b, 56c1f01eef6e394741000046, 56c81fd15795f9a73e00000c,
    587d016ed673c3eb14000002
"""

_BIOASQ_5B_DESCRIPTION = """\
The data are intended to be used as training and development data for BioASQ 5,
which will take place during 2017. There is one file containing the data:
 - BioASQ-trainingDataset5b.json

The file contains the data of the first four editions of the challenge: 1799
questions with their relevant documents, snippets, concepts and rdf triples,
exact and ideal answers.
"""

_BIOASQ_4B_DESCRIPTION = """\
The data are intended to be used as training and development data for BioASQ 4,
which will take place during 2016. There is one file containing the data:
 - BioASQ-trainingDataset4b.json

The file contains the data of the first three editions of the challenge: 1307
questions with their relevant documents, snippets, concepts and rdf triples,
exact and ideal answers from the first two editions and 497 questions with
similar annotations from the third editions of the challenge.
"""

_BIOASQ_3B_DESCRIPTION = """No README provided."""

_BIOASQ_2B_DESCRIPTION = """No README provided."""

_BIOASQ_BLURB_DESCRIPTION = """The BioASQ corpus contains multiple question 
answering tasks annotated by biomedical experts, including yes/no, factoid, list, 
and summary questions. Pertaining to our objective of comparing neural language 
models, we focus on the the yes/no questions (Task 7b), and leave the inclusion 
of other tasks to future work. Each question is paired with a reference text 
containing multiple sentences from a PubMed abstract and a yes/no answer. We use 
the official train/dev/test split of 670/75/140 questions.

See 'Domain-Specific Language Model Pretraining for Biomedical 
Natural Language Processing' """

_DESCRIPTION = {
    "bioasq_10b": _BIOASQ_10B_DESCRIPTION,
    "bioasq_9b": _BIOASQ_9B_DESCRIPTION,
    "bioasq_8b": _BIOASQ_8B_DESCRIPTION,
    "bioasq_7b": _BIOASQ_7B_DESCRIPTION,
    "bioasq_6b": _BIOASQ_6B_DESCRIPTION,
    "bioasq_5b": _BIOASQ_5B_DESCRIPTION,
    "bioasq_4b": _BIOASQ_4B_DESCRIPTION,
    "bioasq_3b": _BIOASQ_3B_DESCRIPTION,
    "bioasq_2b": _BIOASQ_2B_DESCRIPTION,
    "bioasq_blurb": _BIOASQ_BLURB_DESCRIPTION,
}

_HOMEPAGE = "http://participants-area.bioasq.org/datasets/"

# Data access reqires registering with BioASQ.
# See http://participants-area.bioasq.org/accounts/register/
_LICENSE = Licenses.NLM_LICENSE

_URLs = {
    "bioasq_10b": ["BioASQ-training10b.zip", None],
    "bioasq_9b": ["BioASQ-training9b.zip", "Task9BGoldenEnriched.zip"],
    "bioasq_8b": ["BioASQ-training8b.zip", "Task8BGoldenEnriched.zip"],
    "bioasq_7b": ["BioASQ-training7b.zip", "Task7BGoldenEnriched.zip"],
    "bioasq_6b": ["BioASQ-training6b.zip", "Task6BGoldenEnriched.zip"],
    "bioasq_5b": ["BioASQ-training5b.zip", "Task5BGoldenEnriched.zip"],
    "bioasq_4b": ["BioASQ-training4b.zip", "Task4BGoldenEnriched.zip"],
    "bioasq_3b": ["BioASQ-trainingDataset3b.zip", "Task3BGoldenEnriched.zip"],
    "bioasq_2b": ["BioASQ-trainingDataset2b.zip", "Task2BGoldenEnriched.zip"],
    "bioasq_blurb": ["BioASQ-training7b.zip", "Task7BGoldenEnriched.zip"],
}

# BLURB train and dev contain all yesno questions from the offical training split
# test is all yesno question from the official test split
_BLURB_SPLITS = {
    "dev": {
        "5313b049e3eabad021000013",
        "553a8d78f321868558000003",
        "5158a5b8d24251bc05000097",
        "571e3d42bb137a4b0c000007",
        "5175b97a8ed59a060a00002f",
        "56c9e9d15795f9a73e00001d",
        "56d19ffaab2fed4a47000001",
        "518ccac0310faafe0800000b",
        "56f12ca92ac5ed145900000e",
        "51680a49298dcd4e51000062",
        "5339ed7bd6d3ac6a34000060",
        "516e5f33298dcd4e5100007e",
        "5327139ad6d3ac6a3400000d",
        "54e12ae3ae9738404b000004",
        "5321b8579b2d7acc7e000008",
        "514a4679d24251bc0500005b",
        "54c12fd1f693c3b16b000001",
        "52df887498d023950500000c",
        "52f20d802059c6d71c00000a",
        "532f0c4ed6d3ac6a3400002e",
        "52b2f3b74003448f5500000c",
        "52b2f1724003448f5500000b",
        "515d9a42298dcd4e5100000d",
        "5159b990d24251bc050000a3",
        "54e12c30ae9738404b000005",
        "553a6a9fbc4f83e82800001c",
        "5509ec41c2af5d5b70000006",
        "56cae40b5795f9a73e000022",
        "51680b0e298dcd4e51000065",
        "515df89e298dcd4e5100002f",
        "54f49e56d0d681a040000004",
        "571e3e2abb137a4b0c000008",
        "515debe7298dcd4e51000026",
        "56f6ab7009dd18d46b00000d",
        "53302bced6d3ac6a34000039",
        "5322de919b2d7acc7e000012",
        "5709f212cf1c325851000020",
        "5502abd1e9bde69634000008",
        "516c220e298dcd4e51000071",
        "5894597e7d9090f353000004",
        "5895ec5e7d9090f353000015",
        "58bbb8ae22d3005309000018",
        "58bc58c302b8c60953000001",
        "58c276bc02b8c60953000020",
        "58c0825502b8c6095300001b",
        "58ab1f6c9ef3c34033000002",
        "58adbe999ef3c34033000005",
        "58df3e408acda3452900002d",
        "58dfec676fddd3e83e000006",
        "58d8d0cc8acda34529000008",
        "58b67fae22d3005309000009",
        "58dbbbf08acda3452900001d",
        "58dbba438acda3452900001c",
        "58dbbdac8acda3452900001e",
        "58dcbb8c8acda34529000021",
        "5a468785966455904c00000d",
        "5a70de5199e2c3af26000005",
        "5a67a550b750ff4455000009",
        "5a679875b750ff4455000004",
        "5a7a44b4faa1ab7d2e000010",
        "5a67ade5b750ff445500000c",
        "5a8881118cb19eca6b000006",
        "5a67b48cb750ff4455000010",
        "5a679be1b750ff4455000005",
        "5a7340962dc08e987e000017",
        "5a737e233b9d13c70800000d",
        "5a8dc57ffcd1d6a10c000025",
        "5a6d186db750ff4455000031",
        "5a70d43b99e2c3af26000003",
        "5a70ec6899e2c3af2600000c",
        "5a9ac4161d1251d03b000010",
        "5a733d2a2dc08e987e000015",
        "5a74acd80384be9551000006",
        "5aa6800ad6d6b54f79000011",
        "5a9d9ab94e03427e73000003",
    }
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BioasqTaskBDataset(datasets.GeneratorBasedBuilder):
    """
    BioASQ Task B On Biomedical Semantic QA.
    Creates configs for BioASQ2 through BioASQ10.
    """

    DEFAULT_CONFIG_NAME = "bioasq_9b_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # BioASQ2 through BioASQ10
    BUILDER_CONFIGS = []
    for version in range(2, 11):
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"bioasq_{version}b_source",
                version=SOURCE_VERSION,
                description=f"bioasq{version} Task B source schema",
                schema="source",
                subset_id=f"bioasq_{version}b",
            )
        )

        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"bioasq_{version}b_bigbio_qa",
                version=BIGBIO_VERSION,
                description=f"bioasq{version} Task B in simplified BigBio schema",
                schema="bigbio_qa",
                subset_id=f"bioasq_{version}b",
            )
        )

    # BLURB Benchmark config https://microsoft.github.io/BLURB/
    BUILDER_CONFIGS.append(
        BigBioConfig(
            name=f"bioasq_blurb_bigbio_qa",
            version=BIGBIO_VERSION,
            description=f"BLURB benchmark in simplified BigBio schema",
            schema="bigbio_qa",
            subset_id=f"bioasq_blurb",
        )
    )

    def _info(self):

        # BioASQ Task B source schema
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "body": datasets.Value("string"),
                    "documents": datasets.Sequence(datasets.Value("string")),
                    "concepts": datasets.Sequence(datasets.Value("string")),
                    "ideal_answer": datasets.Sequence(datasets.Value("string")),
                    "exact_answer": datasets.Sequence(datasets.Value("string")),
                    "triples": [
                        {
                            "p": datasets.Value("string"),
                            "s": datasets.Value("string"),
                            "o": datasets.Value("string"),
                        }
                    ],
                    "snippets": [
                        {
                            "offsetInBeginSection": datasets.Value("int32"),
                            "offsetInEndSection": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "beginSection": datasets.Value("string"),
                            "endSection": datasets.Value("string"),
                            "document": datasets.Value("string"),
                        }
                    ],
                }
            )
        # simplified schema for QA tasks
        elif self.config.schema == "bigbio_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION[self.config.subset_id],
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _dump_gold_json(self, data_dir):
        """
        BioASQ test data is split into multiple records {9B1_golden.json,...,9B5_golden.json}
        We combine these files into a single test set file 9Bx_golden.json
        """
        # BLURB is based on version 7
        version = (
            re.search(r"bioasq_([0-9]+)b", self.config.subset_id).group(1)
            if "blurb" not in self.config.name
            else "7"
        )
        gold_fpath = os.path.join(
            data_dir, f"Task{version}BGoldenEnriched/bx_golden.json"
        )

        if not os.path.exists(gold_fpath):
            # combine all gold json files
            filelist = glob.glob(os.path.join(data_dir, "*/*.json"))
            data = {"questions": []}
            for fname in sorted(filelist):
                with open(fname, "rt", encoding="utf-8") as file:
                    data["questions"].extend(json.load(file)["questions"])
            # dump gold to json
            with open(gold_fpath, "wt", encoding="utf-8") as file:
                json.dump(data, file, indent=2)

        return f"Task{version}BGoldenEnriched/bx_golden.json"

    def _blurb_split_generator(self, train_dir, test_dir):
        """
        Create splits for BLURB Benchmark
        """
        gold_fpath = self._dump_gold_json(test_dir)

        # create train/dev splits from yesno questions
        train_fpath = os.path.join(train_dir, "blurb_bioasq_train.json")
        dev_fpath = os.path.join(train_dir, "blurb_bioasq_dev.json")

        if not os.path.exists(train_fpath):
            data_fpath = os.path.join(train_dir, "BioASQ-training7b/trainining7b.json")
            with open(data_fpath, "rt", encoding="utf-8") as file:
                data = json.load(file)

            blurb_splits = {
                "train": {"questions": []},
                "dev": {"questions": []},
                "test": {"questions": []},
            }
            for record in data["questions"]:
                if record["type"] != "yesno":
                    continue
                if record["id"] in _BLURB_SPLITS["dev"]:
                    blurb_splits["dev"]["questions"].append(record)
                else:
                    blurb_splits["train"]["questions"].append(record)

            with open(train_fpath, "wt", encoding="utf-8") as file:
                json.dump(blurb_splits["train"], file, indent=2)

            with open(dev_fpath, "wt", encoding="utf-8") as file:
                json.dump(blurb_splits["dev"], file, indent=2)

        # create test split from yesno questions
        with open(os.path.join(test_dir, gold_fpath), "rt", encoding="utf-8") as file:
            data = json.load(file)

        for record in data["questions"]:
            if record["type"] != "yesno":
                continue
            blurb_splits["test"]["questions"].append(record)

        test_fpath = os.path.join(test_dir, "blurb_bioasq_test.json")
        with open(test_fpath, "wt", encoding="utf-8") as file:
            json.dump(blurb_splits["test"], file, indent=2)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_fpath,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev_fpath,
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_fpath,
                    "split": "test",
                },
            ),
        ]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )

        train_dir, test_dir = dl_manager.download_and_extract(
            [
                os.path.join(self.config.data_dir, _url)
                for _url in _URLs[self.config.subset_id]
            ]
        )
        # create gold dump and get path
        gold_fpath = self._dump_gold_json(test_dir)

        # older versions of bioasq have different folder formats
        train_fpaths = {
            "bioasq_2b": "BioASQ_2013_TaskB/BioASQ-trainingDataset2b.json",
            "bioasq_3b": "BioASQ-trainingDataset3b.json",
            "bioasq_4b": "BioASQ-training4b/BioASQ-trainingDataset4b.json",
            "bioasq_5b": "BioASQ-training5b/BioASQ-trainingDataset5b.json",
            "bioasq_6b": "BioASQ-training6b/BioASQ-trainingDataset6b.json",
            "bioasq_7b": "BioASQ-training7b/trainining7b.json",
            "bioasq_8b": "training8b.json",  # HACK - this zipfile strips the dirname
            "bioasq_9b": "BioASQ-training9b/training9b.json",
            "bioasq_10b": "BioASQ-training10b/training10b.json",
        }

        # BLURB has custom train/dev/test splits based on Task 7B
        if "blurb" in self.config.name:
            return self._blurb_split_generator(train_dir, test_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        train_dir, train_fpaths[self.config.subset_id]
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(test_dir, gold_fpath),
                    "split": "test",
                },
            ),
        ]

    def _get_exact_answer(self, record):
        """The value exact_answer can be in different formats based on question type."""
        if record["type"] == "yesno":
            exact_answer = [record["exact_answer"]]
        elif record["type"] == "summary":
            exact_answer = []
            # summary question types only have an ideal answer, so use that for bigbio
            if self.config.schema == "bigbio_qa":
                exact_answer = (
                    record["ideal_answer"]
                    if isinstance(record["ideal_answer"], list)
                    else [record["ideal_answer"]]
                )

        elif record["type"] == "list":
            exact_answer = record["exact_answer"]
        elif record["type"] == "factoid":
            # older version of bioasq sometimes represent this as as string
            exact_answer = (
                record["exact_answer"]
                if isinstance(record["exact_answer"], list)
                else [record["exact_answer"]]
            )
        return exact_answer

    def _generate_examples(self, filepath, split):
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as file:
                data = json.load(file)
                for i, record in enumerate(data["questions"]):
                    yield i, {
                        "id": record["id"],
                        "type": record["type"],
                        "body": record["body"],
                        "documents": record["documents"],
                        "concepts": record["concepts"] if "concepts" in record else [],
                        "triples": record["triples"] if "triples" in record else [],
                        "ideal_answer": record["ideal_answer"]
                        if isinstance(record["ideal_answer"], list)
                        else [record["ideal_answer"]],
                        "exact_answer": self._get_exact_answer(record),
                        "snippets": record["snippets"] if "snippets" in record else [],
                    }

        elif self.config.schema == "bigbio_qa":
            # NOTE: Years 2014-2016 (BioASQ2-BioASQ4) have duplicate records
            cache = set()
            with open(filepath, encoding="utf-8") as file:
                uid = 0
                data = json.load(file)
                for record in data["questions"]:
                    # for questions that do not have snippets, skip
                    if "snippets" not in record:
                        continue
                    for i, snippet in enumerate(record["snippets"]):
                        key = f'{record["id"]}_{i}'
                        # ignore duplicate records
                        if key not in cache:
                            cache.add(key)
                            yield uid, {
                                "id": key,
                                "document_id": snippet["document"],
                                "question_id": record["id"],
                                "question": record["body"],
                                "type": record["type"],
                                "choices": [],
                                "context": snippet["text"],
                                "answer": self._get_exact_answer(record),
                            }
                            uid += 1
