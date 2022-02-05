"""


general_v0 = Features(
    {
        "passages": Sequence(
            {
                "document_id": Value("string"),
                "type": Value("string"),
                "text": Value("string"),

                "snippets": Sequence(
                    {
                        "snippet_id": Value("string"),
                        "offsets": Sequence([Value("int32")]),
                        "text": Value("string"),
                        "type": Value("string"),

                    }
                ),
                "entities": Sequence(
                    {
                        "entity_id": Value("int32"),
                        "offsets": Sequence([Value("int32")]),
                        "text": Value("string"),
                        "type": Value("string"),
                        "entity_kb_id": Value("string"),
                    }
                ),
                "relations": Sequence(
                    {
                        'relation_id': Value("string"),
                        'type': Value("string"),
                        'arg1_id': Value("int32"),
                        'arg2_id': Value("int32"),
                        "relation_kb_id": Value("int32"),
                    }
                )
            }
        )
    }
)


"""

import re
from datasets import load_dataset


ds = load_dataset(
    'n2c2_2011_coref.py',
    name="original",
    data_dir="/home/galtay/data/big_science_biomedical/n2c2_2011_coref",
)


dso = load_dataset(
    'n2c2_2011_coref.py',
    name="coref",
    data_dir="/home/galtay/data/big_science_biomedical/n2c2_2011_coref",
)



text = '# Candidemia :  The patient was noted to have a positive blood culture from 03-12 .'


def tokoff_from_line(text):
    tokoff = []
    start = None
    end = None
    for ii, char in enumerate(text):
        if char != " " and start is None:
            start = ii
        if char == " " and start is not None:
            end = ii
            tokoff.append((start, end))
            start = None
            end = None
    if start is not None:
        end = ii + 1
        tokoff.append((start, end))
    return tokoff


tokoff = tokoff_from_line(text)




C_PATTERN = r"c=\"(.+?)\" (\d+):(\d+) (\d+):(\d+)"
T_PATTERN = r"t=\"(.+?)\""


def parse_con_line(line):
    c_part, t_part = line.split("||")
    c_match = re.match(C_PATTERN, c_part)
    t_match = re.match(T_PATTERN, t_part)

    return {
        "text": c_match.group(1),
        "line_start": int(c_match.group(2)),
        "offset_start": int(c_match.group(3)),
        "line_end": int(c_match.group(4)),
        "offset_end": int(c_match.group(5)),
        "type": t_match.group(1),
    }


def parse_chains_line(line):
    pieces = line.split("||")
    c_parts = pieces[:-1]
    t_part = pieces[-1]
    c_matches = [re.match(C_PATTERN, c_part) for c_part in c_parts]
    t_match = re.match(T_PATTERN, t_part)
    return c_parts, t_part



num_matched = 0
num_missed = 0

for ii_sample, sample in enumerate(ds['train']):

    con_lines = sample['con'].splitlines()
    chains_lines = sample['chains'].splitlines()
    text = sample['txt']

    text_lines = text.splitlines()
    text_line_lengths = [len(el) for el in text_lines]

    con_parsed = sorted([
        parse_con_line(line)
        for line in con_lines
    ], key=lambda x: (x['line_start'], x['offset_start']))

#    c_parts, t_part = parse_chains_line(chains_lines[0])


    for ii_cp, cp in enumerate(con_parsed):

        # line indexes are 1 based
        for ii_line in range(cp['line_start'], cp['line_end'] + 1):

            ii = cp['offset_start']
            ff = cp['offset_end'] + 1
            line_start_off = sum(text_line_lengths[:ii_line-1]) + (ii_line-1)
            tokoff = tokoff_from_line(text_lines[ii_line-1])

            if ii_line == cp['line_start'] == cp['line_end']:
                start_off = line_start_off + tokoff[ii][0]
                end_off = line_start_off + tokoff[ff-1][1]

            elif (ii_line == cp['line_start']) and (ii_line != cp['line_end']):
                start_off = line_start_off + tokoff[ii][0]
                end_off = line_start_off + text_line_lengths[ii_line-1]

            elif (ii_line != cp['line_start']) and (ii_line == cp['line_end']):
                end_off = end_off + tokoff[ff-1][1]

            else:
                end_off += text_line_lengths[ii_line-1]

        con_text = text[start_off:end_off].lower().strip()
        match = con_text == cp["text"]

        if match:
            num_matched += 1
        else:
            num_missed += 1
            print('ii_sample: ', ii_sample, 'ii_cp: ', ii_cp)
            print('cp: ', cp)
            print('con_text: ', con_text)
            print()


        if ii_sample == 144 and ii_cp == 286:
            sys.exit(1)


print('num_matched: ', num_matched, 'num_missed: ', num_missed)
