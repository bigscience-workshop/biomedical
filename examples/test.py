import re
from datasets import load_dataset


ds = load_dataset(
    'n2c2_2011_coref.py',
    name="original",
    data_dir="/home/galtay/data/big_science_biomedical/n2c2_2011_coref",
)



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




for sample in ds['train']:

    con_lines = sample['con'].splitlines()
    chains_lines = sample['chains'].splitlines()
    text = sample['txt']

    text_lines = text.splitlines()
    text_lines_tokens = [line.split() for line in text_lines]

    con_parsed = sorted([
        parse_con_line(line)
        for line in con_lines
    ], key=lambda x: (x['line_start'], x['offset_start']))

#    c_parts, t_part = parse_chains_line(chains_lines[0])

    for cp in con_parsed:

        # line indexes are 1 based
        tokens = []
        for ii_line in range(cp['line_start'], cp['line_end'] + 1):

            if ii_line == cp['line_start'] == cp['line_end']:
                ii = cp['offset_start']
                ff = cp['offset_end'] + 1
                tokens += text_lines_tokens[ii_line-1][ii:ff]

            elif (ii_line == cp['line_start']) and (ii_line != cp['line_end']):
                ii = cp['offset_start']
                tokens += text_lines_tokens[ii_line-1][ii:]

            elif (ii_line != cp['line_start']) and (ii_line == cp['line_end']):
                ff = cp['offset_end'] + 1
                tokens += text_lines_tokens[ii_line-1][:ff]

            else:
                tokens += text_lines_tokens[ii_line-1]


        tokens_lower = [el.lower() for el in tokens]


        match = tuple(tokens_lower) == tuple(cp['text'].split())
        if not match:
            print(ii_line)
            print('text tokens:    ', tokens_lower)
            print('concept tokens: ', cp['text'].split())
            print(cp)
            print(text_lines_tokens[ii_line-1])
            print()
