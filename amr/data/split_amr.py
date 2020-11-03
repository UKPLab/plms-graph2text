import sys
import os


INPUT = sys.argv[1]
OUT_SURF = sys.argv[2]
OUT_GRAPH = sys.argv[3]

with open(INPUT) as f:
    lines = f.readlines()

with open(OUT_SURF, 'w') as surf, open(OUT_GRAPH, 'w') as graph:
    amr_mode = False
    amr_tokens = []
    for line in lines:
        if line.startswith('#'):
            if amr_mode:
                amr_mode = False
                amr = ' '.join(amr_tokens)
                graph.write(amr + '\n')
                amr_tokens = []
            tokens = line.split()
            if tokens[1] == "::snt":
                sent = ' '.join(tokens[2:])
                surf.write(sent + '\n')
        elif line.strip() == '':
            continue
        else:
            amr_mode = True
            amr_tokens.append(line.strip())
    if amr_mode:
        amr_mode = False
        amr = ' '.join(amr_tokens)
        graph.write(amr + '\n')
        amr_tokens = []
