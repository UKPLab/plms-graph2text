import sys
from sacrebleu import corpus_bleu


def open_file(f):
    return [l.strip() for l in open(f, 'r').readlines()]

file_refs = 'data/amr17/test.target'
refs = open_file(file_refs)
generated_file = sys.argv[1]
sys = open_file(generated_file)

assert len(refs) == len(sys)

print(len(refs))

# refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
#         ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
# sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
bleu = corpus_bleu(sys, [refs])
print(bleu.score)