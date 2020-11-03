import sys

def open_file(f):
    f = open(f, 'r').readlines()
    return f

dataset = sys.argv[1]

file1 = dataset + '.target_eval'
file2 = dataset + '.target2_eval'
file3 = dataset + '.target3_eval'

meteor_file = dataset + '.target_eval_meteor'
meteor_file = open(meteor_file, 'w')

f1 = open_file(file1)
f2 = open_file(file2)
f3 = open_file(file3)

for x1, x2, x3 in zip(f1, f2, f3):
    meteor_file.write(x1.strip() + '\n')
    meteor_file.write(x2.strip() + '\n')
    meteor_file.write(x3.strip() + '\n')

