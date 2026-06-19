fname = "dru_log_city2bdd.txt"
content = None
with open(fname, 'r') as f:
    content = f.readlines()
content = [line for line in content if 'mAP' in line]
content = [float(line.split(' ')[-1]) for line in content]
stu_result = content[0::2]
tch_result = content[1::2]
print(stu_result)
print(tch_result)