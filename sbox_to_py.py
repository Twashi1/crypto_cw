with open("sbox.txt", "r") as f:
    lines = f.readlines()

d = {}

for line in lines:
    l, r = line.split(":")
    d[int(l)] = int(r)

with open("sbox.py", "w+") as f:
    f.write(str(list(d.values())))

print(d)
