import hashlib
import os
"""
In solving these I discovered that there are duplicate captchas given to me
This script should find all the duplicates so I can keep track of them
and solve them later
"""
captcha_dir = "../../data/captchas/"

hashes = []
for file in os.listdir(captcha_dir):
    hashes.append(
        (file, hashlib.md5(open(captcha_dir+file, 'rb').read()).hexdigest())
    )

# print(hashes)

seen = {}
dupes = []
dupe_hashes = []
for x in hashes:
    if x[1] not in seen:
        seen[x[1]] = [x[0]]
    else:
        if seen[x[1]] is not None:
            seen[x[1]].append(x[0])
            dupes.append(x[0])
dupes.sort()

# print(seen)
for key, value in seen.items():
    if value.__len__() > 1:
        value.sort()
        print(key,":",value)
