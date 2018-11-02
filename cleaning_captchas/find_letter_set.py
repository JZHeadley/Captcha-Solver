import os
CAPTCHA_DIR = "../../captchas_solved/"


"""
The captchas use only a certain set of letters and numbers.
This script should find that set 
"""
uniqueLetters = set()
for file in os.listdir(CAPTCHA_DIR):
    file = file[:-4]
    if file.__len__() == 6:
        for i in range(0, 5):
            uniqueLetters.add(file[i])


uniqueLetters=list(uniqueLetters)
uniqueLetters.sort()
print(uniqueLetters)
print(uniqueLetters.__len__())
