import deathbycaptcha
import configparser
import os

CAPTCHA_DIR = "../data/captchas/"
OUTPUT_DIR = "../data/captchas_solved/"
INCORRECT_DIR = "../data/incorrect_solved/"
config = configparser.ConfigParser()
config.read('config.ini')
credentials = config['credentials']
username = credentials['username']
password = credentials['password']

client = deathbycaptcha.SocketClient(username, password)

captchas = os.listdir(CAPTCHA_DIR)
balance = client.get_balance()
print("We can solve", balance, "more captchas")
bad_chars = ['1', 'l', 'u', 'v', 's', 't', 'i',
             'j', 'o', 'r', 'q', 'z', 's', '9', 'h', 'k']
incorrect = []
while captchas.__len__() > 0 and balance > 0:
    flag = True
    print(balance)
    to_solve = captchas.pop()
    try:
        captcha = client.decode(CAPTCHA_DIR+to_solve)
        solution = captcha['text']
        print(solution)
        for bad_char in bad_chars:
            if bad_char in solution:
                incorrect.append(to_solve)
                print("Reporting incorrect solution")
                print(to_solve, "was incorrect")
                client.report(captcha['captcha'])
                os.rename(CAPTCHA_DIR+to_solve, INCORRECT_DIR+to_solve)
                flag = False
        if flag:
            output_file = OUTPUT_DIR+solution
            while os.path.isfile(output_file+".jpg"):
                """
                If the file already exists append '_duplicate' to it so we know its a dupe
                In a while loop just in case we have more than 2 of any one image
                Hopefully this ensures moving the file never causes any errors
                """
                print("We have a duplicate!")
                output_file = output_file+"_duplicate"
            os.rename(CAPTCHA_DIR+to_solve, output_file + ".jpg")
    except deathbycaptcha.AccessDeniedException:
        exit(1)
        print("Access error occured")
    balance = client.get_balance()
print(incorrect)
