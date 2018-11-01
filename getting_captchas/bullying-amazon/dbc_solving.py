import deathbycaptcha
import configparser
import os
CAPTCHA_DIR = "captchas_gathered/"
OUTPUT_DIR = "../../captchas_solved/"
INCORRECT_DIR = "incorrect_solved/"
config = configparser.ConfigParser()
config.read('config.ini')
credentials = config['credentials']
username = credentials['username']
password = credentials['password']

client = deathbycaptcha.SocketClient(username, password)

captchas = os.listdir(CAPTCHA_DIR)
balance = client.get_balance()
print("We can solve", balance, "more captchas")
incorrect = []
while captchas.__len__() > 0 and balance > 0:
    print(balance)
    to_solve = captchas.pop()
    try:
        captcha = client.decode(CAPTCHA_DIR+to_solve)
        solution = captcha['text']
        print(solution)
        if 'q' in solution or '1' in solution:
            incorrect.append(to_solve)
            print("Reporting incorrect solution")
            print(to_solve, "was incorrect")
            client.report(captcha['captcha'])
            os.rename(CAPTCHA_DIR+to_solve, INCORRECT_DIR+to_solve)
        else:
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
        print("Access error occured")
    balance = client.get_balance()
print(incorrect)
