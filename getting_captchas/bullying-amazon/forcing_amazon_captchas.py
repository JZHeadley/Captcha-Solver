import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ActionChains
import urllib
import time
numCaptchas = 1000

if __name__ == '__main__':
    # sign_in_link="https://www.amazon.com/ap/signin?openid.return_to=https%3A%2F%2Fwww.amazon.com%2F%3Fref_%3Dnav_ya_signin&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=usflex&openid.mode=checkid_setup&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&&openid.pape.max_auth_age=0"
    options=Options()
    # options.add_argument("--headless")
    driver = webdriver.Chrome(chrome_options=options)
    driver.get("https://www.amazon.com")
    hover_elem = driver.find_element_by_css_selector("#nav-link-accountList")
    # hover over the sign in nav entry to get the dropdown
    ActionChains(driver).move_to_element(hover_elem).perform()
    # click the link in that dropdown
    driver.find_element_by_css_selector(".nav-action-button").click()
    email = driver.find_element_by_name("email")
    password = driver.find_element_by_name("password")
    email.send_keys("zephyrinkg@gmail.com")
    password.send_keys("blah")
    driver.find_element_by_css_selector("#signInSubmit").click()

    # just generate a timestamp so our files are unique between runs
    timestr = time.strftime("%Y%m%d-%H-%M%S")
    for i in range(0, numCaptchas):
        captcha = driver.find_element_by_id("auth-captcha-image")
        captcha_src = captcha.get_attribute("src")
        urllib.request.urlretrieve(
            captcha_src, "../../data/captchas/captcha-%s-%i.jpg" % (timestr, i))
        driver.find_element_by_id("auth-captcha-refresh-link").click()
        time.sleep(1)
        if i % 10 == 0:
            print(i)
    driver.quit()
