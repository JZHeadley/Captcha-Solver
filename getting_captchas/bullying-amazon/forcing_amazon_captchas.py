import requests
from bs4 import BeautifulSoup
import mechanicalsoup

if __name__ == '__main__':
    sign_in_link="https://www.amazon.com/ap/signin?openid.return_to=https%3A%2F%2Fwww.amazon.com%2F%3Fref_%3Dnav_ya_signin&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=usflex&openid.mode=checkid_setup&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&&openid.pape.max_auth_age=0"
    sign_in_page=requests.get(sign_in_link)
    sign_in_page_soup = BeautifulSoup(sign_in_page.content, 'html.parser')
    # print(sign_in_page_soup)
    browser=mechanicalsoup.StatefulBrowser()
    browser.set_user_agent("Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36")
    browser.open(sign_in_link)
    browser.select_form('form[name="signIn"]')
    browser["email"]="zephyrinkg@gmail.com"
    browser["password"]="blah"
    browser.launch_browser()
    # browser.submit_selected()

    # print(browser.get_current_page())
