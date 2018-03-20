import requests
import webbrowser
#data = {"usename":'陈堂明','password':'19970923ctm...'}
r = requests.post("http://bbs.uestc.edu.cn/member.php")
print(r.url)
webbrowser.open(r.url)
