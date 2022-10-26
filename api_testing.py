import requests

def query(Question):

    reqUrl = f'''http://127.0.0.1:8000/predict?question={Question}'''
    payload = f'''{ "inputs" : {"question" : {Question} }}'''
    response = requests.request("POST", reqUrl, data=payload,  )

    return (response.text)

print(query("Domain ?"))