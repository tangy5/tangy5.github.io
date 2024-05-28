How to run VISTA NIM Client?
============================

GitHub Hosted
--------------
Access https://sachidanandalle.github.io/ GitHub hosted page and run VISTA 3D NIM over accessible images.

![image](https://github.com/SachidanandAlle/sachidanandalle.github.io/assets/7339051/69243a7c-3609-49d0-9944-da1bd1eece1c)

> Make sure image uri is also accessible to your NIM Endpoint


If your NIM Endpoint doesn't support CORS then you might want to run browser with disabled security mode.
```bash
edge --args --disable-web-security --allow-running-insecure-content -user-data-dir=my_browser_edge
```
> Make sure you have closed every instance of the brwoser before running it in disabled/unsecured mode.

Proxy Mode
----------
In case of CORS issue etc... this shall be a simple approach to run a local proxy and run the NIM
```bash
pip install -r requirements.txt
python client.py
```

By default, it runs on port `9097` and so open http://localhost:9097 in your browser to play around with VISTA-3D NIM.

> This will proxy all the requests to NIM via local server. Hence, it is recommended for development/debugging purpose only.


