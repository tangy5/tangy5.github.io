import argparse
import os.path
import pathlib
import socketserver
from http.server import SimpleHTTPRequestHandler
from urllib.request import urlopen

import requests

session = requests.Session()


class MyProxy(SimpleHTTPRequestHandler):
    def do_GET(self):
        url = self.path[1:]
        # print(f"GET: {self.path} => {url}")

        if not url:
            url = "index.html"
        if url in ["index.html", "vista3dNIM.js", "vista3dNIM.js.map", "favicon.ico"]:
            self.send_response(200)
            self.end_headers()

            p = pathlib.Path(os.path.abspath(f'./{url}'))
            if url in ["vista3dNIM.js", "vista3dNIM.js.map"]:
                with open(url, "rb") as f:
                    c = f.read().replace(b"const NIM_PROXY_URL = '';", b"const NIM_PROXY_URL = '/';")
                    self.wfile.write(c)
                return
            self.copyfile(urlopen(p.as_uri()), self.wfile)
            return

        headers = self._proxy_headers()
        response = session.get(url, headers=headers)

        self.send_response(response.status_code)
        for k, v in response.headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(response.content)

    def do_POST(self):
        url = self.path[1:]
        # print(f"POST: {self.path} => {url}")

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        headers = self._proxy_headers()
        response = session.post(url, headers=headers, data=post_data, verify=False)

        self.send_response(response.status_code)
        for k, v in response.headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(response.content)

    def _proxy_headers(self):
        filters = {"host", "origin", "referer", "cookie", "user-agent"}
        return {
            k: v for k, v in self.headers.items() if k.lower() not in filters and not k.lower().startswith("sec-")
        }


def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--host", default="", type=str, help="Local/Server Address (default: localhost)")
    parser.add_argument("-p", "--port", default=9097, type=int, help="Local/Server Port (default: 9097)")

    args = parser.parse_args()

    httpd = socketserver.ThreadingTCPServer((args.host, args.port), MyProxy)
    print(f"Now serving at: http://{args.host if args.host else 'localhost' }:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
