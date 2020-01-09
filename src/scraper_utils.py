import json
from urllib.request import urlopen
from urllib.parse import urlencode


# UTILITIES
def get_page(url):
    print(f"GET {url}")
    client = urlopen(url)
    page = client.read()
    client.close()

    return page


def urlencode_string(s):
    return urlencode({'s': s})[2:]