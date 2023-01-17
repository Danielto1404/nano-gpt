import typing as tp

import bs4
import requests


class GeniusScraper:
    """https://genius.com/ scrapper, which allow to download artist songs texts."""

    def base_link(self):
        pass

    def download(self, artists: tp.List[str]):
        pass


if __name__ == '__main__':
    html = requests.get("https://genius.com/Morgenshtern-olala-lyrics").text
    index =html.find("<p>[Текст песни")
    index_b = html.find("lyricsPlaceholderReason")
    print(html[index:index_b].split("<br>"))

