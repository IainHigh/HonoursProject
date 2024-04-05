# pylint: disable=missing-module-docstring

import sys
import argparse
from argparse import Namespace
import logging
from rich.traceback import install
from rich.console import Console
from src.api import PexelsAPI
from src.banner import cli_banner
from src.args import PayloadCheck
import pexel_key

console = Console()
install()

search = "Mirror"


def main():
    """Get Pexels images"""
    pexels_api = PexelsAPI(pexel_key.pexel_key)
    pexels_api.search_for_photos({"query": search})


if __name__ == "__main__":
    cli_banner("Pexels  Scraper")

    try:
        main()
    except KeyboardInterrupt:
        console.print("\r\n[red]Execution cancelled by user[/red]")
        sys.exit()
