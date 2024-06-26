# pylint: disable=missing-module-docstring

import logging
import requests
from requests import Response
from src.image import Image
from src.exception import UnexpectedResponse


# pylint: disable=too-few-public-methods
class PexelsAPI:
    """Pexels API base functionality."""

    base = "https://api.pexels.com/videos/search"

    default_timeout = 30

    def __init__(self, api_key: str):
        self.authorization_header = {"Authorization": api_key}

    def _send_request(self, url: str, parameters: dict) -> Response:
        response = requests.get(
            url,
            headers=self.authorization_header,
            params=parameters,
            timeout=self.default_timeout,
        )
        if response.status_code != 200:
            response_message = f"Unexpected status code `{response.status_code}`. Message: `{(response.json())}."
            logging.error(response_message)
            raise UnexpectedResponse(response_message)

        return response

    # noinspection PyMethodMayBeStatic
    def _iterate_response_photos(self, photo_collection: list, count: int) -> int:
        for photo in photo_collection:
            original_image = None
            for p in photo["video_files"]:
                if p["width"] == 1280 and p["height"] == 720:
                    original_image = p["link"]
                    break

            if original_image is None:
                continue

            video_length = photo["duration"]

            file_name = str(count) + ".mp4"
            save_file_image = Image.image_output_directory.joinpath(file_name)

            print(file_name)

            logging.info("URL image: `%s`.", original_image)
            response = requests.get(original_image, timeout=self.default_timeout)
            if response.status_code == 200:
                Image.save_image_from_response(response, save_file_image)
                logging.info("Saved image: `%s`.", save_file_image)
                count += 1

        return count

    def _send_request_and_iterate_over_response_photos(
        self, url: str, payload: dict, image_count: int = 0
    ) -> tuple:
        response = self._send_request(url, parameters=payload).json()

        photos = response["videos"]
        if len(photos) == 0:
            logging.warning("No images found on this page.")

        logging.info("Start scraping page `%s`.", response["page"])
        return response, self._iterate_response_photos(photos, image_count)

    def search_for_photos(self, payload: dict):
        """Search for Photos"""
        logging.info("Sending first request payload.")

        (
            response,
            save_image_count,
        ) = self._send_request_and_iterate_over_response_photos(
            url=self.base, payload=payload
        )

        while response["next_page"]:
            (
                response,
                save_image_count,
            ) = self._send_request_and_iterate_over_response_photos(
                url=response["next_page"], payload=payload, image_count=save_image_count
            )

        logging.info("Scraping complete for a total of `%s` images.", save_image_count)
