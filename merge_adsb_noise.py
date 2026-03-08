"""Compatibility wrapper for the canonical scripts/merge_adsb_noise.py entrypoint."""

from scripts.merge_adsb_noise import *  # noqa: F401,F403
from scripts.merge_adsb_noise import main


if __name__ == "__main__":
    main()
