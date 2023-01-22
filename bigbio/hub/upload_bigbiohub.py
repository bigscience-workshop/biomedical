"""
Upload local bigbiohub.py to hub repos.
"""
import argparse
import hubtools


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hotyb',
        action='store_true',
        help="set this flag and hold onto your butts",
    )
    args = parser.parse_args()
    dryrun = not args.hotyb
    hubtools.upload_bigbiohub(dryrun=dryrun)
