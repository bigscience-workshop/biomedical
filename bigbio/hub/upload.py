"""
Upload local bigbiohub.py to hub repos.
"""
import argparse
import hubtools


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset to be uploaded",
    )
    parser.add_argument(
        "-c", "--create",
        action="store_true",
        help="Set this flag if a the repo should be created first (before uploading files)",
    )
    parser.add_argument(
        "-d", "--dryrun",
        action="store_true",
        help="Set this flag to test your command without uploading to the hub",
    )
    args = parser.parse_args()

    hubtools.update_dataset(
        dataset_name=args.dataset,
        create_repo=args.create,
        dryrun=args.dryrun
    )


