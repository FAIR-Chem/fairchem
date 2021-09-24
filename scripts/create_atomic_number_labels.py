import argparse
import requests

TEMPLATE = (
    lambda labels: f"""\
ATOMIC_NUMBER_LABELS = {str(labels)}
"""
)


def main(args: argparse.Namespace):
    result = requests.get(args.input).json()
    labels = {
        element["number"]: element["symbol"] for element in result["elements"]
    }
    file_content = TEMPLATE(labels)
    if args.stdout:
        print(file_content)
    else:
        with open(args.output, "w") as f:
            f.write(file_content)

        print(f"Wrote {len(labels)} labels to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create labels for atomic numbers."
    )
    parser.add_argument(
        "-i",
        "--input",
        help="URL to the JSON file containing the atomic numbers.",
        default="https://raw.githubusercontent.com/Bowserinator/Periodic-Table-JSON/master/PeriodicTableJSON.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file name.",
        default="ocpmodels/datasets/embeddings/atomic_number_labels.py",
    )
    parser.add_argument(
        "-s",
        "--stdout",
        help="Output to stdout instead of a file.",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
