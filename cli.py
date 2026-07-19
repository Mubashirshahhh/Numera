import argparse
from numera.pipeline import generate_animation

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "expression",
        help="Mathematical expression"
    )

    args = parser.parse_args()

    video = generate_animation(args.expression)

    print(video)
