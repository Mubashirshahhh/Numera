import argparse
import uuid

from .pipeline import init_db, process_job


def main():
    parser = argparse.ArgumentParser(
        prog="numera",
        description="Numera AI Mathematics Visualization Engine"
    )

    parser.add_argument(
        "expression",
        help="Mathematical expression or prompt to visualize"
    )

    parser.add_argument(
        "--xmin",
        type=float,
        default=-5,
        help="Minimum x value"
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=5,
        help="Maximum x value"
    )

    parser.add_argument(
        "--step",
        type=float,
        default=1,
        help="Grid step size"
    )

    args = parser.parse_args()

    init_db()

    job_id = str(uuid.uuid4())

    ui_params = {
        "x_range": [
            args.xmin,
            args.xmax,
            args.step
        ]
    }

    process_job(
        job_id=job_id,
        user_math_request=args.expression,
        params=ui_params
    )

    print(f"\n✓ Job completed successfully.")
    print(f"Job ID: {job_id}")


if __name__ == "__main__":
    main()
