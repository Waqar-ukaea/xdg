#!/usr/bin/env python3
import csv
import subprocess


BVH_ROBUSTNESS_CHECK = "./build/cubql_llvm_ada/tools/cubql-bvh-robustness"
MESH = "../xdg-event-particle-sim/atr.h5m"
VOLUME = "5250"
RAY_ORIGIN = ["2.0338775313376072", "-1.5124337131024441", "-20.769725351874694"]
RAY_DIRECTION = ["-0.0252313333856292", "0.38871772275741456", "-0.92101135271497769"]
BVH_BUILD_MODE = "--full-model-bvh"
NO_HIT_SURFACE = "-1"
DEFAULT_OUTPUT = "BVH_inconsistency_investigation/BVH_inconsistency_volume_5250_full_model.csv"


def parse_ray_fire_output(output):
    result = {
        "distance": "",
        "surface": "",
        "primitive": "",
        "next_volume": "",
        "boundary_condition": "",
        "stdout": output,
    }

    for line in output.splitlines():
        if line.startswith("Distance:"):
            result["distance"] = line.split(":", 1)[1].strip()
        elif line.startswith("Surface:"):
            result["surface"] = line.split(":", 1)[1].strip()
        elif line.startswith("Primitive:"):
            result["primitive"] = line.split(":", 1)[1].strip()
        elif line.startswith("Next volume:"):
            result["next_volume"] = line.split(":", 1)[1].strip()
        elif line.startswith("Boundary condition:"):
            result["boundary_condition"] = line.split(":", 1)[1].strip()

    return result


def one_line(text):
    return text.replace("\r", "\\r").replace("\n", "\\n")


def print_failure_diagnostics(stdout, stderr):
    """Print everything emitted by the robustness executable on failure."""
    print("--- cubql_bvh_robustness stdout ---")
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    else:
        print("(no stdout)")

    if stderr:
        print("--- cubql_bvh_robustness stderr ---")
        print(stderr, end="" if stderr.endswith("\n") else "\n")


def main():
    n = int(input("Number of ray-fire runs: "))

    command = [
        BVH_ROBUSTNESS_CHECK,
        MESH,
        VOLUME,
        "-r",
        "CUBQL",
        "--batch",
        BVH_BUILD_MODE,
        "-o",
        *RAY_ORIGIN,
        "-d",
        *RAY_DIRECTION,
    ]

    with open(DEFAULT_OUTPUT, "w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "run",
                "returncode",
                "distance",
                "surface",
                "primitive",
                "next_volume",
                "boundary_condition",
                "failed",
                "stdout",
                "stderr",
            ],
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            lineterminator="\n",
        )
        writer.writeheader()

        failed_runs = []

        for run in range(1, n + 1):
            completed = subprocess.run(command, text=True, capture_output=True)
            result = parse_ray_fire_output(completed.stdout)
            try:
                distance = float(result["distance"])
            except ValueError:
                distance = None

            failed = (
                completed.returncode != 0
                or distance is None
                or result["surface"] == NO_HIT_SURFACE
            )
            if failed:
                failed_runs.append(run)

            writer.writerow({
                "run": run,
                "returncode": completed.returncode,
                "distance": result["distance"],
                "surface": result["surface"],
                "primitive": result["primitive"],
                "next_volume": result["next_volume"],
                "boundary_condition": result["boundary_condition"],
                "failed": failed,
                "stdout": one_line(completed.stdout),
                "stderr": one_line(completed.stderr),
            })

            print(
                f"{run}: returncode={completed.returncode} "
                f"distance={result['distance']} surface={result['surface']} "
                f"primitive={result['primitive']} failed={failed}"
            )

            if failed:
                print_failure_diagnostics(completed.stdout, completed.stderr)
                output_file.flush()
                print(f"Stopping after first failed BVH build (run {run}).")
                break

    print(f"Failed runs: {failed_runs}")


if __name__ == "__main__":
    main()
