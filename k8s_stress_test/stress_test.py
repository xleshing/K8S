import argparse
import random
import subprocess
import time
import yaml


def generate_job_yaml(job_name, cpu_limit, memory_limit, cpu_request, memory_request, duration):
    job_yaml = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": "default"
        },
        "spec": {
            "ttlSecondsAfterFinished": 10,
            "template": {
                "metadata": {
                    "name": job_name
                },
                "spec": {
                    "containers": [
                        {
                            "name": "stress",
                            "image": "containerstack/stress",
                            "imagePullPolicy": "Always",
                            "command": ["stress"],
                            "args": [
                                "--cpu", str(cpu_limit),
                                "--vm", "1",
                                "--vm-bytes", memory_limit,
                                "--timeout", f"{duration}s"
                            ],
                            "resources": {
                                "limits": {
                                    "memory": memory_limit,
                                    "cpu": f"{cpu_limit}m"
                                },
                                "requests": {
                                    "memory": memory_request,
                                    "cpu": f"{cpu_request}m"
                                }
                            }
                        }
                    ],
                    "restartPolicy": "Never"
                }
            },
            "backoffLimit": 0
        }
    }
    return job_yaml


def submit_job(job_yaml):
    yaml_content = yaml.dump(job_yaml)
    subprocess.run(["kubectl", "apply", "-f", "-"], input=yaml_content.encode(), check=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Randomly generate Kubernetes stress test Jobs")
    parser.add_argument("--cpu-limit-range", type=str, default="500,3000",
                        help="Range of CPU limits in the format 'min,max', default is '500,3000'")
    parser.add_argument("--memory-limit-range", type=str, default="128,1024",
                        help="Range of memory limits in the format 'min,max', default is '128,1024' (unit: Mi)")
    parser.add_argument("--job-count-range", type=str, default="1,10",
                        help="Range of the number of jobs in the format 'min,max', default is '1,10'")
    parser.add_argument("--duration-range", type=str, default="60,600",
                        help="Range of job duration in the format 'min,max', default is '60,600' seconds")
    parser.add_argument("--script-duration", type=int, default=1800,
                        help="Total running time of the script, default is 1800 seconds")
    parser.add_argument("--cpu-request-percent", type=float, default=80.0,
                        help="Percentage of CPU requests relative to limits, default is 80%%")
    parser.add_argument("--memory-request-percent", type=float, default=80.0,
                        help="Percentage of Memory requests relative to limits, default is 80%%")
    parser.add_argument("--sleep-time", type=int, default=5,
                        help="Wait time (seconds) after each job submission, default is 5 seconds")

    args = parser.parse_args()

    try:
        for arg_name in ["cpu_limit_range", "memory_limit_range", "job_count_range", "duration_range"]:
            if len(args.__dict__[arg_name].split(',')) != 2:
                raise ValueError(f"Parameter {arg_name.replace('_', '-')} must be in 'min,max' format.")

        if not (0 <= args.cpu_request_percent <= 100):
            raise ValueError("--cpu-request-percent must be between 0 and 100.")
        if not (0 <= args.memory_request_percent <= 100):
            raise ValueError("--memory-request-percent must be between 0 and 100.")
    except ValueError as e:
        parser.error(str(e))

    return args


def main():
    args = parse_arguments()

    cpu_limit_range = list(map(int, args.cpu_limit_range.split(',')))
    memory_limit_range = list(map(int, args.memory_limit_range.split(',')))
    job_count_range = list(map(int, args.job_count_range.split(',')))
    duration_range = list(map(int, args.duration_range.split(',')))
    script_duration = args.script_duration
    cpu_request_percent = args.cpu_request_percent / 100
    memory_request_percent = args.memory_request_percent / 100
    sleep_time = args.sleep_time

    start_time = time.time()

    while time.time() - start_time < script_duration:
        job_count = random.randint(*job_count_range)
        for i in range(job_count):
            cpu_limit = random.randint(*cpu_limit_range)
            memory_limit = random.randint(*memory_limit_range)
            memory_limit_str = f"{memory_limit}M"
            cpu_request = int(cpu_limit * cpu_request_percent)
            memory_request = f"{int(memory_limit * memory_request_percent)}M"

            duration = random.randint(*duration_range)

            job_name = f"stress-test-{int(time.time())}-{i}"

            job_yaml = generate_job_yaml(job_name, cpu_limit, memory_limit_str, cpu_request, memory_request, duration)
            submit_job(job_yaml)

        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
