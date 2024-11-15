# K8s Stress Test Script

## Installing dependencies

```bash
pip install pyyaml
```

## Quick Start

To execute the stress test script, use the following command:

```bash
python stress_test.py
```
## Parameters
Below is a detailed explanation of the script parameters and their default values:

```bash
python stress_test.py \
    --cpu-limit-range \
    --memory-limit-range \
    --job-count-range \
    --duration-range \
    --script-duration \
    --cpu-request-percent \
    --sleep-time
```
## Parameter Descriptions
- ```--cpu-limit-range```
  - Default: ```500,3000```
  -  Specifies the CPU limit range in millicores (m). For example, ```500m``` to ```3000m```.
- ```--memory-limit-range```

  - Default: ```128,1024```
  - Specifies the memory limit range in MiB (Mi). For example, ```128Mi``` to ```1024Mi```.
- ```--job-count-range```

  - Default: ```1,10```
  - Indicates the range for the number of Jobs generated per batch. For example, between ```1``` and ```10``` Jobs.
- ```--duration-range```

  - Default: ```60,600```
  - Specifies the range of duration for each Job, in seconds. For example, between ```60``` and ```600``` seconds.
- ```--script-duration```

  - Default: ```1800```
  - Defines the total runtime of the script, in seconds.
- ```--cpu-request-percent```

  - Default: ```80.0```
  - Sets the percentage of CPU requests relative to the CPU limit. For example, ```80%``` of the limit.
- ```--sleep-time```

  - Default: ```5```
  - Specifies the interval, in seconds, to wait after each Job submission.
