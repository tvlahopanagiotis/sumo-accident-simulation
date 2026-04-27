# Command: `sas-generate-sioux-falls`

Generate the Sioux Falls benchmark network for SUMO.

## What It Does

1. writes the benchmark node and edge definitions
2. compiles the network for SUMO
3. generates synthetic demand
4. writes a `.sumocfg`
5. optionally patches a YAML config

## Main Inputs

- built-in benchmark topology
- optional demand `--period`

## Main Outputs

Under the benchmark network directory:

- SUMO network files
- route file
- `.sumocfg`

## Typical Usage

```bash
sas-generate-sioux-falls
```

```bash
sas-generate-sioux-falls --update-config
```

```bash
sas-generate-sioux-falls --period 2.0
```

## Operational Notes

- This is the smallest bundled benchmark and is useful for rapid testing.
- Use it when you want fast turnaround on simulator logic rather than city
  realism.
