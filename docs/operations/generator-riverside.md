# Command: `sas-generate-riverside`

Generate the synthetic Riverside District SUMO network.

## What It Does

1. writes the synthetic node, edge, and type definitions
2. compiles the network with `netconvert`
3. writes a demand file and `.sumocfg`
4. optionally patches a YAML config

## Main Inputs

- built-in synthetic network geometry
- optional output directory override

## Main Outputs

Under `data/synthetic/riverside/network/` by default:

- `riverside.nod.xml`
- `riverside.edg.xml`
- `riverside.typ.xml`
- `riverside.net.xml`
- `riverside.rou.xml`
- `riverside.sumocfg`

## Typical Usage

```bash
sas-generate-riverside
```

```bash
sas-generate-riverside --out-dir /tmp/riverside-test
```

```bash
sas-generate-riverside --update-config
```

## Operational Notes

- This is a synthetic development network, not a real city.
- It is useful for deterministic experiments and fast local iteration.
