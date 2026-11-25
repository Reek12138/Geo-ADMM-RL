"""Retain early teacher examples from worlds that also complete in native BlueSky."""

import argparse
import json
from pathlib import Path

import torch

from eval.bluesky_pair import _checksum


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    summary = {}
    for source in args.plans:
        plan = json.loads(source.read_text())
        family, = plan['families']
        report = plan['families'][family]
        digest = _checksum(source)
        allowed = []
        count = len(report['search'])
        for index in range(count):
            path = args.native / f'{family}-{index}' / 'summary.json'
            audit = json.loads(path.read_text())
            if (audit['plan_sha256'] != digest or audit['event_sampling_seconds'] != .05
                    or audit['heading_recovery'] != 'geodesic' or audit['world'] != index
                    or audit['family'] != family or audit['torch_layout_seed'] != report['seed']
                    or audit['aircraft'] != plan['aircraft']
                    or sum(audit['outcomes'].values()) != plan['aircraft']):
                raise ValueError(f'Native audit does not match teacher plan: {path}')
            if audit['outcomes']['arrivals'] == plan['aircraft']:
                allowed.append(index)
        rows = report['teacher_rows']
        record = {'worlds': count, 'verified_worlds': allowed, 'rows_before': rows, 'rows_kept': 0,
                  'plan': str(source), 'plan_sha256': digest}
        summary[family] = record
        if not rows:
            continue
        source_data = args.teacher / family / f'{family}.pt'
        if _checksum(source_data) != report['teacher_sha256']:
            raise ValueError(f'Teacher examples do not match the preview: {source_data}')
        examples = torch.load(source_data, map_location='cpu', weights_only=True)
        if examples['family'] != family or examples['seed'] != report['seed'] or len(examples['world']) != rows:
            raise ValueError(f'Teacher rows do not match the preview: {source_data}')
        keep = torch.isin(examples['world'], torch.tensor(allowed, dtype=torch.long))
        if not keep.any():
            continue
        target = args.output / f'{family}.pt'
        selected = {
            **examples,
            'graph': tuple(value[keep] for value in examples['graph']),
            **{key: examples[key][keep] for key in ('commands', 'world', 'decision')},
            'native_verification': str(args.native),
            'plan_sha256': digest,
        }
        torch.save(selected, target)
        record.update(rows_kept=int(keep.sum()), output_sha256=_checksum(target))
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plans', type=Path, nargs='+', required=True)
    cli.add_argument('--teacher', type=Path, required=True)
    cli.add_argument('--native', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args())
