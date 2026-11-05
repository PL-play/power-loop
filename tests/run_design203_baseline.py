"""Sequential timing runs; keeps database work and CPU workloads from overlapping."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.design203_benchmark import SCENARIOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', default=['sqlite', 'postgres', 'mysql'])
    parser.add_argument('--scenarios', nargs='+', default=list(SCENARIOS))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for backend in args.backends:
        dsn = str(args.out_dir / 'timing_test.db') if backend == 'sqlite' else os.environ[
            'POWER_LOOP_TASK_PG_DSN' if backend == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
        for scenario in args.scenarios:
            for concurrency in ([8] if scenario == 'same_session' else [1, 8]):
                name = f'{backend}-{scenario}-c{concurrency}'
                output = args.out_dir / (name + '.json')
                if output.exists():
                    raise ValueError(f'refusing to overwrite a frozen measurement: {output}')
                command = [sys.executable, '-m', 'tests.design203_benchmark', '--dsn', dsn,
                           '--scenario', scenario, '--concurrency', str(concurrency),
                           '--head', head, '--out', str(output)]
                print(json.dumps({'starting': name}), flush=True)
                result = subprocess.run(command)
                if backend == 'sqlite':
                    for suffix in ('', '-wal', '-shm'):
                        Path(dsn + suffix).unlink(missing_ok=True)
                if result.returncode:
                    raise SystemExit(result.returncode)
                print(json.dumps({'finished': name}), flush=True)


if __name__ == '__main__':
    main()
