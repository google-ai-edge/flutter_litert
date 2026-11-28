#!/usr/bin/env python3
"""Plans and runs the repeat runs of new and changed tests (test-repeat.yml).

  python3 -I scripts/test_repeat.py plan --changes changes.txt
  python3 -I scripts/test_repeat.py run

A test counts as trusted only when it passes many runs in a row, alone, on the
runners CI uses. `plan` finds the tests a pull request adds or changes (from
`git diff --name-status`), or takes the files and lanes a dispatch names, and
maps each one to the lanes of .github/test-lanes.json that run it. A changed
helper adds the tests that import it, or name its path. Each lane gets one job,
or up to max_shards jobs that split the runs when one job would take more than
budget_fraction of the lane's timeout, or its emulator step more than that
fraction of the step's own limit. The matrix goes to $GITHUB_OUTPUT.

`run` runs one job of that matrix: every file of the lane, the job's number of
runs, with the lane's exact command. Unit lanes shuffle each run with its own
seed, printed with the run. A failed run is never run again: the remaining runs
go on, and the job fails at the end with how many failed. Each run's JSON report
gives its passed, skipped and failed counts; a file whose tests only skipped on
the lane gets a warning, since the lane did not exercise it.

Inputs come from the environment the workflow sets (TEST_REPEAT_*), never from
the command line, so nothing from a pull request or a dispatch is interpolated
into a shell command.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANES_FILE = '.github/test-lanes.json'
FILE_RE = re.compile(r'^[A-Za-z0-9_.][A-Za-z0-9_./-]*$')
LANE_RE = re.compile(r'^[a-z0-9][a-z0-9-]*$')
DEFAULT_RUN_TIMEOUT = 20


class PlanError(Exception):
    pass


def load_table(root=ROOT):
    with open(os.path.join(root, LANES_FILE), encoding='utf-8') as f:
        return json.load(f)


# Globs: ** spans directories, * and ? stay within one.
def glob_regex(pattern):
    out = ''
    i = 0
    while i < len(pattern):
        if pattern.startswith('**/', i):
            out += '(?:[^/]+/)*'
            i += 3
        elif pattern.startswith('**', i):
            out += '.*'
            i += 2
        elif pattern[i] == '*':
            out += '[^/]*'
            i += 1
        elif pattern[i] == '?':
            out += '[^/]'
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r'\Z')


def matches(path, patterns):
    return any(glob_regex(p).match(path) for p in patterns)


def read_text(root, path):
    try:
        with open(os.path.join(root, path), encoding='utf-8', errors='replace') as f:
            return f.read()
    except OSError:
        return None


# @TestOn platform selectors: names, !, &&, || and parentheses.
TEST_ON_RE = re.compile(r'''^\s*@TestOn\(\s*(['"])(.*?)\1\s*\)''', re.M)


def test_on(text):
    m = TEST_ON_RE.search(text or '')
    return m.group(2) if m else None


def platform_matches(expr, names):
    tokens = re.findall(r'\|\||&&|[!()]|[A-Za-z0-9_-]+|\S', expr)
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def take():
        pos[0] += 1
        return tokens[pos[0] - 1]

    def unary():
        tok = take() if peek() is not None else None
        if tok == '!':
            return not unary()
        if tok == '(':
            value = either()
            if take() != ')':
                raise ValueError(expr)
            return value
        if tok is None or not re.match(r'^[A-Za-z0-9_-]+$', tok):
            raise ValueError(expr)
        return tok in names

    def both():
        value = unary()
        while peek() == '&&':
            take()
            value = unary() and value
        return value

    def either():
        value = both()
        while peek() == '||':
            take()
            value = both() or value
        return value

    value = either()
    if peek() is not None:
        raise ValueError(expr)
    return value


def entry_of(item):
    return {'path': item} if isinstance(item, str) else dict(item)


def lane_entry(lane, path):
    for item in lane.get('files', []):
        entry = entry_of(item)
        if entry['path'] == path:
            return entry
    if matches(path, lane.get('include', [])):
        return {'path': path}
    return None


def walk(root, top):
    for base, dirs, names in os.walk(os.path.join(root, top)):
        dirs.sort()
        for name in sorted(names):
            yield os.path.relpath(os.path.join(base, name), root).replace(os.sep, '/')


def lane_all_files(root, lane):
    paths = [entry_of(item)['path'] for item in lane.get('files', [])]
    for pattern in lane.get('include', []):
        top = pattern.split('*')[0].rsplit('/', 1)[0]
        paths += [p for p in walk(root, top) if matches(p, [pattern])]
    return sorted(set(paths))


def runs_on_lane(root, lane, path):
    """None when the lane runs the file, otherwise why it does not."""
    names = lane.get('test_on')
    if not names:
        return None
    expr = test_on(read_text(root, path))
    if expr is None:
        return None
    try:
        return None if platform_matches(expr, set(names)) else "@TestOn('%s')" % expr
    except ValueError:
        return None


def under(path, roots):
    return any(path == r or path.startswith(r + '/') for r in roots)


def is_test(table, path):
    if path.endswith('_test.dart') and under(path, table['test_roots']):
        return True
    return any(lane_entry(lane, path) for lane in table['lanes'].values())


IMPORT_RE = re.compile(r'''^\s*(?:import|export|part)\s+['"]([^'"]+)['"]''', re.M)


def dart_files(root, table):
    files = []
    for top in table['test_roots']:
        files += [p for p in walk(root, top) if p.endswith('.dart')]
    return files


def users_of(root, table, helper):
    """Test files that import the helper, directly or through other files, or
    name its repo-relative path (a test that runs it in another process)."""
    users = {}
    for path in dart_files(root, table):
        text = read_text(root, path) or ''
        deps = set()
        for uri in IMPORT_RE.findall(text):
            if ':' in uri:
                continue
            deps.add(os.path.normpath(os.path.join(os.path.dirname(path), uri)).replace(os.sep, '/'))
        for target in deps:
            users.setdefault(target, set()).add(path)
        for other in re.findall(r'''['"]([A-Za-z0-9_./-]+\.dart)['"]''', text):
            users.setdefault(other, set()).add(path)
    found = set()
    todo = [helper]
    while todo:
        for user in users.get(todo.pop(), ()):
            if user not in found and user != helper:
                found.add(user)
                todo.append(user)
    return sorted(p for p in found if is_test(table, p))


def parse_changes(text):
    """Paths a `git diff --name-status -M` adds, modifies or renames to."""
    paths = []
    for line in text.splitlines():
        fields = line.split('\t')
        if len(fields) < 2 or not fields[0]:
            continue
        status = fields[0][0]
        if status == 'D':
            continue
        paths.append(fields[-1])
    return paths


def split_words(value):
    return [w for w in re.split(r'[\s,]+', value or '') if w]


def parse_runs(value, limits):
    if not (value or '').strip():
        return limits['default_runs']
    if not re.match(r'^[0-9]+$', value.strip()):
        raise PlanError('runs must be a whole number, not %r' % value)
    runs = int(value)
    if not 1 <= runs <= limits['max_runs']:
        raise PlanError('runs must be from 1 to %d, not %d' % (limits['max_runs'], runs))
    return runs


def run_minutes(lane, entries):
    return lane.get('iteration_minutes', 0) + sum(
        e.get('minutes', lane['run_minutes']) for e in entries)


def build_minutes(lane, entries):
    """Each job's first run builds the app from cold; a file entry adds its own
    build when it builds the app another way."""
    return lane.get('build_minutes', 0) + sum(e.get('build_minutes', 0) for e in entries)


def step_limit(lane):
    """The emulator step's own limit in minutes, for an Android lane whose
    setup sets step_limit, else None. A start that hangs before any test runs
    then fails the step instead of the job, and the second emulator start
    runs: the step gets half of the job's time after its setup, so the second
    start has as long as the first."""
    android = lane.get('setup', {}).get('android') or {}
    if not android.get('step_limit'):
        return None
    return int((lane['timeout'] - lane['fixed_minutes']) // 2)


def shard(lane_id, lane, entries, runs, limits):
    """Splits the runs over as few jobs as keep each one within budget, and
    each emulator step within the same fraction of its own limit."""
    budget = limits['budget_fraction'] * lane['timeout']
    per_run = run_minutes(lane, entries)
    build = build_minutes(lane, entries)
    fixed = lane['fixed_minutes'] + build
    limit = step_limit(lane)
    for shards in range(1, min(limits['max_shards'], runs) + 1):
        need = fixed + math.ceil(runs / shards) * per_run
        step = build + math.ceil(runs / shards) * per_run
        if need <= budget and (limit is None or step <= limits['budget_fraction'] * limit):
            return [runs // shards + (1 if k < runs % shards else 0) for k in range(shards)], need
    most = math.ceil(runs / min(limits['max_shards'], runs)) * per_run
    if limit is not None and fixed + most <= budget:
        raise PlanError(
            '%s: %d runs of %s need about %.0f minutes of emulator step in each of %d jobs, '
            'over the %.0f minute budget (%s of its %d minute step limit). Start test-repeat '
            'by hand with fewer files or runs at a time.' % (
                lane_id, runs, ' '.join(e['path'] for e in entries) if len(entries) <= 4 else '%d files' % len(entries),
                build + most, min(limits['max_shards'], runs), limits['budget_fraction'] * limit,
                limits['budget_fraction'], limit))
    raise PlanError(
        '%s: %d runs of %s need about %.0f minutes in each of %d jobs, over the '
        '%.0f minute budget (%s of its %d minute timeout). Start test-repeat by hand '
        'with fewer files or runs at a time.' % (
            lane_id, runs, ' '.join(e['path'] for e in entries) if len(entries) <= 4 else '%d files' % len(entries),
            fixed + most, min(limits['max_shards'], runs), budget, limits['budget_fraction'], lane['timeout']))


def job_fields(lane_id, lane):
    setup = lane.get('setup', {})
    android = setup.get('android') or {}
    return {
        'lane': lane_id,
        'runner': lane['runner'],
        'timeout': lane['timeout'],
        'flutter': bool(setup.get('flutter')),
        'free_disk': bool(setup.get('free_disk')),
        'xcode': setup.get('xcode', ''),
        'apt': setup.get('apt', ''),
        'desktop': setup.get('desktop', ''),
        'pub_get': ' '.join(setup.get('pub_get', [])),
        'flex_cache': setup.get('flex_cache', ''),
        'ios_simulator': bool(setup.get('ios_simulator')),
        'android': bool(android),
        'emu_api': str(android.get('api', '')),
        'emu_target': android.get('target', ''),
        'emu_ram': android.get('ram', ''),
        'emu_cores': str(android.get('cores', '')),
        'sdk_packages': android.get('packages', ''),
        'gradle_downloads': android.get('downloads', ''),
        # The job's own timeout when the lane sets no step limit.
        'emu_timeout': step_limit(lane) or lane['timeout'],
        'hooks_cache': setup.get('hooks_cache', ''),
    }


def plan(root, table, event, changes='', files='', lanes='', runs=''):
    """Returns (jobs, notes); notes are (level, message) for the log."""
    limits = table['limits']
    all_lanes = table['lanes']
    notes = []
    runs = parse_runs(runs, limits)
    named = split_words(lanes)
    for lane_id in named:
        if not LANE_RE.match(lane_id) or lane_id not in all_lanes:
            raise PlanError('no lane named %r; the lanes are %s' % (lane_id, ', '.join(sorted(all_lanes))))
    if event == 'pull_request':
        named = []
        changed = parse_changes(changes)
    else:
        changed = split_words(files)
        for path in changed:
            if not FILE_RE.match(path) or '..' in path.split('/'):
                raise PlanError('%r is not a repo-relative file path' % path)
            if not os.path.isfile(os.path.join(root, path)):
                raise PlanError('%s does not exist on this ref' % path)
        if not changed and not named:
            raise PlanError('name the test files to repeat (files), a lane (lanes), or both')
    selected = named or [l for l, lane in all_lanes.items() if lane.get('auto', True)]

    wanted = {}  # path -> why
    lane_wide = {}  # lane -> why

    def want(path, why):
        wanted.setdefault(path, why)

    for path in changed:
        if is_test(table, path):
            want(path, 'changed' if event == 'pull_request' else 'named in files')
        elif path.endswith('.dart') and under(path, table['test_roots']):
            users = users_of(root, table, path)
            if not users:
                notes.append(('notice', '%s: no test file uses it' % path))
            for user in users:
                want(user, 'uses %s' % path)
        for lane_id in selected:
            if matches(path, all_lanes[lane_id].get('deps', [])):
                lane_wide.setdefault(lane_id, 'changed %s' % path)
        if event != 'pull_request' and not is_test(table, path) and not path.endswith('.dart'):
            notes.append(('warning', '%s is not a test file' % path))
    if event != 'pull_request' and not changed:
        for lane_id in named:
            lane_wide.setdefault(lane_id, 'named in lanes')

    per_lane = {}  # lane -> {path: entry}

    def add(lane_id, entry):
        per_lane.setdefault(lane_id, {})[entry['path']] = entry

    for path in sorted(wanted):
        covering = []
        for lane_id, lane in all_lanes.items():
            entry = lane_entry(lane, path)
            if entry is None:
                continue
            reason = runs_on_lane(root, lane, path)
            if reason:
                if lane_id in selected and named:
                    notes.append(('warning', '%s does not run on %s: %s' % (path, lane_id, reason)))
                continue
            covering.append((lane_id, entry))
        chosen = [(l, e) for l, e in covering if l in selected]
        if not chosen:
            others = [l for l, _ in covering]
            if not others:
                notes.append(('warning', '%s (%s): no lane runs it. flutter-ci.yml does not run it '
                              'either; add it to a job there and to .github/test-lanes.json.' % (path, wanted[path])))
            elif named:
                notes.append(('warning', '%s: the lanes named (%s) do not run it; %s do.' % (
                    path, ', '.join(named), ', '.join(others))))
            else:
                notes.append(('warning', '%s (%s): only lanes that never run on their own run it (%s). '
                              'Start test-repeat by hand with files=%s and lanes=%s.' % (
                                  path, wanted[path], ', '.join(others), path, ' '.join(others))))
            continue
        for lane_id, entry in chosen:
            add(lane_id, entry)

    for lane_id, why in sorted(lane_wide.items()):
        lane = all_lanes[lane_id]
        for path in lane_all_files(root, lane):
            if runs_on_lane(root, lane, path) is None:
                add(lane_id, lane_entry(lane, path))
        notes.append(('notice', '%s runs all its files: %s' % (lane_id, why)))

    jobs = []
    for lane_id in sorted(per_lane):
        lane = all_lanes[lane_id]
        entries = [per_lane[lane_id][p] for p in sorted(per_lane[lane_id])]
        counts, need = shard(lane_id, lane, entries, runs, limits)
        first = 1
        for k, count in enumerate(counts):
            job = job_fields(lane_id, lane)
            label = '%s: %d runs' % (lane_id, runs) if len(counts) == 1 else \
                '%s %d/%d: runs %d-%d of %d' % (lane_id, k + 1, len(counts), first, first + count - 1, runs)
            job.update({
                'id': '%s-%d' % (lane_id, k + 1),
                'name': label,
                'files': ' '.join(e['path'] for e in entries),
                'runs': count,
                'first': first,
                'total': runs,
                'estimate': round(need),
            })
            jobs.append(job)
            first += count
    cap = limits['max_jobs'].get(event, limits['max_jobs']['workflow_dispatch'])
    if len(jobs) > cap:
        raise PlanError(
            '%d repeat jobs are needed, more than the %d a %s may start: %s. Start '
            'test-repeat by hand (Actions, test repeat, Run workflow) with some of the '
            'files or lanes at a time.' % (len(jobs), cap, event.replace('_', ' '),
                                           ', '.join(j['name'] for j in jobs)))
    return jobs, notes


def cmd_plan(args):
    table = load_table(ROOT)
    changes = ''
    if args.changes and os.path.exists(args.changes):
        with open(args.changes, encoding='utf-8') as f:
            changes = f.read()
    event = os.environ.get('TEST_REPEAT_EVENT', 'workflow_dispatch')
    try:
        jobs, notes = plan(ROOT, table, event, changes,
                           os.environ.get('TEST_REPEAT_FILES', ''),
                           os.environ.get('TEST_REPEAT_LANES', ''),
                           os.environ.get('TEST_REPEAT_RUNS', ''))
    except PlanError as e:
        print('::error title=test-repeat plan::%s' % e, flush=True)
        return 1
    for level, message in notes:
        print('::%s::%s' % (level, message), flush=True)
    lines = ['## Repeat runs', '']
    if jobs:
        lines += ['| job | runner | files | estimate |', '| --- | --- | --- | --- |']
        lines += ['| %s | %s | %s | %d min |' % (j['name'], j['runner'], '<br>'.join(j['files'].split()), j['estimate'])
                  for j in jobs]
    else:
        lines.append('No test to repeat.')
    lines += [''] + ['- %s: %s' % (level, message) for level, message in notes]
    print('\n'.join(lines), flush=True)
    write_env_file('GITHUB_STEP_SUMMARY', '\n'.join(lines) + '\n')
    write_env_file('GITHUB_OUTPUT', 'matrix=%s\ncount=%d\n' % (json.dumps({'include': jobs}), len(jobs)))
    return 0


def write_env_file(name, text):
    path = os.environ.get(name)
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text)


# --- run -------------------------------------------------------------------

VAR_RE = re.compile(r'\$(?:\{(\w+)\}|(\w+))')


def expand(value, env, values):
    for key, replacement in values.items():
        value = value.replace('{%s}' % key, replacement)
    return VAR_RE.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), value)


def build_argv(template, env, values, files, args):
    argv = []
    for token in template:
        if token == '{files}':
            argv += files
        elif token == '{args}':
            argv += args
        else:
            argv.append(expand(token, env, values))
    return argv


def kill_tree(proc):
    if os.name == 'nt':
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(proc.pid)])
        return
    for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass


def execute(argv, cwd, env, timeout):
    """Returns (rc, seconds, timed_out). The command gets its own process group
    so a timeout ends everything it started."""
    exe = shutil.which(argv[0], path=env.get('PATH')) or argv[0]
    kwargs = {'start_new_session': True} if os.name != 'nt' else \
        {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    started = time.time()
    try:
        proc = subprocess.Popen([exe] + argv[1:], cwd=cwd, env=env, **kwargs)
    except OSError as e:
        print('::error::could not start %s: %s' % (argv[0], e), flush=True)
        return 127, 0.0, False
    execute.current = proc
    try:
        rc = proc.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        rc, timed_out = 124, True
    finally:
        execute.current = None
    return rc, time.time() - started, timed_out


execute.current = None


def hook(commands, cwd, env, limit=600):
    for command in commands:
        argv = [expand(token, env, {'python': sys.executable}) for token in command]
        execute(argv, cwd, env, limit)


def parse_report(path, paths):
    """Per test file counts from a flutter test JSON report, or None."""
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    suites = {}
    tests = {}
    counts = {p: {'passed': 0, 'skipped': 0, 'failed': 0, 'skipped_names': []} for p in paths}
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get('type') if isinstance(event, dict) else None
        if kind == 'suite':
            suite = event['suite']
            where = (suite.get('path') or '').replace('\\', '/')
            for p in paths:
                if where == p or where.endswith('/' + p):
                    suites[suite['id']] = p
        elif kind == 'testStart':
            test = event['test']
            tests[test['id']] = (suites.get(test.get('suiteID')), test.get('name', ''))
        elif kind == 'testDone' and not event.get('hidden'):
            p, name = tests.get(event.get('testID'), (None, ''))
            if p is None:
                if len(paths) != 1:
                    continue
                p = paths[0]
            if event.get('skipped'):
                counts[p]['skipped'] += 1
                counts[p]['skipped_names'].append(name)
            elif event.get('result') == 'success':
                counts[p]['passed'] += 1
            else:
                counts[p]['failed'] += 1
    return counts


def run(root, table, lane_id, files, runs, first, total, temp=None):
    """Runs the job and returns its exit code: 1 when any run failed."""
    lanes = table['lanes']
    if lane_id not in lanes:
        print('::error::no lane named %r' % lane_id, flush=True)
        return 2
    lane = lanes[lane_id]
    entries = []
    for path in files:
        entry = lane_entry(lane, path)
        if entry is None:
            print('::error::%s does not run %s' % (lane_id, path), flush=True)
            return 2
        entries.append(entry)
    if not entries or runs < 1:
        print('::error::nothing to run', flush=True)
        return 2
    cwd = os.path.join(root, lane['cwd'])
    work = os.path.join(temp or os.environ.get('RUNNER_TEMP') or tempfile.gettempdir(), 'test-repeat', lane_id)
    os.makedirs(os.path.join(work, 'reports'), exist_ok=True)
    base = dict(os.environ)
    base.setdefault('GITHUB_WORKSPACE', root)

    def rel(path):
        return os.path.relpath(os.path.join(root, path), cwd).replace(os.sep, '/')

    # The lane's environment for its services and prepare step.
    lane_env = dict(base)
    for key, value in lane.get('env', {}).items():
        lane_env[key] = expand(value, base, {'python': sys.executable, 'root': root,
                                             'logdir': os.path.join(work, 'android-logs', 'prepare')})
    services = []
    for command in lane.get('services', []):
        argv = [expand(token, lane_env, {'python': sys.executable}) for token in command]
        print('Starting %s' % ' '.join(argv), flush=True)
        services.append(subprocess.Popen(argv, cwd=cwd, env=lane_env))

    def stop(*_):
        if execute.current is not None:
            kill_tree(execute.current)
        for service in services:
            service.terminate()
            try:
                service.wait(timeout=15)
            except subprocess.TimeoutExpired:
                service.kill()
        if _:
            sys.exit(130)

    previous = signal.signal(signal.SIGTERM, stop) if os.name != 'nt' else None

    if lane.get('prepare'):
        print('Preparing %s' % lane_id, flush=True)
        rc, _, _ = execute(['bash', '-c', '\n'.join(lane['prepare'])], cwd, lane_env, 60 * 30)
        if rc != 0:
            print('::error::%s: preparing the lane failed (rc=%d), no run started' % (lane_id, rc), flush=True)
            stop()
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
            return 1

    groups = [entries] if lane.get('batch') else [[e] for e in entries]
    results = []
    first_of_job = True
    seeds = random.SystemRandom()
    for group in groups:
        paths = [e['path'] for e in group]
        label = paths[0] if len(paths) == 1 else '%d files' % len(paths)
        for number in range(first, first + runs):
            seed = str(seeds.randrange(1, 2 ** 31)) if lane.get('shuffle') else ''
            tag = re.sub(r'[^A-Za-z0-9_-]', '_', os.path.basename(paths[0]) if len(paths) == 1 else 'batch')
            report = os.path.join(work, 'reports', '%s-run%d.json' % (tag, number))
            logdir = os.path.join(work, 'android-logs', '%s-run%d' % (tag, number))
            values = {'seed': seed, 'report': report, 'logdir': logdir, 'root': root,
                      'python': sys.executable, 'file': rel(paths[0])}
            env = dict(base)
            for source in (lane.get('env', {}),) + tuple(e.get('env', {}) for e in group):
                for key, value in source.items():
                    env[key] = expand(value, base, values)
            args = group[0].get('args', lane.get('default_args', [])) if len(group) == 1 else []
            argv = build_argv(lane['command'], env, values, [rel(p) for p in paths], args)
            minutes = lane.get('first_run_timeout') if first_of_job and lane.get('first_run_timeout') \
                else lane.get('run_timeout', DEFAULT_RUN_TIMEOUT)
            first_of_job = False
            response = os.path.join(cwd, lane['response_file']) if lane.get('response_file') else None
            if response and os.path.exists(response):
                os.remove(response)
            if os.path.exists(report):
                os.remove(report)
            print('\n=== %s on %s: run %d of %d%s ===\n%s' % (
                label, lane_id, number, total, ', seed ' + seed if seed else '', ' '.join(argv)), flush=True)
            rc, seconds, timed_out = execute(argv, cwd, env, minutes * 60)
            hook(lane.get('after_run', []), cwd, env)
            if response:
                print('--- %s' % lane['response_file'], flush=True)
                try:
                    with open(response, encoding='utf-8', errors='replace') as f:
                        print(f.read(), flush=True)
                except OSError:
                    print('(none written)', flush=True)
            counts = parse_report(report, paths) if lane.get('report', True) else None
            if counts is not None and rc == 0 and any(c['failed'] for c in counts.values()):
                rc = 1
            results.append({'paths': paths, 'run': number, 'seed': seed, 'rc': rc,
                            'seconds': round(seconds, 1), 'timed_out': timed_out, 'counts': counts})
            tally = ''
            if counts is not None:
                tally = ' (%s)' % ', '.join('+%d ~%d -%d' % (c['passed'], c['skipped'], c['failed'])
                                            for c in counts.values())
            state = 'passed' if rc == 0 else ('timed out after %d min' % minutes if timed_out else 'FAILED (rc=%d)' % rc)
            print('=== %s on %s: run %d of %d %s in %ds%s%s ===' % (
                label, lane_id, number, total, state, seconds, tally, ', seed ' + seed if seed else ''), flush=True)
            if rc != 0:
                print('::error title=%s run %d of %d failed::%s on %s %s%s' % (
                    label, number, total, ' '.join(paths), lane_id, state,
                    ' with --test-randomize-ordering-seed=%s' % seed if seed else ''), flush=True)
                hook(lane.get('after_failure', []), cwd, env)
    stop()
    if previous is not None:
        signal.signal(signal.SIGTERM, previous)
    with open(os.path.join(work, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1)
    return summarize(lane_id, lane, entries, results, total)


def summarize(lane_id, lane, entries, results, total):
    lines = ['## %s' % lane_id, '',
             '| file | runs | failed | tests per run (passed ~skipped -failed) | mean | max |',
             '| --- | --- | --- | --- | --- | --- |']
    failed_runs = sum(1 for r in results if r['rc'] != 0)
    for entry in entries:
        path = entry['path']
        mine = [r for r in results if path in r['paths']]
        counts = [r['counts'][path] for r in mine if r['counts'] and path in r['counts']]
        failed = [r for r in mine if r['rc'] != 0]
        seconds = [r['seconds'] for r in mine]
        if counts:
            shapes = sorted(set('+%d ~%d -%d' % (c['passed'], c['skipped'], c['failed']) for c in counts))
            tests = ', '.join(shapes)
            if all(c['passed'] == 0 and c['failed'] == 0 for c in counts):
                if any(c['skipped'] for c in counts):
                    print('::warning title=Only skipped on %s::every test in %s skipped in every run on %s, '
                          'so this lane did not exercise it' % (lane_id, path, lane_id), flush=True)
                else:
                    print('::warning title=No tests on %s::%s reported no test on %s' % (lane_id, path, lane_id), flush=True)
            elif any(c['skipped'] for c in counts):
                names = sorted(set(n for c in counts for n in c['skipped_names']))
                print('::notice title=Skipped on %s::%s: %d of its tests skipped on %s: %s' % (
                    lane_id, path, len(names), lane_id, '; '.join(names)[:900]), flush=True)
        else:
            tests = 'no report' if lane.get('report', True) else 'not reported by this command'
        lines.append('| %s | %d | %d%s | %s | %ds | %ds |' % (
            path, len(mine), len(failed),
            ' (runs %s)' % ', '.join(str(r['run']) for r in failed) if failed else '',
            tests, sum(seconds) / max(len(seconds), 1), max(seconds or [0])))
    seeded = [r for r in results if r['rc'] != 0 and r['seed']]
    if seeded:
        lines += ['', 'Failed seeds: ' + ', '.join('run %d: %s' % (r['run'], r['seed']) for r in seeded)]
    verdict = '%s: %d of %d runs failed' % (lane_id, failed_runs, len(results)) if failed_runs else \
        '%s: all %d runs passed' % (lane_id, len(results))
    lines += ['', verdict]
    print('\n'.join(lines), flush=True)
    write_env_file('GITHUB_STEP_SUMMARY', '\n'.join(lines) + '\n')
    if failed_runs:
        print('::error title=%s::%s' % (lane_id, verdict), flush=True)
        return 1
    return 0


def cmd_run(_args):
    table = load_table(ROOT)
    try:
        runs = int(os.environ.get('TEST_REPEAT_RUNS', '0'))
        first = int(os.environ.get('TEST_REPEAT_FIRST', '1'))
        total = int(os.environ.get('TEST_REPEAT_TOTAL', '0')) or runs
    except ValueError:
        print('::error::TEST_REPEAT_RUNS, TEST_REPEAT_FIRST and TEST_REPEAT_TOTAL must be numbers', flush=True)
        return 2
    files = split_words(os.environ.get('TEST_REPEAT_FILES', ''))
    return run(ROOT, table, os.environ.get('TEST_REPEAT_LANE', ''), files, runs, first, total)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('plan')
    p.add_argument('--changes', help='output of git diff --name-status -M')
    sub.add_parser('run')
    args = parser.parse_args(argv)
    if args.command == 'plan':
        return cmd_plan(args)
    if args.command == 'run':
        return cmd_run(args)
    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
