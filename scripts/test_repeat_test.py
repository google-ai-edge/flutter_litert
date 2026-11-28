#!/usr/bin/env python3
"""Tests scripts/test_repeat.py and the lane table it reads.

  python3 -I scripts/test_repeat_test.py

The planner and runner run on small fixture repos made in a temporary
directory. The checks at the end read the real .github/test-lanes.json and
the workflows it mirrors, so a lane that drifts from its flutter-ci.yml job
fails here. test-repeat.yml runs this file 10 times when it changes.
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location('test_repeat', os.path.join(HERE, 'test_repeat.py'))
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)

LIMITS = {
    'default_runs': 10,
    'max_runs': 50,
    'budget_fraction': 0.6,
    'max_shards': 3,
    'max_jobs': {'pull_request': 8, 'workflow_dispatch': 16},
}


def fixture_table(**lanes):
    table = {
        'limits': json.loads(json.dumps(LIMITS)),
        'test_roots': ['test', 'example/integration_test', 'example/test'],
        'lanes': {
            'unit-linux': {
                'runner': 'ubuntu-24.04', 'timeout': 45, 'setup': {'flutter': True}, 'cwd': '.',
                'include': ['test/**/*_test.dart'], 'test_on': ['vm', 'linux', 'posix'],
                'batch': True, 'shuffle': True, 'command': ['flutter', 'test', '{files}'],
                'fixed_minutes': 4, 'iteration_minutes': 0.5, 'run_minutes': 0.2,
            },
            'unit-chrome': {
                'runner': 'ubuntu-24.04', 'timeout': 50, 'setup': {}, 'cwd': '.',
                'files': ['test/web_test.dart', 'test/shared_test.dart'],
                'test_on': ['browser', 'chrome', 'js'], 'batch': True,
                'command': ['flutter', 'test', '--platform', 'chrome', '{files}'],
                'fixed_minutes': 3, 'iteration_minutes': 1, 'run_minutes': 0.3,
            },
            'desktop-linux': {
                'runner': 'ubuntu-24.04', 'timeout': 45, 'setup': {}, 'cwd': 'example',
                'files': ['example/integration_test/app_test.dart'],
                'deps': ['example/test_driver/**'],
                'command': ['flutter', 'test', '{file}', '-d', 'linux'],
                'fixed_minutes': 6, 'run_minutes': 1,
            },
            'device-manual': {
                'auto': False, 'runner': 'macos-15', 'timeout': 90, 'setup': {}, 'cwd': 'example',
                'files': ['example/integration_test/app_test.dart', 'example/integration_test/device_test.dart'],
                'command': ['flutter', 'test', '{file}'],
                'fixed_minutes': 18, 'run_minutes': 2,
            },
        },
    }
    table['lanes'].update(lanes)
    return table


FILES = {
    'test/shared_test.dart': "import 'helpers/util.dart';\nvoid main() {}\n",
    'test/linux_only_test.dart': "@TestOn('linux')\nlibrary;\nvoid main() {}\n",
    'test/web_test.dart': "@TestOn('browser')\nlibrary;\nvoid main() {}\n",
    'test/mac_test.dart': "@TestOn('mac-os || windows')\nlibrary;\nvoid main() {}\n",
    'test/deep/nested_test.dart': "import '../helpers/wrap.dart';\nvoid main() {}\n",
    'test/helpers/util.dart': "int util() => 1;\n",
    'test/helpers/wrap.dart': "export 'util.dart';\n",
    'test/helpers/unused.dart': "int unused() => 1;\n",
    'test/native/_impl.dart': "void main() {}\n",
    'test/native/runner_test.dart': "const impl = 'test/native/_impl.dart';\nvoid main() {}\n",
    'test/notes.txt': 'not a test\n',
    'example/integration_test/app_test.dart': "void main() {}\n",
    'example/integration_test/device_test.dart': "void main() {}\n",
    'example/test/orphan_test.dart': "void main() {}\n",
    'example/test_driver/driver.dart': "void main() {}\n",
    'README.md': 'readme\n',
}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='test_repeat_')
        self.addCleanup(shutil.rmtree, self.root, True)
        for path, text in FILES.items():
            full = os.path.join(self.root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, 'w', encoding='utf-8') as f:
                f.write(text)

    def pr(self, changes, table=None):
        return tr.plan(self.root, table or fixture_table(), 'pull_request', changes=changes)

    def dispatch(self, files='', lanes='', runs='', table=None):
        return tr.plan(self.root, table or fixture_table(), 'workflow_dispatch', files=files, lanes=lanes, runs=runs)


def lanes_of(jobs):
    return sorted(set(j['lane'] for j in jobs))


def files_of(jobs, lane):
    return sorted(set(f for j in jobs if j['lane'] == lane for f in j['files'].split()))


def warnings(notes):
    return [m for level, m in notes if level == 'warning']


class GlobAndTestOn(unittest.TestCase):
    def test_globs(self):
        self.assertTrue(tr.matches('test/a_test.dart', ['test/**/*_test.dart']))
        self.assertTrue(tr.matches('test/x/y/a_test.dart', ['test/**/*_test.dart']))
        self.assertFalse(tr.matches('test/a.dart', ['test/**/*_test.dart']))
        self.assertFalse(tr.matches('example/test/a_test.dart', ['test/**/*_test.dart']))
        self.assertTrue(tr.matches('example_web/test_driver/x/y.dart', ['example_web/test_driver/**']))
        self.assertFalse(tr.matches('test/x/a_test.dart', ['test/*_test.dart']))

    def test_platform_selectors(self):
        vm = {'vm', 'linux', 'posix'}
        self.assertTrue(tr.platform_matches('linux', vm))
        self.assertFalse(tr.platform_matches('browser', vm))
        self.assertTrue(tr.platform_matches('mac-os || linux || windows', vm))
        self.assertFalse(tr.platform_matches('mac-os || windows', vm))
        self.assertTrue(tr.platform_matches('vm && !windows', vm))
        self.assertFalse(tr.platform_matches('!(linux || mac-os)', vm))
        with self.assertRaises(ValueError):
            tr.platform_matches('linux ||', vm)

    def test_test_on_is_read_from_the_file_head(self):
        self.assertEqual(tr.test_on("// c\n@TestOn('mac-os')\nlibrary;\n"), 'mac-os')
        self.assertEqual(tr.test_on('@TestOn("browser")\n'), 'browser')
        self.assertIsNone(tr.test_on("void main() { print('@TestOn'); }\n"))

    def test_parse_changes(self):
        changes = 'M\ttest/a_test.dart\nA\ttest/b_test.dart\nD\ttest/gone_test.dart\n' \
                  'R087\ttest/old_test.dart\ttest/new_test.dart\nC100\tx.dart\ty.dart\n\n'
        self.assertEqual(tr.parse_changes(changes),
                         ['test/a_test.dart', 'test/b_test.dart', 'test/new_test.dart', 'y.dart'])


class PullRequestPlans(Fixture):
    def test_no_test_change_plans_no_job(self):
        jobs, notes = self.pr('M\tREADME.md\nM\ttest/notes.txt\nD\ttest/shared_test.dart\n')
        self.assertEqual(jobs, [])
        self.assertEqual(warnings(notes), [])

    def test_changed_test_runs_on_every_lane_that_runs_it(self):
        jobs, _ = self.pr('M\ttest/shared_test.dart\n')
        self.assertEqual(lanes_of(jobs), ['unit-chrome', 'unit-linux'])
        self.assertEqual(files_of(jobs, 'unit-linux'), ['test/shared_test.dart'])
        self.assertTrue(all(j['runs'] == 10 and j['total'] == 10 and j['first'] == 1 for j in jobs))

    def test_test_on_leaves_out_lanes_it_does_not_run_on(self):
        jobs, notes = self.pr('M\ttest/linux_only_test.dart\nA\ttest/web_test.dart\n')
        self.assertEqual(files_of(jobs, 'unit-linux'), ['test/linux_only_test.dart'])
        self.assertEqual(files_of(jobs, 'unit-chrome'), ['test/web_test.dart'])
        self.assertEqual(warnings(notes), [])

    def test_test_no_lane_runs_warns(self):
        jobs, notes = self.pr('M\ttest/mac_test.dart\nA\texample/test/orphan_test.dart\n')
        self.assertEqual(jobs, [])
        self.assertEqual(len(warnings(notes)), 2)
        self.assertTrue(any('orphan_test.dart (changed): no lane runs it' in w for w in warnings(notes)))
        self.assertTrue(any('test/mac_test.dart' in w for w in warnings(notes)))

    def test_changed_helper_adds_its_importers(self):
        jobs, notes = self.pr('M\ttest/helpers/util.dart\n')
        self.assertEqual(files_of(jobs, 'unit-linux'), ['test/deep/nested_test.dart', 'test/shared_test.dart'])
        self.assertEqual(files_of(jobs, 'unit-chrome'), ['test/shared_test.dart'])
        self.assertEqual(warnings(notes), [])

    def test_helper_named_by_path_adds_the_test_that_names_it(self):
        jobs, _ = self.pr('M\ttest/native/_impl.dart\n')
        self.assertEqual(files_of(jobs, 'unit-linux'), ['test/native/runner_test.dart'])

    def test_unused_helper_plans_nothing(self):
        jobs, notes = self.pr('M\ttest/helpers/unused.dart\n')
        self.assertEqual(jobs, [])
        self.assertTrue(any('no test file uses it' in m for _, m in notes))

    def test_dependency_change_reruns_the_whole_lane(self):
        jobs, _ = self.pr('M\texample/test_driver/driver.dart\n')
        self.assertEqual(lanes_of(jobs), ['desktop-linux'])
        self.assertEqual(files_of(jobs, 'desktop-linux'), ['example/integration_test/app_test.dart'])

    def test_dispatch_only_lane_is_named_not_run(self):
        jobs, notes = self.pr('M\texample/integration_test/device_test.dart\n')
        self.assertEqual(jobs, [])
        self.assertTrue(any('device-manual' in w and 'by hand' in w for w in warnings(notes)))

    def test_dispatch_only_lane_does_not_add_to_an_auto_lane(self):
        jobs, notes = self.pr('M\texample/integration_test/app_test.dart\n')
        self.assertEqual(lanes_of(jobs), ['desktop-linux'])
        self.assertEqual(warnings(notes), [])

    def test_pull_request_ignores_dispatch_inputs(self):
        jobs, _ = tr.plan(self.root, fixture_table(), 'pull_request', changes='M\ttest/shared_test.dart\n',
                          files='example/integration_test/device_test.dart', lanes='device-manual')
        self.assertEqual(lanes_of(jobs), ['unit-chrome', 'unit-linux'])

    def test_too_many_jobs_fails_and_says_to_dispatch(self):
        table = fixture_table()
        table['limits']['max_jobs']['pull_request'] = 1
        with self.assertRaises(tr.PlanError) as e:
            self.pr('M\ttest/shared_test.dart\n', table)
        self.assertIn('by hand', str(e.exception))


class Sharding(Fixture):
    def test_runs_split_over_the_fewest_shards_within_budget(self):
        table = fixture_table()
        # 6 + 10 * 2 = 26 > 0.6 * 30 = 18; 6 + 5 * 2 = 16 fits.
        table['lanes']['desktop-linux'].update({'timeout': 30, 'run_minutes': 2})
        jobs, _ = self.pr('M\texample/integration_test/app_test.dart\n', table)
        self.assertEqual([(j['runs'], j['first'], j['total']) for j in jobs], [(5, 1, 10), (5, 6, 10)])
        self.assertEqual(len(set(j['id'] for j in jobs)), 2)
        self.assertEqual(sum(j['runs'] for j in jobs), 10)

    def test_uneven_split_still_adds_up(self):
        table = fixture_table()
        # 6 + ceil(10 / 3) * 1 = 10 <= 0.6 * 17 = 10.2, while 2 shards need 11.
        table['lanes']['desktop-linux'].update({'timeout': 17})
        jobs, _ = self.pr('M\texample/integration_test/app_test.dart\n', table)
        self.assertEqual([(j['runs'], j['first']) for j in jobs], [(4, 1), (3, 5), (3, 8)])

    def test_over_three_shards_fails(self):
        table = fixture_table()
        table['lanes']['desktop-linux'].update({'timeout': 12})
        with self.assertRaises(tr.PlanError) as e:
            self.pr('M\texample/integration_test/app_test.dart\n', table)
        self.assertIn('desktop-linux', str(e.exception))

    def test_more_files_need_more_shards(self):
        table = fixture_table()
        # One file: 4 + 10 * (0.5 + 0.2) = 11 fits 27; 50 runs of it need
        # 4 + 50 * 0.7 = 39, so 2 shards: 4 + 25 * 0.7 = 21.5.
        jobs, _ = self.dispatch(files='test/shared_test.dart', lanes='unit-linux', runs='50', table=table)
        self.assertEqual([j['runs'] for j in jobs], [25, 25])

    def test_each_shard_pays_the_build_again(self):
        table = fixture_table()
        # 6 + 3 + 10 * 1 = 19 > 0.6 * 30 = 18; 6 + 3 + 5 * 1 = 14 fits. Without
        # the build, one job of 16 would fit.
        table['lanes']['desktop-linux'].update({'timeout': 30, 'build_minutes': 3})
        jobs, _ = self.pr('M\texample/integration_test/app_test.dart\n', table)
        self.assertEqual([(j['runs'], j['estimate']) for j in jobs], [(5, 14), (5, 14)])

    def test_the_cold_build_counts_once_per_job(self):
        lane = dict(fixture_table()['lanes']['desktop-linux'], build_minutes=2)
        entries = [{'path': 'a'}, {'path': 'b'}]
        self.assertEqual(tr.build_minutes(lane, entries), 2)
        # A file that builds the app another way adds its own build.
        self.assertEqual(tr.build_minutes(lane, entries + [{'path': 'c', 'build_minutes': 5}]), 7)

    def test_an_emulator_step_limit_splits_the_runs_too(self):
        table = fixture_table()
        # The step gets (45 - 3) // 2 = 21 minutes. The job needs
        # 3 + 4 + 10 * 1 = 17 <= 27, but its step 4 + 10 * 1 = 14 > 0.6 * 21
        # = 12.6; 2 shards: 4 + 5 * 1 = 9 fits.
        table['lanes']['desktop-linux'].update({
            'setup': {'android': {'api': 24, 'step_limit': True}}, 'fixed_minutes': 3, 'build_minutes': 4})
        jobs, _ = self.pr('M\texample/integration_test/app_test.dart\n', table)
        self.assertEqual([(j['runs'], j['emu_timeout'], j['timeout']) for j in jobs], [(5, 21, 45), (5, 21, 45)])
        del table['lanes']['desktop-linux']['setup']['android']['step_limit']
        jobs, _ = self.pr('M\texample/integration_test/app_test.dart\n', table)
        self.assertEqual([(j['runs'], j['emu_timeout']) for j in jobs], [(10, 45)])

    def test_an_emulator_step_over_its_limit_fails(self):
        table = fixture_table()
        # One run's step is 4 + 9 = 13 > 12.6, in any number of shards.
        table['lanes']['desktop-linux'].update({
            'setup': {'android': {'api': 24, 'step_limit': True}}, 'fixed_minutes': 3, 'build_minutes': 4,
            'run_minutes': 9})
        with self.assertRaises(tr.PlanError) as e:
            self.dispatch(files='example/integration_test/app_test.dart', lanes='desktop-linux', runs='3', table=table)
        self.assertIn('step limit', str(e.exception))

    def test_runs_come_from_the_input(self):
        jobs, _ = self.dispatch(files='test/shared_test.dart', lanes='unit-linux', runs='3')
        self.assertEqual([(j['runs'], j['total']) for j in jobs], [(3, 3)])


class DispatchPlans(Fixture):
    def test_files_go_to_auto_lanes_and_name_manual_ones(self):
        jobs, notes = self.dispatch(files='example/integration_test/device_test.dart')
        self.assertEqual(jobs, [])
        self.assertTrue(any('device-manual' in w for w in warnings(notes)))

    def test_named_lane_runs_a_manual_lane(self):
        jobs, notes = self.dispatch(files='example/integration_test/device_test.dart', lanes='device-manual')
        self.assertEqual(lanes_of(jobs), ['device-manual'])
        self.assertEqual(warnings(notes), [])

    def test_lane_without_files_runs_all_of_them(self):
        jobs, _ = self.dispatch(lanes='device-manual')
        self.assertEqual(files_of(jobs, 'device-manual'),
                         ['example/integration_test/app_test.dart', 'example/integration_test/device_test.dart'])

    def test_lane_without_files_follows_include_and_test_on(self):
        jobs, _ = self.dispatch(lanes='unit-linux')
        self.assertEqual(files_of(jobs, 'unit-linux'), [
            'test/deep/nested_test.dart', 'test/linux_only_test.dart',
            'test/native/runner_test.dart', 'test/shared_test.dart'])

    def test_file_no_lane_runs_warns(self):
        jobs, notes = self.dispatch(files='example/test/orphan_test.dart')
        self.assertEqual(jobs, [])
        self.assertTrue(any('orphan_test.dart (named in files): no lane runs it' in w for w in warnings(notes)))

    def test_named_lane_that_does_not_run_the_file_warns(self):
        jobs, notes = self.dispatch(files='test/web_test.dart', lanes='unit-linux')
        self.assertEqual(jobs, [])
        self.assertTrue(warnings(notes))

    def test_helper_files_are_followed_by_dispatch_too(self):
        jobs, _ = self.dispatch(files='test/helpers/wrap.dart', lanes='unit-linux')
        self.assertEqual(files_of(jobs, 'unit-linux'), ['test/deep/nested_test.dart'])

    def test_inputs_are_checked(self):
        for files, lanes, runs in [
                ('../etc/passwd', '', ''), ('test/$(id)_test.dart', '', ''), ('/abs_test.dart', '', ''),
                ('test/missing_test.dart', '', ''), ('test/shared_test.dart', 'no-such-lane', ''),
                ('test/shared_test.dart', '', '0'), ('test/shared_test.dart', '', '51'),
                ('test/shared_test.dart', '', 'ten'), ('', '', '')]:
            with self.assertRaises(tr.PlanError, msg=(files, lanes, runs)):
                self.dispatch(files=files, lanes=lanes, runs=runs)

    def test_space_and_comma_separated_inputs(self):
        jobs, _ = self.dispatch(files=' test/shared_test.dart,test/linux_only_test.dart ', lanes='unit-linux')
        self.assertEqual(files_of(jobs, 'unit-linux'), ['test/linux_only_test.dart', 'test/shared_test.dart'])


FAKE_TEST = r'''
import json, os, sys, time
path, report, mode, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
counter = os.environ['FAKE_COUNTER']
n = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
open(counter, 'w').write(str(n))
with open(os.environ['FAKE_SEEDS'], 'a') as f:
    f.write(seed + '\n')
if mode == 'hang':
    time.sleep(120)
fail = mode == 'fail' or (mode == 'flaky' and n % 3 == 2)
skip = mode == 'skip'
events = [
    {'type': 'suite', 'suite': {'id': 0, 'path': os.path.abspath(path)}},
    {'type': 'testStart', 'test': {'id': 1, 'suiteID': 0, 'name': 'loading'}},
    {'type': 'testDone', 'testID': 1, 'result': 'success', 'skipped': False, 'hidden': True},
    {'type': 'testStart', 'test': {'id': 2, 'suiteID': 0, 'name': 'adds'}},
    {'type': 'testDone', 'testID': 2, 'result': 'success', 'skipped': skip, 'hidden': False},
    {'type': 'testStart', 'test': {'id': 3, 'suiteID': 0, 'name': 'divides'}},
    {'type': 'testDone', 'testID': 3, 'result': 'failure' if fail else 'success', 'skipped': skip, 'hidden': False},
]
with open(report, 'w') as f:
    f.write('\n'.join(json.dumps(e) for e in events) + '\n')
sys.exit(1 if fail else 0)
'''


class Runs(Fixture):
    def setUp(self):
        super().setUp()
        self.fake = os.path.join(self.root, 'fake_test.py')
        with open(self.fake, 'w', encoding='utf-8') as f:
            f.write(FAKE_TEST)
        self.counter = os.path.join(self.root, 'counter')
        self.seeds = os.path.join(self.root, 'seeds')
        self.temp = os.path.join(self.root, 'temp')
        patcher = mock.patch.dict(os.environ, {'FAKE_COUNTER': self.counter, 'FAKE_SEEDS': self.seeds})
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ('GITHUB_STEP_SUMMARY', 'GITHUB_OUTPUT'):
            os.environ.pop(name, None)

    def lane(self, mode, **extra):
        lane = {
            'runner': 'ubuntu-24.04', 'timeout': 45, 'setup': {}, 'cwd': 'example',
            'files': [{'path': 'example/integration_test/app_test.dart', 'args': [mode]}],
            'shuffle': True,
            'command': ['{python}', '-I', self.fake, '{file}', '{report}', '{args}', 's{seed}'],
            'fixed_minutes': 1, 'run_minutes': 1,
        }
        lane.update(extra)
        return fixture_table(fake=lane)

    def go(self, mode, runs=10, first=1, total=None, **extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = tr.run(self.root, self.lane(mode, **extra), 'fake',
                        ['example/integration_test/app_test.dart'], runs, first, total or runs, self.temp)
        return rc, out.getvalue()

    def count(self):
        with open(self.counter) as f:
            return int(f.read())

    def results(self):
        with open(os.path.join(self.temp, 'test-repeat', 'fake', 'results.json')) as f:
            return json.load(f)

    def test_all_green(self):
        rc, out = self.go('pass')
        self.assertEqual(rc, 0)
        self.assertEqual(self.count(), 10)
        self.assertIn('all 10 runs passed', out)
        self.assertNotIn('::error', out)
        self.assertEqual([r['counts']['example/integration_test/app_test.dart']['passed'] for r in self.results()], [2] * 10)

    def test_each_run_gets_its_own_printed_seed(self):
        rc, out = self.go('pass')
        with open(self.seeds) as f:
            seeds = [line.strip()[1:] for line in f if line.strip()]
        self.assertEqual(len(seeds), 10)
        self.assertTrue(all(s.isdigit() for s in seeds))
        self.assertGreater(len(set(seeds)), 1)
        for s in seeds:
            self.assertIn('seed %s' % s, out)

    def test_a_failure_fails_the_job_and_is_never_retried(self):
        rc, out = self.go('fail', runs=4)
        self.assertEqual(rc, 1)
        self.assertEqual(self.count(), 4)
        self.assertIn('4 of 4 runs failed', out)
        self.assertEqual(out.count('::error title=example/integration_test/app_test.dart run'), 4)

    def test_a_flaky_test_is_caught(self):
        rc, out = self.go('flaky', runs=10)
        self.assertEqual(rc, 1)
        self.assertEqual(self.count(), 10)
        self.assertIn('3 of 10 runs failed', out)
        failed = [r['run'] for r in self.results() if r['rc']]
        self.assertEqual(failed, [2, 5, 8])
        self.assertIn('--test-randomize-ordering-seed=', out)

    def test_failed_test_in_report_fails_the_run_even_on_exit_zero(self):
        fake_zero = self.fake.replace('fake_test', 'fake_zero')
        with open(self.fake) as f, open(fake_zero, 'w') as g:
            g.write(f.read().replace('sys.exit(1 if fail else 0)', 'sys.exit(0)'))
        rc, out = self.go('fail', runs=2,
                          command=['{python}', '-I', fake_zero, '{file}', '{report}', '{args}', 's{seed}'])
        self.assertEqual(rc, 1)
        self.assertIn('2 of 2 runs failed', out)

    def test_only_skipped_warns(self):
        rc, out = self.go('skip', runs=3)
        self.assertEqual(rc, 0)
        self.assertIn('::warning title=Only skipped on fake::', out)

    def test_shard_numbers_its_runs(self):
        rc, out = self.go('pass', runs=3, first=8, total=10)
        self.assertEqual(rc, 0)
        self.assertEqual([r['run'] for r in self.results()], [8, 9, 10])
        self.assertIn('run 10 of 10', out)

    def test_a_hung_run_is_ended_and_counted_as_failed(self):
        started = time.time()
        rc, out = self.go('hang', runs=2, run_timeout=0.05)
        self.assertLess(time.time() - started, 60)
        self.assertEqual(rc, 1)
        self.assertIn('timed out', out)
        self.assertEqual([r['timed_out'] for r in self.results()], [True, True])

    def test_hooks_env_and_services(self):
        marker = os.path.join(self.root, 'hooks')
        service = os.path.join(self.root, 'service.py')
        with open(service, 'w') as f:
            f.write('import time\nopen(%r, "a").write("service\\n")\ntime.sleep(120)\n' % marker)
        rc, out = self.go(
            'flaky', runs=3, shuffle=False,
            env={'HOOK_LOG': marker, 'LOGDIR': '{logdir}'},
            services=[['{python}', service]],
            # Waits for the service, so the check below cannot race its start.
            prepare=['for i in $(seq 1 300); do grep -q service "$HOOK_LOG" 2>/dev/null && break; sleep 0.1; done',
                     'echo prepare >> "$HOOK_LOG"'],
            after_run=[['{python}', '-c', 'import os; open(os.environ["HOOK_LOG"], "a").write("after\\n")']],
            after_failure=[[sys.executable, '-c', 'import os; open(os.environ["HOOK_LOG"], "a").write("failure\\n")']])
        self.assertEqual(rc, 1)
        with open(marker) as f:
            lines = f.read().split()
        self.assertEqual(lines.count('prepare'), 1)
        self.assertEqual(lines.count('after'), 3)
        self.assertEqual(lines.count('failure'), 1)
        self.assertEqual(lines.count('service'), 1)
        with open(self.seeds) as f:
            self.assertEqual([line.strip() for line in f], ['s'] * 3)

    def test_env_root_is_the_checkout_not_the_workspace(self):
        marker = os.path.join(self.root, 'root_seen')
        with mock.patch.dict(os.environ, {'GITHUB_WORKSPACE': os.path.join(self.root, 'elsewhere')}):
            rc, _ = self.go('pass', runs=1, env={'ROOT_SEEN': '{root}/scripts', 'MARKER': marker},
                            after_run=[['{python}', '-c',
                                        'import os; open(os.environ["MARKER"], "w").write(os.environ["ROOT_SEEN"])']])
        self.assertEqual(rc, 0)
        with open(marker) as f:
            self.assertEqual(f.read(), self.root + '/scripts')

    def test_failed_prepare_runs_nothing(self):
        rc, out = self.go('pass', runs=3, prepare=['exit 3'])
        self.assertEqual(rc, 1)
        self.assertFalse(os.path.exists(self.counter))
        self.assertIn('preparing the lane failed', out)

    def test_file_the_lane_does_not_run_is_refused(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = tr.run(self.root, self.lane('pass'), 'fake', ['test/shared_test.dart'], 1, 1, 1, self.temp)
        self.assertEqual(rc, 2)

    def test_report_parsing_maps_suites_to_files(self):
        report = os.path.join(self.root, 'r.json')
        events = [
            {'type': 'suite', 'suite': {'id': 0, 'path': '/w/repo/test/a_test.dart'}},
            {'type': 'suite', 'suite': {'id': 1, 'path': 'C:\\w\\repo\\test\\b_test.dart'}},
            {'type': 'testStart', 'test': {'id': 5, 'suiteID': 0, 'name': 'x'}},
            {'type': 'testDone', 'testID': 5, 'result': 'success', 'skipped': False, 'hidden': False},
            {'type': 'testStart', 'test': {'id': 6, 'suiteID': 1, 'name': 'y'}},
            {'type': 'testDone', 'testID': 6, 'result': 'success', 'skipped': True, 'hidden': False},
            {'type': 'testStart', 'test': {'id': 7, 'suiteID': 1, 'name': 'z'}},
            {'type': 'testDone', 'testID': 7, 'result': 'error', 'skipped': False, 'hidden': False},
        ]
        with open(report, 'w') as f:
            f.write('not json\n' + '\n'.join(json.dumps(e) for e in events))
        counts = tr.parse_report(report, ['test/a_test.dart', 'test/b_test.dart'])
        self.assertEqual(counts['test/a_test.dart']['passed'], 1)
        self.assertEqual((counts['test/b_test.dart']['skipped'], counts['test/b_test.dart']['failed']), (1, 1))
        self.assertIsNone(tr.parse_report(os.path.join(self.root, 'none.json'), ['x']))


# --- the real table ----------------------------------------------------------

def workflow_jobs(name):
    with open(os.path.join(ROOT, '.github', 'workflows', name), encoding='utf-8') as f:
        text = f.read()
    jobs = {}
    for m in re.finditer(r'^  ([A-Za-z0-9_-]+):\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)', text.split('\njobs:\n', 1)[1], re.M | re.S):
        jobs[m.group(1)] = m.group(2)
    return jobs


class RealTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = tr.load_table(ROOT)
        cls.lanes = cls.table['lanes']

    def test_limits_are_sane(self):
        limits = self.table['limits']
        self.assertEqual(limits['default_runs'], 10)
        self.assertEqual(limits['max_shards'], 3)
        self.assertLessEqual(limits['max_jobs']['pull_request'], 8)

    def test_every_lane_is_complete(self):
        known = {'auto', 'mirrors', 'mirror_directory', 'runner', 'timeout', 'setup', 'cwd', 'files', 'include',
                 'test_on', 'deps', 'batch', 'shuffle', 'report', 'services', 'response_file', 'env', 'prepare',
                 'after_run', 'after_failure', 'command', 'default_args', 'run_timeout', 'first_run_timeout',
                 'fixed_minutes', 'build_minutes', 'iteration_minutes', 'run_minutes'}
        setup_keys = {'flutter', 'desktop', 'apt', 'pub_get', 'flex_cache', 'android', 'xcode', 'ios_simulator', 'free_disk',
                      'hooks_cache'}
        for lane_id, lane in self.lanes.items():
            with self.subTest(lane=lane_id):
                self.assertRegex(lane_id, tr.LANE_RE)
                self.assertLessEqual(set(lane), known)
                self.assertLessEqual(set(lane['setup']), setup_keys)
                self.assertTrue(lane.get('files') or lane.get('include'))
                self.assertTrue(os.path.isdir(os.path.join(ROOT, lane['cwd'])))
                for directory in lane['setup'].get('pub_get', []):
                    self.assertTrue(os.path.isfile(os.path.join(ROOT, directory, 'pubspec.yaml')), directory)
                self.assertIn(lane['setup'].get('flex_cache', 'macos'), ('macos', 'linux', 'windows'))
                self.assertIn(lane['setup'].get('hooks_cache', 'macos'), ('macos',))
                self.assertIn('{report}' in ' '.join(lane['command']), (lane.get('report', True),))
                if lane.get('batch'):
                    self.assertIn('{files}', lane['command'])
                else:
                    self.assertIn('{file}', ' '.join(lane['command']))
                if lane.get('shuffle'):
                    self.assertIn('{seed}', ' '.join(lane['command']))
                for key in ('fixed_minutes', 'run_minutes', 'timeout'):
                    self.assertGreater(lane[key], 0)
                if lane['setup'].get('android'):
                    self.assertEqual(lane['env']['ANDROID_TEST_LAUNCH_ATTEMPTS'], '2')
                    self.assertEqual(lane['env']['ANDROID_TEST_ATTEMPTS'], '1')

    def test_explicit_files_exist_and_are_under_a_test_root(self):
        for lane_id, lane in self.lanes.items():
            for item in lane.get('files', []):
                path = tr.entry_of(item)['path']
                with self.subTest(lane=lane_id, path=path):
                    self.assertTrue(os.path.isfile(os.path.join(ROOT, path)))
                    if lane_id != 'repeat-plan':
                        self.assertTrue(tr.under(path, self.table['test_roots']))
            for pattern in lane.get('include', []):
                with self.subTest(lane=lane_id, pattern=pattern):
                    self.assertTrue(tr.lane_all_files(ROOT, lane), pattern)

    def test_device_lanes_are_dispatch_only(self):
        for lane_id in ('ios-simulator', 'android-play-services', 'android-api24'):
            self.assertIs(self.lanes[lane_id]['auto'], False, lane_id)

    def test_lanes_match_the_jobs_they_mirror(self):
        jobs = {name: workflow_jobs(name) for name in ('flutter-ci.yml', 'device-tests-manual.yml')}
        for lane_id, lane in self.lanes.items():
            if 'mirrors' not in lane:
                continue
            with self.subTest(lane=lane_id):
                workflow, job = lane['mirrors'].split(':')
                self.assertIn(job, jobs[workflow], 'the job %s mirrors is gone' % lane_id)
                block = jobs[workflow][job]
                self.assertRegex(block, r'(?m)^    runs-on: %s$' % re.escape(lane['runner']))
                if lane.get('mirror_directory'):
                    self.assertIn(lane['mirror_directory'], block)
                for item in lane.get('files', []):
                    path = tr.entry_of(item)['path']
                    rel = os.path.relpath(path, lane['cwd']).replace(os.sep, '/')
                    stem = os.path.basename(path)[:-len('.dart')]
                    self.assertTrue(rel in block or re.search(r'\b%s\b' % re.escape(stem), block),
                                    '%s no longer runs %s' % (lane['mirrors'], rel))
                for directory in lane['setup'].get('pub_get', []):
                    if directory != '.':
                        self.assertIn(directory, block)
                android = lane['setup'].get('android')
                if android:
                    self.assertRegex(block, r'api-level: %s\b' % android['api'])
                    for package in android['packages'].split():
                        self.assertIn(package, block)
                    # A lane gets the emulator step limit where its job has one.
                    self.assertEqual(bool(android.get('step_limit')),
                                     bool(re.search(r'(?m)^        timeout-minutes:', block)))
                    # The lane restores the Gradle download entry its job uses.
                    self.assertRegex(block, r'(?m)^          downloads: %s$' % re.escape(android['downloads']))
                if lane['setup'].get('hooks_cache'):
                    self.assertIn('example-hooks-%s-' % lane['setup']['hooks_cache'], block)
                    self.assertIn('example', lane['setup']['pub_get'])

    def test_hooks_cache_restore_matches_build_macos(self):
        def keys(name):
            with open(os.path.join(ROOT, '.github', 'workflows', name), encoding='utf-8') as f:
                return re.findall(r'(?m)^ +key: (example-hooks-macos-.*)$', f.read())
        self.assertEqual(keys('test-repeat.yml'), keys('flutter-ci.yml'))
        self.assertEqual(len(keys('flutter-ci.yml')), 1)
        with open(os.path.join(ROOT, '.github', 'workflows', 'test-repeat.yml'), encoding='utf-8') as f:
            self.assertNotIn('actions/cache/save', f.read())

    def test_flex_caches_are_only_restored(self):
        def keys(name):
            with open(os.path.join(ROOT, '.github', 'workflows', name), encoding='utf-8') as f:
                return re.findall(r'(?m)^ +key: (flex-delegate-.*)$', f.read())
        self.assertEqual(keys('test-repeat.yml'), keys('flutter-ci.yml'))
        self.assertEqual(len(keys('flutter-ci.yml')), 3)
        with open(os.path.join(ROOT, '.github', 'workflows', 'test-repeat.yml'), encoding='utf-8') as f:
            self.assertNotIn('actions/cache@', f.read())

    def test_flutter_is_set_up_as_in_flutter_ci(self):
        for name in ('flutter-ci.yml', 'test-repeat.yml'):
            with open(os.path.join(ROOT, '.github', 'workflows', name), encoding='utf-8') as f:
                text = f.read()
            self.assertNotIn('subosito/flutter-action', text, name)
            self.assertIn('uses: ./.github/actions/flutter\n', text, name)
            # flutter-ci.yml saves the SDK and pub caches, test-repeat.yml only restores them.
            self.assertEqual("restore-only: 'true'" in text, name == 'test-repeat.yml', name)

    def test_play_services_lane_builds_as_its_job(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'device-tests-manual.yml'), encoding='utf-8') as f:
            job = f.read()
        build = self.lanes['android-play-services']['prepare'][0]
        for line in ("grep -q 'preparing SDK package NDK'", 'elif ! bash ../../scripts/gradle_not_found.sh'):
            self.assertIn(line, build)
            self.assertIn(line.replace("'", '"'), job)

    def test_gradle_downloads_are_only_restored(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'test-repeat.yml'), encoding='utf-8') as f:
            text = f.read()
        self.assertNotIn('cache: gradle', text)
        self.assertIn('downloads: ${{ matrix.gradle_downloads }}', text)
        self.assertNotRegex(text, r'(?m)^ +save:')
        jobs, _ = tr.plan(ROOT, self.table, 'workflow_dispatch', lanes='android-host,android-api24')
        self.assertEqual({j['lane']: j['gradle_downloads'] for j in jobs},
                         {'android-host': 'play-services-test-host', 'android-api24': 'example'})

    def test_api24_lane_fits_its_step_limit(self):
        lane = self.lanes['android-api24']
        self.assertEqual(tr.step_limit(lane), 21)
        jobs, _ = tr.plan(ROOT, self.table, 'workflow_dispatch', lanes='android-api24')
        self.assertEqual([j['runs'] for j in jobs], [4, 3, 3])
        self.assertTrue(all(j['emu_timeout'] == 21 and j['timeout'] == 45 for j in jobs))

    def test_workflow_filter_covers_every_trigger(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'test-repeat.yml'), encoding='utf-8') as f:
            text = f.read()
        paths = re.search(r'(?ms)^    paths:\n(.*?)^  \S', text).group(1)
        patterns = re.findall(r"^      - '([^']+)'$", paths, re.M)
        wanted = [root + '/x_test.dart' for root in self.table['test_roots']]
        for lane in self.lanes.values():
            wanted += [tr.entry_of(item)['path'] for item in lane.get('files', [])]
            wanted += [d.replace('**', 'x/y.dart') for d in lane.get('deps', [])]
        for path in wanted:
            self.assertTrue(tr.matches(path, patterns), '%s is not in the paths filter' % path)
        self.assertFalse(tr.matches('README.md', patterns))
        self.assertFalse(tr.matches('lib/flutter_litert.dart', patterns))

    def test_the_brief_pull_request(self):
        changes = 'M\ttest/gpu_options_payload_test.dart\nM\texample/integration_test/macos_gpu_cache_test.dart\n'
        jobs, notes = tr.plan(ROOT, self.table, 'pull_request', changes=changes)
        self.assertEqual(lanes_of(jobs), ['desktop-macos', 'unit-linux', 'unit-macos'])
        self.assertEqual(files_of(jobs, 'desktop-macos'), ['example/integration_test/macos_gpu_cache_test.dart'])
        self.assertEqual(warnings(notes), [])

    def test_example_and_flex_host_unit_tests_run_on_build_linux_lanes(self):
        changes = ('M\ttest/gpu_options_payload_test.dart\n'
                   'M\texample/test/merge_android_model_matrix_results_test.dart\n'
                   'M\texample/flex_test_host/test/widget_test.dart\n')
        jobs, notes = tr.plan(ROOT, self.table, 'pull_request', changes=changes)
        self.assertEqual(lanes_of(jobs), ['example-unit-linux', 'flex-unit-linux', 'unit-linux', 'unit-macos'])
        self.assertEqual(files_of(jobs, 'example-unit-linux'),
                         ['example/test/merge_android_model_matrix_results_test.dart'])
        self.assertEqual(warnings(notes), [])
        # A changed tool reruns the tests that import it.
        jobs, _ = tr.plan(ROOT, self.table, 'pull_request', changes='M\texample/tool/android_model_matrix_manifest.dart\n')
        self.assertEqual(files_of(jobs, 'example-unit-linux'), [
            'example/test/android_model_matrix_manifest_test.dart',
            'example/test/merge_android_model_matrix_results_test.dart'])

    def test_every_test_flutter_ci_runs_explicitly_is_in_a_lane(self):
        with open(os.path.join(ROOT, '.github', 'workflows', 'flutter-ci.yml'), encoding='utf-8') as f:
            text = f.read()
        covered = set()
        for lane in self.lanes.values():
            covered |= set(os.path.basename(tr.entry_of(i)['path'])[:-len('.dart')] for i in lane.get('files', []))
        for stem in sorted(set(re.findall(r'(?:integration_)?test/([a-z0-9_]+_test)\.dart', text))):
            self.assertIn(stem, covered)


if __name__ == '__main__':
    unittest.main(verbosity=2)
