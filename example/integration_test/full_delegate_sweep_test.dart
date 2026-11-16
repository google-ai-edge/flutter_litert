// ignore_for_file: avoid_print, deprecated_member_use

// Comprehensive delegate sweep across every .tflite model in the sibling
// detection repos. Delegates are built DIRECTLY (not via InterpreterFactory)
// so a delegate that fails to load throws here instead of silently becoming
// CPU. Each (model, mode) cell is independently guarded so one bad model
// (e.g. dynamic input shape) doesn't abort the matrix.
//
//   flutter test integration_test/full_delegate_sweep_test.dart -d macos

import 'dart:io';
import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_litert/native.dart';

const int iterations = 25;
const int warmup = 8;

// Directory holding the detection package checkouts (animal_detection,
// face_detection_tflite, ...). Pass it with
// --dart-define=MODEL_REPOS_ROOT=<dir>; the tests skip when it is unset.
const _repositoriesRoot = String.fromEnvironment('MODEL_REPOS_ROOT');
const _skip = _repositoriesRoot == ''
    ? 'Pass --dart-define=MODEL_REPOS_ROOT=<dir> to run'
    : null;

// Searched for .tflite files under <MODEL_REPOS_ROOT>/<repository>/assets.
const repositories = <String>[
  'animal_detection',
  'face_detection_tflite',
  'object_detection',
  'pose_detection',
  'hand_detection',
  'cat_detection',
  'dog_detection',
];

// mode name -> builds (options, delegate-to-delete)
final modeBuilders = <String, (InterpreterOptions, Delegate?) Function()>{
  'cpu': () => (InterpreterOptions()..threads = 4, null),
  'xnnpack': () {
    final o = InterpreterOptions()..threads = 4;
    final d = XNNPackDelegate(options: XNNPackDelegateOptions(numThreads: 4));
    o.addDelegate(d);
    return (o, d);
  },
  'gpu_fp16': () {
    final o = InterpreterOptions();
    final d = GpuDelegate(
      options: GpuDelegateOptions(allowPrecisionLoss: true),
    );
    o.addDelegate(d);
    return (o, d);
  },
  'gpu_fp32': () {
    final o = InterpreterOptions();
    final d = GpuDelegate(
      options: GpuDelegateOptions(allowPrecisionLoss: false),
    );
    o.addDelegate(d);
    return (o, d);
  },
  // AllDevices(=1) matches what InterpreterFactory ships; engages CoreML on
  // hardware without a dedicated ANE path (e.g. macOS).
  'coreml': () {
    final o = InterpreterOptions();
    final d = CoreMlDelegate(options: CoreMlDelegateOptions(enabledDevices: 1));
    o.addDelegate(d);
    return (o, d);
  },
};

double _p50(List<int> t) {
  final s = List<int>.from(t)..sort();
  return s[((s.length - 1) * 0.50).floor()].toDouble();
}

double _std(List<int> t) {
  final m = t.reduce((a, b) => a + b) / t.length;
  return sqrt(t.map((v) => pow(v - m, 2)).reduce((a, b) => a + b) / t.length);
}

// Returns "  12.3" ms p50, or a short error tag.
String _benchCell(
  String path,
  (InterpreterOptions, Delegate?) Function() build,
) {
  InterpreterOptions? options;
  Delegate? delegate;
  Interpreter? interp;
  try {
    final r = build();
    options = r.$1;
    delegate = r.$2;
    final bytes = File(path).readAsBytesSync();
    interp = Interpreter.fromBuffer(bytes, options: options);
    interp.allocateTensors();
    for (int i = 0; i < warmup; i++) {
      interp.invoke();
    }
    final us = <int>[];
    for (int i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      interp.invoke();
      sw.stop();
      us.add(sw.elapsedMicroseconds);
    }
    final ms = us.map((u) => (u / 1000)).toList();
    final p50 = _p50(ms.map((e) => e.round()).toList());
    final std = _std(ms.map((e) => e.round()).toList());
    return '${p50.toStringAsFixed(0).padLeft(5)}±${std.toStringAsFixed(0)}';
  } catch (e) {
    final msg = e.toString();
    if (msg.contains('llocate') || msg.contains('precondition')) return ' DYN ';
    return ' ERR ';
  } finally {
    interp?.close();
    delegate?.delete();
  }
}

List<File> _discoverModels() {
  final found = <File>[];
  for (final repository in repositories) {
    final dir = Directory('$_repositoriesRoot/$repository/assets');
    if (!dir.existsSync()) continue;
    for (final e in dir.listSync(recursive: true, followLinks: false)) {
      if (e is! File) continue;
      final p = e.path;
      if (!p.endsWith('.tflite')) continue;
      if (p.contains('/build/') || p.contains('/.dart_tool/')) continue;
      found.add(e);
    }
  }
  found.sort((a, b) => a.lengthSync().compareTo(b.lengthSync()));
  return found;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final models = _discoverModels();
  final modeNames = modeBuilders.keys.toList();
  final rows = <String>[];

  group('Full delegate sweep (-d macos)', skip: _skip, () {
    for (final f in models) {
      final repo = f.path.substring(_repositoriesRoot.length + 1).split('/')[0];
      final name = f.path.split('/').last.replaceAll('.tflite', '');
      final sizeMb = (f.lengthSync() / (1024 * 1024)).toStringAsFixed(1);
      final label = '$repo/$name';

      test(label, timeout: const Timeout(Duration(minutes: 15)), () async {
        final cells = <String>[];
        for (final mode in modeNames) {
          cells.add(_benchCell(f.path, modeBuilders[mode]!));
        }
        final row =
            '${label.padRight(38)} ${sizeMb.padLeft(5)}MB  ${cells.map((c) => c.padLeft(8)).join(' ')}';
        rows.add(row);
        print('\n>>> $row\n');
      });
    }

    tearDownAll(() {
      print('\n${'=' * 110}');
      print(
        'FULL DELEGATE SWEEP: p50±std ms, -d macos (M-series). '
        'DYN=dynamic-shape/alloc-fail, ERR=error',
      );
      print('=' * 110);
      final header =
          '${'model'.padRight(38)} ${'size'.padLeft(5)}    ${modeNames.map((m) => m.padLeft(8)).join(' ')}';
      print(header);
      print('-' * 110);
      for (final r in rows) {
        print(r);
      }
      print('=' * 110);
    });
  });
}
