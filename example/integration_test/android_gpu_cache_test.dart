// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// Emulators have no working OpenCL, so the GPU test cannot pass there; pass
/// `--dart-define=FLUTTER_LITERT_EXPECT_GPU=false` to run only the checks that
/// need no GPU.
const _expectGpu = bool.fromEnvironment(
  'FLUTTER_LITERT_EXPECT_GPU',
  defaultValue: true,
);

Future<Uint8List> _loadModel() async {
  final data = await rootBundle.load('assets/mobilefacenet.tflite');
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

/// Physical-device check for [GpuCompilationCache]: the second compile with
/// the same cache is faster and the directory gets files, outputs match the
/// uncached run, and a different key does not reuse the old entries.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('with a GPU', skip: !_expectGpu, () {
    testWidgets('GPU kernel cache is written, reused, and keyed', (
      tester,
    ) async {
      expect(Platform.isAndroid, isTrue);
      final bytes = await _loadModel();

      final root = Directory(
        '${Directory.systemTemp.path}/gpu_cache_test_${DateTime.now().microsecondsSinceEpoch}',
      );
      addTearDown(() {
        if (root.existsSync()) root.deleteSync(recursive: true);
      });
      final dirA = '${root.path}/a'; // intentionally not created

      Future<(int, List<double>)> load(CompiledModelConfig config) async {
        final sw = Stopwatch()..start();
        final model = CompiledModel.fromBufferWithConfig(bytes, config: config);
        sw.stop();
        try {
          expect(model.accelerators, contains(Accelerator.gpu));
          final inputs = <Float32List>[
            for (final n in model.inputByteSizes)
              Float32List(n ~/ 4)
                ..setAll(0, [for (var i = 0; i < n ~/ 4; i++) (i % 17) / 16]),
          ];
          final out = model.run(inputs).first;
          return (sw.elapsedMilliseconds, List<double>.of(out));
        } finally {
          model.close();
        }
      }

      int fileCount(String dir) => Directory(dir).existsSync()
          ? Directory(dir).listSync(recursive: true).whereType<File>().length
          : 0;

      final (plainMs, plainOut) = await load(
        const CompiledModelConfig.gpuWithCpuFallback(),
      );
      print('no cache: ${plainMs}ms');

      final cacheA = GpuCompilationCache(directory: dirA, key: 'model_a');
      final (firstMs, firstOut) = await load(
        CompiledModelConfig.gpuWithCpuFallback(gpuCache: cacheA),
      );
      final filesAfterFirst = fileCount(dirA);
      print('cache first: ${firstMs}ms, files=$filesAfterFirst');
      expect(filesAfterFirst, greaterThan(0), reason: 'cache directory empty');

      final (secondMs, secondOut) = await load(
        CompiledModelConfig.gpuWithCpuFallback(gpuCache: cacheA),
      );
      print('cache second: ${secondMs}ms, files=${fileCount(dirA)}');
      expect(
        fileCount(dirA),
        filesAfterFirst,
        reason: 'same key must reuse entries',
      );
      expect(
        secondMs,
        lessThan(firstMs ~/ 2 + 1),
        reason: 'second load not faster',
      );

      expect(firstOut, plainOut);
      expect(secondOut, plainOut);

      final cacheB = GpuCompilationCache(directory: dirA, key: 'model_b');
      final (thirdMs, thirdOut) = await load(
        CompiledModelConfig.gpuWithCpuFallback(gpuCache: cacheB),
      );
      print('different key: ${thirdMs}ms, files=${fileCount(dirA)}');
      expect(
        fileCount(dirA),
        greaterThan(filesAfterFirst),
        reason: 'new key must write new entries',
      );
      expect(thirdOut, plainOut);
    });
  });

  // Needs no GPU: the settings are checked before the GPU attempt, so they
  // fail the same way where that attempt would fail and fall back to CPU.
  testWidgets('invalid cache settings fail instead of falling back to CPU', (
    tester,
  ) async {
    expect(Platform.isAndroid, isTrue);
    final bytes = await _loadModel();

    final root = Directory.systemTemp.createTempSync('gpu_cache_invalid_');
    addTearDown(() {
      if (root.existsSync()) root.deleteSync(recursive: true);
    });
    final file = File('${root.path}/file')..writeAsStringSync('');

    final cases = <(GpuCompilationCache, Matcher)>[
      (GpuCompilationCache(directory: '/x"y', key: 'k'), throwsArgumentError),
      (
        const GpuCompilationCache(directory: 'relative/gpu', key: 'k'),
        throwsArgumentError,
      ),
      (
        GpuCompilationCache(directory: '${file.path}/gpu', key: 'k'),
        throwsA(isA<FileSystemException>()),
      ),
    ];
    for (final (cache, matcher) in cases) {
      final fallbacks = <Object>[];
      expect(
        () => CompiledModel.fromBufferWithConfig(
          bytes,
          config: CompiledModelConfig.auto(gpuCache: cache),
          onFallback: fallbacks.add,
        ),
        matcher,
        reason: '$cache',
      );
      expect(fallbacks, isEmpty, reason: '$cache');
    }
  });
}
