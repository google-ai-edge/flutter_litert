// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

String? get _skip => Platform.isMacOS ? null : 'Metal GPU cache, macOS only';

/// [GpuCompilationCache] on the bundled macOS Metal accelerator.
///
/// Metal writes one `<key>_mldrift_program_cache.bin` per key and reuses it
/// for the same key. Loading from the cache is not measurably faster on the
/// CI runners (Metal's own shader cache already speeds up a second compile),
/// so unlike android_gpu_cache_test.dart this checks the files and the
/// outputs, never the load time.
///
/// Run with:
///   flutter test integration_test/macos_gpu_cache_test.dart -d macos
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late Uint8List bytes;
  late List<double> uncachedOut;
  late Directory root;

  // Compiles on strict GPU, runs once and closes the model, so whatever the
  // load writes is on disk when this returns.
  List<double> load(CompiledModelConfig config) {
    final model = CompiledModel.fromBufferWithConfig(bytes, config: config);
    try {
      expect(model.accelerators, contains(Accelerator.gpu));
      final inputs = <Float32List>[
        for (final n in model.inputByteSizes)
          Float32List(n ~/ 4)
            ..setAll(0, [for (var i = 0; i < n ~/ 4; i++) (i % 17) / 16]),
      ];
      return List<double>.of(model.run(inputs).first);
    } finally {
      model.close();
    }
  }

  // File name to size in bytes, for every file under [dir].
  Map<String, int> files(String dir) {
    final d = Directory(dir);
    if (!d.existsSync()) return {};
    return {
      for (final f in d.listSync(recursive: true).whereType<File>())
        f.path.substring(dir.length + 1): f.lengthSync(),
    };
  }

  group('GpuCompilationCache (macOS Metal)', skip: _skip, () {
    setUpAll(() async {
      final data = await rootBundle.load('assets/simple_model.tflite');
      bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      uncachedOut = load(const CompiledModelConfig.gpu());
    });

    setUp(() {
      root = Directory(
        '${Directory.systemTemp.path}/macos_gpu_cache_test_${DateTime.now().microsecondsSinceEpoch}',
      );
    });

    tearDown(() {
      if (root.existsSync()) root.deleteSync(recursive: true);
    });

    testWidgets('one file per key, reused for the same key', (tester) async {
      final dir = '${root.path}/cache'; // intentionally not created
      const fileA = 'model_a_mldrift_program_cache.bin';
      const fileB = 'model_b_mldrift_program_cache.bin';
      final cacheA = GpuCompilationCache(directory: dir, key: 'model_a');
      final cacheB = GpuCompilationCache(directory: dir, key: 'model_b');

      final firstOut = load(CompiledModelConfig.gpu(gpuCache: cacheA));
      final afterFirst = files(dir);
      print('after first load: $afterFirst');
      expect(afterFirst.keys, [fileA]);
      expect(afterFirst[fileA], greaterThan(0));
      expect(firstOut, uncachedOut);

      final secondOut = load(CompiledModelConfig.gpu(gpuCache: cacheA));
      final afterSecond = files(dir);
      print('after second load: $afterSecond');
      expect(afterSecond, afterFirst, reason: 'same key must reuse its file');
      expect(secondOut, uncachedOut);

      final otherKeyOut = load(CompiledModelConfig.gpu(gpuCache: cacheB));
      final afterOtherKey = files(dir);
      print('after a new key: $afterOtherKey');
      expect(afterOtherKey.keys, unorderedEquals([fileA, fileB]));
      expect(afterOtherKey[fileA], afterFirst[fileA]);
      expect(afterOtherKey[fileB], greaterThan(0));
      expect(otherKeyOut, uncachedOut);
    });

    testWidgets('programCache: false writes nothing', (tester) async {
      final dir = '${root.path}/no_program_cache';
      final config = CompiledModelConfig.gpu(
        gpuCache: GpuCompilationCache(
          directory: dir,
          key: 'model_a',
          programCache: false,
        ),
      );

      expect(load(config), uncachedOut);
      expect(load(config), uncachedOut);
      expect(files(dir), isEmpty);
    });
  });
}
