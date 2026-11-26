import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_litert/native.dart';
import 'package:flutter_litert/src/compiled_model/compiled_model_native.dart'
    show buildGpuOptionsPayload, prepareGpuCompilationCache;
import 'package:flutter_test/flutter_test.dart';

void main() {
  const cache = GpuCompilationCache(directory: '/data/cache/gpu', key: 'm_v1');

  group('buildGpuOptionsPayload', () {
    test('fp16 without cache sends nothing', () {
      expect(buildGpuOptionsPayload(precision: Precision.fp16), isNull);
    });

    test('fp32 without cache sends precision only', () {
      expect(
        buildGpuOptionsPayload(precision: Precision.fp32),
        'precision = 2\n',
      );
    });

    test('fp16 with cache sends the three cache lines', () {
      expect(
        buildGpuOptionsPayload(precision: Precision.fp16, cache: cache),
        'serialization_dir = "/data/cache/gpu"\n'
        'model_cache_key = "m_v1"\n'
        'serialize_program_cache = true\n',
      );
    });

    test('fp32 with cache prepends precision; programCache false', () {
      expect(
        buildGpuOptionsPayload(
          precision: Precision.fp32,
          cache: const GpuCompilationCache(
            directory: '/d',
            key: 'k',
            programCache: false,
          ),
        ),
        'precision = 2\n'
        'serialization_dir = "/d"\n'
        'model_cache_key = "k"\n'
        'serialize_program_cache = false\n',
      );
    });

    test('keeps single quotes', () {
      // LiteRT strips only the outer pair of quotes, so an apostrophe inside
      // the double-quoted value reaches the GPU options unchanged.
      expect(
        buildGpuOptionsPayload(
          precision: Precision.fp16,
          cache: const GpuCompilationCache(directory: "/d/it's", key: "m's"),
        ),
        'serialization_dir = "/d/it\'s"\n'
        'model_cache_key = "m\'s"\n'
        'serialize_program_cache = true\n',
      );
    });

    test('rejects double quotes, line breaks and empty values', () {
      for (final bad in ['a"b', 'a\nb', 'a\rb', '']) {
        expect(
          () => buildGpuOptionsPayload(
            precision: Precision.fp32,
            cache: GpuCompilationCache(directory: bad, key: 'k'),
          ),
          throwsArgumentError,
          reason: 'directory ${bad.codeUnits}',
        );
        expect(
          () => buildGpuOptionsPayload(
            precision: Precision.fp32,
            cache: GpuCompilationCache(directory: '/d', key: bad),
          ),
          throwsArgumentError,
          reason: 'key ${bad.codeUnits}',
        );
      }
    });

    test('rejects path separators in the key, not in the directory', () {
      // LiteRT names the cache file after the key inside the directory, so a
      // separator would point at a subfolder that does not exist.
      for (final bad in ['models/m_v1', r'models\m_v1']) {
        expect(
          () => buildGpuOptionsPayload(
            precision: Precision.fp16,
            cache: GpuCompilationCache(directory: '/d', key: bad),
          ),
          throwsArgumentError,
          reason: bad,
        );
      }
      expect(
        buildGpuOptionsPayload(
          precision: Precision.fp16,
          cache: const GpuCompilationCache(directory: r'C:\d\gpu', key: 'k'),
        ),
        'serialization_dir = "C:\\d\\gpu"\n'
        'model_cache_key = "k"\n'
        'serialize_program_cache = true\n',
      );
    });
  });

  group('prepareGpuCompilationCache', () {
    late Directory root;

    setUp(() {
      root = Directory.systemTemp.createTempSync('flutter_litert_gpu_cache_');
    });

    tearDown(() {
      if (root.existsSync()) root.deleteSync(recursive: true);
    });

    test('creates the directory and leaves an existing one as it is', () {
      final directory = Directory('${root.path}/a/b');
      final cache = GpuCompilationCache(directory: directory.path, key: 'k');
      prepareGpuCompilationCache(cache);
      expect(directory.existsSync(), isTrue);

      final entry = File('${directory.path}/entry.bin')..writeAsStringSync('x');
      prepareGpuCompilationCache(cache);
      expect(entry.readAsStringSync(), 'x');
    });

    test('accepts single quotes', () {
      final directory = Directory("${root.path}/it's");
      prepareGpuCompilationCache(
        GpuCompilationCache(directory: directory.path, key: "m's"),
      );
      expect(directory.existsSync(), isTrue);
    });

    test('rejects a relative directory without creating it', () {
      const relative = 'flutter_litert_relative_gpu_cache';
      expect(
        () => prepareGpuCompilationCache(
          const GpuCompilationCache(directory: relative, key: 'k'),
        ),
        throwsArgumentError,
      );
      expect(Directory(relative).existsSync(), isFalse);
    });

    test('rejects invalid values', () {
      expect(
        () => prepareGpuCompilationCache(
          GpuCompilationCache(directory: '${root.path}/a"b', key: 'k'),
        ),
        throwsArgumentError,
      );
      expect(
        () => prepareGpuCompilationCache(
          GpuCompilationCache(directory: root.path, key: ''),
        ),
        throwsArgumentError,
      );
    });

    test('rejects separators in the key and creates nothing', () {
      final directory = Directory('${root.path}/gpu');
      for (final bad in ['models/m_v1', r'models\m_v1']) {
        expect(
          () => prepareGpuCompilationCache(
            GpuCompilationCache(directory: directory.path, key: bad),
          ),
          throwsArgumentError,
          reason: bad,
        );
      }
      expect(directory.existsSync(), isFalse);
    });

    test('throws when the directory cannot be created', () {
      final file = File('${root.path}/file')..writeAsStringSync('');
      expect(
        () => prepareGpuCompilationCache(
          GpuCompilationCache(directory: '${file.path}/gpu', key: 'k'),
        ),
        throwsA(isA<FileSystemException>()),
      );
    });

    test('a bad cache throws before the GPU attempt and the CPU fallback', () {
      final file = File('${root.path}/file')..writeAsStringSync('');
      final cases = <(GpuCompilationCache, Matcher)>[
        (
          const GpuCompilationCache(directory: 'relative/gpu', key: 'k'),
          throwsArgumentError,
        ),
        (
          GpuCompilationCache(directory: '${file.path}/gpu', key: 'k'),
          throwsA(isA<FileSystemException>()),
        ),
        (
          GpuCompilationCache(directory: root.path, key: 'models/m_v1'),
          throwsArgumentError,
        ),
        (
          GpuCompilationCache(directory: root.path, key: r'models\m_v1'),
          throwsArgumentError,
        ),
      ];
      for (final (cache, matcher) in cases) {
        for (final config in [
          CompiledModelConfig.auto(gpuCache: cache),
          CompiledModelConfig.gpuWithCpuFallback(gpuCache: cache),
        ]) {
          final fallbacks = <Object>[];
          expect(
            () => CompiledModel.fromFileWithConfig(
              '${root.path}/missing.tflite',
              config: config,
              onFallback: fallbacks.add,
            ),
            matcher,
            reason: '$config',
          );
          expect(
            () => CompiledModel.fromBufferWithConfig(
              Uint8List(8),
              config: config,
              onFallback: fallbacks.add,
            ),
            matcher,
            reason: '$config',
          );
          expect(fallbacks, isEmpty, reason: '$config');
        }
      }
    });
  });

  group('CompiledModelConfig.gpuCache', () {
    test('is carried by GPU-capable constructors and copyWith', () {
      for (final c in [
        const CompiledModelConfig(gpuCache: cache),
        const CompiledModelConfig.auto(gpuCache: cache),
        const CompiledModelConfig.gpu(gpuCache: cache),
        const CompiledModelConfig.gpuWithCpuFallback(gpuCache: cache),
      ]) {
        expect(c.gpuCache, cache);
        expect(c.copyWith(policy: CompiledModelPolicy.cpu).gpuCache, cache);
        expect(c.copyWith(clearGpuCache: true).gpuCache, isNull);
        expect(c, isNot(const CompiledModelConfig()));
        expect(c.toString(), contains('m_v1'));
      }
      expect(const CompiledModelConfig.cpu().gpuCache, isNull);
    });

    test('equality and hashCode include the cache', () {
      const other = GpuCompilationCache(
        directory: '/data/cache/gpu',
        key: 'm_v1',
      );
      expect(
        const CompiledModelConfig.gpu(gpuCache: cache),
        const CompiledModelConfig.gpu(gpuCache: other),
      );
      expect(
        const CompiledModelConfig.gpu(gpuCache: cache).hashCode,
        const CompiledModelConfig.gpu(gpuCache: other).hashCode,
      );
    });
  });
}
