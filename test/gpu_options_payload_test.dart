import 'package:flutter_litert/native.dart';
import 'package:flutter_litert/src/compiled_model/compiled_model_native.dart'
    show buildGpuOptionsPayload;
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

    test('rejects quotes, newlines and empty values', () {
      for (final bad in ['a"b', "a'b", 'a\nb', 'a\rb', '']) {
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
