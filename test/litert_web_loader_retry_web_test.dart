@TestOn('browser')
library;

import 'dart:async';

import 'package:flutter_litert/src/web/js_interop/litertjs_bindings.dart';
import 'package:flutter_test/flutter_test.dart';

/// Contract test for the LiteRT.js auto-loader's retry: a module or WASM
/// request that fails once with HTTP 503, as jsDelivr does when it cannot
/// reach its origin, must not fail the load.
///
/// Lives in its own file because the loader runs once per page. A local
/// server on the test host stands in for the CDN: it answers the first
/// request for each file with a 503. Its fake `loadLiteRt` fetches a file
/// under the WASM URL the way LiteRT.js loads its WASM files.

/// Runs on the test host and serves the fake LiteRT.js files.
const _flakyCdn = r'''
import 'dart:io';

import 'package:stream_channel/stream_channel.dart';

const _module = """
export async function loadLiteRt(wasmUrl) {
  const response = await fetch(wasmUrl + 'litert.wasm');
  if (!response.ok) throw new Error('WASM fetch failed: ' + response.status);
}
""";

Future<void> hybridMain(StreamChannel<Object?> channel) async {
  final counts = <String, int>{};
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
  server.listen((request) {
    final path = request.uri.path;
    final count = counts[path] = (counts[path] ?? 0) + 1;
    final response = request.response
      ..headers.set('access-control-allow-origin', '*');
    if (count == 1) {
      response.statusCode = HttpStatus.serviceUnavailable;
    } else if (path == '/core.mjs') {
      response.headers.contentType =
          ContentType('text', 'javascript', charset: 'utf-8');
      response.write(_module);
    } else if (path != '/wasm/litert.wasm') {
      response.statusCode = HttpStatus.notFound;
    }
    response.close();
  });
  channel.sink.add(server.port);
  // Any message asks for the request counts so far.
  await for (final _ in channel.stream) {
    channel.sink.add(counts);
  }
}
''';

void main() {
  test('a module and a WASM file that each fail once still load', () async {
    final cdn = spawnHybridCode(_flakyCdn);
    final replies = StreamIterator(cdn.stream);
    await replies.moveNext();
    final port = replies.current as int;

    configureLiteRtWebLoader(
      moduleUrl: 'http://127.0.0.1:$port/core.mjs',
      wasmUrl: 'http://127.0.0.1:$port/wasm/',
    );
    await waitForLiteRt();
    expect(isLiteRtReady(), isTrue);
    expect(liteRtLoadError(), isNull);

    cdn.sink.add('counts');
    await replies.moveNext();
    expect(replies.current, {'/core.mjs': 2, '/wasm/litert.wasm': 2});
  });
}
