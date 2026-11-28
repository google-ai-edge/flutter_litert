@TestOn('browser')
library;

import 'dart:js_interop';

import 'package:flutter_litert/src/web/js_interop/script_loader.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:web/web.dart' as web;

/// Contract test for the `<script>` loader behind `initializeWeb()`: a failed
/// load is retried before it throws, a load that succeeds is not repeated, and
/// a script with a Subresource Integrity value is loaded with it.
void main() {
  /// Counts the `<script>` tags with [url] appended to the head while [body]
  /// runs, including ones the loader removed again.
  Future<int> countAppended(String url, Future<void> Function() body) async {
    var appended = 0;
    void count(JSArray<web.MutationRecord> records) {
      for (final record in records.toDart) {
        final nodes = record.addedNodes;
        for (var i = 0; i < nodes.length; i++) {
          final node = nodes.item(i);
          if (node != null &&
              node.isA<web.HTMLScriptElement>() &&
              (node as web.HTMLScriptElement).src == url) {
            appended++;
          }
        }
      }
    }

    final observer = web.MutationObserver(
      (JSArray<web.MutationRecord> records, web.MutationObserver _) {
        count(records);
      }.toJS,
    );
    observer.observe(
      web.document.head!,
      web.MutationObserverInit(childList: true),
    );
    try {
      await body();
    } finally {
      count(observer.takeRecords());
      observer.disconnect();
    }
    return appended;
  }

  int tagsInHead(String url) =>
      web.document.querySelectorAll('script[src="$url"]').length;

  /// The `<script>` tag for [url] that is in the head now.
  web.HTMLScriptElement tagInHead(String url) =>
      web.document.querySelector('script[src="$url"]')!
          as web.HTMLScriptElement;

  test(
    'a script that keeps failing is tried every attempt, then throws',
    () async {
      // Port 1 is never listening, so every load fails at once.
      const url = 'http://127.0.0.1:1/missing-script.js';
      final appended = await countAppended(url, () async {
        await expectLater(
          loadScript([url], attempts: 3, retryDelay: Duration.zero),
          throwsA(
            isA<Exception>().having((e) => '$e', 'message', contains(url)),
          ),
        );
      });
      expect(appended, 3);
      expect(tagsInHead(url), 0, reason: 'failed tags are removed');
    },
  );

  test('a script that loads is appended once', () async {
    const url = 'data:application/javascript,void%200';
    final appended = await countAppended(url, () async {
      await loadScript([url], retryDelay: Duration.zero);
    });
    expect(appended, 1);
    expect(tagsInHead(url), 1);
  });

  test('a script listed in integrity carries its hash', () async {
    const url = 'data:application/javascript,void%201';
    // SHA-384 of the script's bytes, `void 1`.
    const hash =
        'sha384-SMVLBa9259ov2wxjn/Ghu0wBnOhWiGblZGeiVXf64SqUNhQhvXSyXtjno/n0HVTS';
    await loadScript([url], integrity: {url: hash}, retryDelay: Duration.zero);
    final tag = tagInHead(url);
    expect(tag.integrity, hash);
    expect(tag.crossOrigin, 'anonymous');
  });

  test('a script not listed in integrity is loaded without a check', () async {
    const url = 'data:application/javascript,void%202';
    await loadScript(
      [url],
      integrity: {'https://example.com/other.js': 'sha384-AAAA'},
      retryDelay: Duration.zero,
    );
    final tag = tagInHead(url);
    expect(tag.integrity, isEmpty);
    expect(tag.crossOrigin, isNull);
  });

  test('a script whose bytes do not match its hash fails to load', () async {
    const url = 'data:application/javascript,void%203';
    // SHA-384 of `void 1`, not of this script.
    const hash =
        'sha384-SMVLBa9259ov2wxjn/Ghu0wBnOhWiGblZGeiVXf64SqUNhQhvXSyXtjno/n0HVTS';
    await expectLater(
      loadScript(
        [url],
        integrity: {url: hash},
        attempts: 2,
        retryDelay: Duration.zero,
      ),
      throwsA(isA<Exception>().having((e) => '$e', 'message', contains(url))),
    );
    expect(tagsInHead(url), 0, reason: 'failed tags are removed');
  });
}
