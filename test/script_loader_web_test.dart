@TestOn('browser')
library;

import 'dart:js_interop';

import 'package:flutter_litert/src/web/js_interop/script_loader.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:web/web.dart' as web;

/// Contract test for the `<script>` loader behind `initializeWeb()`: a failed
/// load is retried before it throws, and a load that succeeds is not repeated.
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
}
