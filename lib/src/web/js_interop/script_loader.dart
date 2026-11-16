import 'dart:async';

import 'package:web/web.dart' as web;

/// Loads JavaScript scripts by injecting <script> tags into the document head.
///
/// A script that fails to load is retried, up to [attempts] tries in all,
/// waiting [retryDelay] before the second try and twice as long before each
/// one after it. The default CDN answers a cache miss with an HTTP 503 when it
/// cannot reach its origin, and that clears up within seconds.
Future<void> loadScript(
  List<String> urls, {
  int attempts = 3,
  Duration retryDelay = const Duration(seconds: 1),
}) async {
  final head = web.document.querySelector('head')!;

  for (final url in urls) {
    var delay = retryDelay;
    for (var attempt = 1; ; attempt++) {
      final scriptTag = web.HTMLScriptElement()
        ..src = url
        ..type = 'application/javascript'
        ..async = true;
      head.append(scriptTag);

      final loadEvent = scriptTag.onLoad.first;
      final errorEvent = scriptTag.onError.first;

      final success = await _waitForFirst(loadEvent, errorEvent);
      if (success) break;
      scriptTag.remove();
      if (attempt >= attempts) {
        throw Exception('Failed to load script: $url');
      }
      await Future<void>.delayed(delay);
      delay *= 2;
    }
  }
}

Future<bool> _waitForFirst(
  Future<web.Event> onLoad,
  Future<web.Event> onError,
) {
  final completer = Completer<bool>();
  onLoad.then((value) => completer.complete(true));
  onError.then((value) => completer.complete(false));
  return completer.future;
}
