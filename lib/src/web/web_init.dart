import 'js_interop/script_loader.dart';
import 'js_interop/tflite_js.dart';

/// tflite-js v0.0.1-alpha.10, addressed by the commit that tag points to. A
/// commit cannot be changed after the fact the way a tag can, and
/// `tf-tflite.min.js` loads its own loader script and WASM files from the same
/// folder, so they are pinned with it.
const _cdnBaseUrl =
    'https://cdn.jsdelivr.net/gh/hoomanmmd/tflite-js@b4430b71023538f08229a37e27cda31ee6a32d94';

/// Subresource Integrity values of the default scripts under [_cdnBaseUrl].
/// The browser refuses to run a default script whose bytes differ, so a
/// changed CDN response fails to load instead of running. Update these
/// together with [_cdnBaseUrl]:
///
///     curl -s <url> | openssl dgst -sha384 -binary | openssl base64 -A
const _cdnScriptIntegrity = <String, String>{
  'tf-core.js':
      'sha384-AUR5UU1ZfWidxVzuEqf8Azm+AMb9cBZ4MOhV2sPZAwe0wj21P82vP8ugFGBp/KEj',
  'tf-backend-cpu.js':
      'sha384-rpJ3gTOQOFaI/6oftY0OYuiMQIfdySw4DkdG4sasrRe4bIC2xLyBsoseoUDiJJFR',
  'tf-tflite.min.js':
      'sha384-woJnhcHArQcCSry/AgCVygXpK/hxhtZ9LlaH3uEBjWXlgnsOYvVENb7jk/TwpBFv',
};

/// Initializes the TFLite.js WASM runtime for web.
///
/// Must be called before creating any interpreter on web.
/// On native platforms, this is a no-op.
///
/// By default loads scripts from CDN. Pass custom URLs to load from
/// your own server (e.g., for offline support).
///
/// The default scripts are checked against pinned Subresource Integrity
/// hashes before they run. Custom URLs are loaded without a check.
Future<void> initializeWeb({
  String? tfJsScriptUrl,
  List<String>? tfBackendScriptUrls,
  String? tfliteScriptUrl,
}) async {
  if (isTFLiteInitialized()) return;

  final coreUrl = tfJsScriptUrl ?? '$_cdnBaseUrl/tf-core.js';
  final backendUrls = tfBackendScriptUrls ?? ['$_cdnBaseUrl/tf-backend-cpu.js'];
  final tfliteUrl = tfliteScriptUrl ?? '$_cdnBaseUrl/tf-tflite.min.js';

  await loadScript(
    [coreUrl, ...backendUrls, tfliteUrl],
    integrity: {
      for (final entry in _cdnScriptIntegrity.entries)
        '$_cdnBaseUrl/${entry.key}': entry.value,
    },
  );
}
