# Contributing to flutter_litert

Thanks for helping improve `flutter_litert`. This repository is a Flutter
plugin for LiteRT / TensorFlow Lite inference, native delegate loading,
custom ops, web runtimes, and shared ML utility code.

## Before opening a pull request

- Run `flutter pub get`.
- Run `dart format --output=none --set-exit-if-changed .`.
- Run `flutter analyze .`.
- Run `flutter test`.
- Add or update focused tests when changing public API, delegate behavior,
  tensor conversion, model training, custom ops, or web runtime code.

## Repeat runs of new tests

A pull request that adds or changes a test also starts the `test repeat`
workflow (`.github/workflows/test-repeat.yml`). It runs each new or changed
test 10 times in a row, alone, on every runner that runs it in `build`
(`flutter-ci.yml`) and with the same command, and fails when any run fails.
A changed helper repeats the tests that import it. `.github/test-lanes.json`
says which job runs which test; when `flutter-ci.yml` starts running a test,
or runs it another way, change its lane there too.

To repeat tests by hand, for example more often or on a device lane:

```sh
gh workflow run test-repeat.yml --ref <branch> \
  -f files='test/a_test.dart example/integration_test/b_test.dart' \
  -f runs=20 -f lanes=''
```

`lanes` empty picks every lane that runs the files. The `ios-simulator`,
`android-play-services` and `android-api24` lanes run only when named in
`lanes`. A lane named with no `files` runs all of its tests.

## Native and platform changes

- Keep platform-specific behavior documented in `README.md` and matching
  Dart docs.
- Do not change bundled runtime versions, generated FFI bindings, podspecs,
  or CMake files without updating the related docs and tests.
- When changing TensorFlow Lite / LiteRT C APIs, regenerate bindings with the
  `melos run ffigen` script.
- For custom ops, keep the op name, exported C registration symbol, and Dart
  FFI registration in sync.

## Pull request style

- Keep changes scoped to one behavior or documentation update.
- Call out platform coverage in the PR description.
- Mention any tests that were skipped and why.

## License

New files should use the Apache-2.0 license header used by the surrounding
source files unless the file is generated or follows an existing local pattern.
