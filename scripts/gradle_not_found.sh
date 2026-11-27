#!/usr/bin/env bash
# Tells a failed Gradle build whether to build again: exits 0 when its log
# shows that a repository answered "not found" for Kotlin or the kotlin-dsl
# plugin, after making Gradle ask the repositories again; exits 1 for any
# other failure.
#
# Maven Central answered "not found" for kotlin-stdlib 2.3.20, which it has,
# on 7 of 50 runners for the whole job (mediapipe_flutter 8307ab8,
# 2026-11-19). Flutter's own Gradle build, which needs it for the kotlin-dsl
# plugin, then failed on "Could not find org.jetbrains.kotlin:kotlin-stdlib:
# 2.3.20" or "Plugin [id: 'org.gradle.kotlin.kotlin-dsl', version: '6.5.7']
# was not found" (1 in about 7 Android jobs, 2026-11-26). Gradle tries a
# download again only after a network error or a 5xx, 408 or 429; a 404 is
# its answer, and it asks the next repository. gradle-plugin-repositories
# now asks Google's copy of Maven Central instead, which answered on all 50
# runners; this is for the copy or the plugin portal doing the same.
#
# Gradle remembers which repository did not have a module, with no expiry, so
# a second build would fail on the remembered answer without asking. This
# stops the Gradle daemons and removes the cached module metadata; the
# downloaded files stay, and Gradle reuses them when their checksums match.
#
# Usage: gradle_not_found.sh <build log>
set -u

line=$(grep -aoE "Plugin \[id: 'org\.(gradle\.kotlin\.kotlin-dsl|jetbrains\.kotlin\.[a-z.-]+)'[^]]*\] was not found|Could not find org\.(jetbrains|gradle)\.kotlin[.:][^ ]*" "$1" | head -n 1)
[ -n "$line" ] || exit 1
echo "::warning::Gradle: $line; asking the repositories again in 30 seconds"

pkill -f GradleDaemon
for _ in $(seq 30); do
  pgrep -f GradleDaemon > /dev/null || break
  sleep 1
done
pkill -9 -f GradleDaemon
rm -rf "${GRADLE_USER_HOME:-$HOME/.gradle}"/caches/modules-2/metadata-*
sleep 30
