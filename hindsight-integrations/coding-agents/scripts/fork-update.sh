#!/usr/bin/env bash
# Maintain the fork's coding-agents plugin: rebase the fork's commits onto an upstream release,
# test, build, and install the build over the local plugin runtime.
#
#   scripts/fork-update.sh                 build and install the current branch as-is
#   scripts/fork-update.sh v0.9.0          first rebase onto upstream tag integrations/coding-agents/v0.9.0
#
# The plugin's own auto-update must stay off ("autoUpdate": false in ~/.hindsight/coding-agent.json),
# or it replaces this build with the npm release.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNTIME="${HINDSIGHT_RUNTIME_DIR:-$HOME/.hindsight/coding-agents}"

if [[ $# -ge 1 ]]; then
  # The fork's commits sit on top of the release named in package.json. This checkout is shallow, so
  # the two release tags share no history git can see: name the old base explicitly and replay only
  # the fork's own commits onto the new one.
  old="integrations/coding-agents/v$(node -p "require('./package.json').version")"
  new="integrations/coding-agents/$1"
  git fetch --depth 1 upstream tag "$old" tag "$new"
  git rebase --onto "$new" "$old"
fi

npm ci --no-audit --no-fund
# docs-harness-roster.test.ts needs the docs tree, which a sparse checkout of this package lacks.
npx vitest run --exclude '**/*.e2e.test.ts' --exclude 'src/docs-harness-roster.test.ts'
npm run build

version="$(node -p "require('./package.json').version")"
[[ -d "$RUNTIME/dist" ]] && rm -rf "$RUNTIME/dist.previous" && cp -R "$RUNTIME/dist" "$RUNTIME/dist.previous"
mkdir -p "$RUNTIME/dist"
cp -R dist/. "$RUNTIME/dist/"
echo "installed fork build of $version ($(git rev-parse --short HEAD)) into $RUNTIME/dist; previous build kept in dist.previous"
