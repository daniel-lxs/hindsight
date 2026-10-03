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
  tag="integrations/coding-agents/$1"
  git fetch --depth 1 upstream tag "$tag"
  git rebase "$tag"
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
