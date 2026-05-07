#!/usr/bin/env bash
# Release htpolynet at a given version.
#
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 2.0.0
#
# Prerequisites (checked automatically):
#   - Working tree must be clean (no uncommitted changes)
#   - Must be on the main branch
#   - CHANGELOG.md must have an "## [Unreleased]" section
#
# What it does:
#   1. Rotates CHANGELOG.md: renames [Unreleased] to [<version>] - <date>
#      and inserts a fresh empty [Unreleased] section above it
#   2. Updates the version in pyproject.toml
#   3. Commits both changes as "Release v<version>"
#   4. Creates tag v<version>
#   5. Pushes the commit and the tag to origin
#
# The pushed tag triggers release.yaml, which runs tests, builds the package,
# publishes to PyPI, creates a GitHub Release with the CHANGELOG notes, and
# triggers a ReadTheDocs build.

set -euo pipefail

VERSION="${1:?Usage: scripts/release.sh <version>  (e.g. 2.0.0)}"
TODAY="$(date +%Y-%m-%d)"

# ── Preconditions ─────────────────────────────────────────────────────────────

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree has uncommitted changes — commit or stash them first"
    exit 1
fi

BRANCH="$(git branch --show-current)"
if [ "$BRANCH" != "main" ]; then
    echo "ERROR: must be on main branch (currently on '$BRANCH')"
    exit 1
fi

if ! grep -q "^## \[Unreleased\]" CHANGELOG.md; then
    echo "ERROR: no '## [Unreleased]' section found in CHANGELOG.md"
    exit 1
fi

if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "ERROR: tag v$VERSION already exists locally"
    exit 1
fi

if git ls-remote --tags origin "refs/tags/v$VERSION" | grep -q .; then
    echo "ERROR: tag v$VERSION already exists on origin"
    exit 1
fi

# ── CHANGELOG rotation ────────────────────────────────────────────────────────

echo "Rotating CHANGELOG.md: [Unreleased] -> [$VERSION] - $TODAY"
sed -i "s/^## \[Unreleased\]/## [$VERSION] - $TODAY/" CHANGELOG.md

# Insert a fresh [Unreleased] section before the new release
sed -i "s/^## \[$VERSION\] - $TODAY/## [Unreleased]\n\n## [$VERSION] - $TODAY/" CHANGELOG.md

# ── Version bump ──────────────────────────────────────────────────────────────

echo "Bumping pyproject.toml version to $VERSION"
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

ACTUAL="$(grep '^version = ' pyproject.toml | sed 's/version = \"\(.*\)\"/\1/')"
if [ "$ACTUAL" != "$VERSION" ]; then
    echo "ERROR: version in pyproject.toml is '$ACTUAL' after sed — check the file"
    git checkout pyproject.toml CHANGELOG.md
    exit 1
fi

# ── Commit, tag, push ─────────────────────────────────────────────────────────

git add pyproject.toml CHANGELOG.md
git commit -m "Release v$VERSION"
git tag "v$VERSION"

echo "Pushing commit and tag v$VERSION to origin..."
git push origin main
git push origin "v$VERSION"

echo ""
echo "Done. The release.yaml workflow will now build and publish v$VERSION."
