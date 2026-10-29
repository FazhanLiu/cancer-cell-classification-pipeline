#!/usr/bin/env bash
# Run with an authenticated GitHub CLI. Does not enable deployment.
set -Eeuo pipefail
repo=FazhanLiu/cancer-cell-classification-pipeline
configuration=$(mktemp -d)
trap 'rm -rf "$configuration"' EXIT
cat > "$configuration/environment.json" <<'JSON'
{"deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}}
JSON
cat > "$configuration/protection.json" <<'JSON'
{"required_status_checks":{"strict":true,"contexts":["test"]},"enforce_admins":false,"required_pull_request_reviews":{"required_approving_review_count":0,"dismiss_stale_reviews":true},"restrictions":null,"allow_force_pushes":false,"allow_deletions":false}
JSON
gh() {
    local attempt
    for attempt in 1 2 3; do
        command gh "$@" && return 0
        sleep 2
    done
    return 1
}
# Existing master is preserved for recovery.
gh api --method PATCH "repos/$repo" -f default_branch=main --silent
for pair in development:dev production:main; do
    environment=${pair%:*}; branch=${pair#*:}
    gh api --method PUT "repos/$repo/environments/$environment" --silent --input "$configuration/environment.json"
    policies=$(gh api "repos/$repo/environments/$environment/deployment-branch-policies" --jq ".branch_policies[] | select(.name == \"$branch\") | .id")
    if [[ -z "$policies" ]]; then
        gh api --method POST "repos/$repo/environments/$environment/deployment-branch-policies" -f name="$branch" -f type=branch --silent
    fi
done
# One owner can merge a green PR without needing an impossible self-approval.
gh api --method PUT "repos/$repo/branches/main/protection" --silent --input "$configuration/protection.json"
echo 'CI configured. Deploy stays disabled until both environments are ready.'
