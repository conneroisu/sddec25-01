# Change: Add GitHub Profile Links to Team Members

## Why
Team members should have clickable GitHub profile links to showcase their GitHub contributions and make it easy for visitors to connect with them on GitHub. This addresses GitHub issue #157.

## What Changes
- Add GitHub profile icon link below each team member's name and position
- Add hover effect styling for team member GitHub links
- Links open in new tab with proper accessibility attributes

## Impact
- Affected specs: website-navigation (MODIFIED)
- Affected code: www/index.html (4 team member cards), www/css/site.css (new .team-github-link class)
