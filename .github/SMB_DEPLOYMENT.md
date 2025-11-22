# SMB Deployment Workflow

This workflow automatically deploys the contents of the `www/` directory to an SMB network share whenever changes are pushed to the `main` branch.

## Installation

**The workflow file must be added manually** due to GitHub security restrictions on automated workflow creation.

### Steps to Add the Workflow:

1. Go to your GitHub repository
2. Navigate to `.github/workflows/`
3. Create a new file named `deploy-smb.yml`
4. Copy the contents from `.github/deploy-smb.yml.template` in this repository
5. Commit the file

Alternatively, you can use the GitHub web interface:
- Click "Add file" → "Create new file"
- Name it `.github/workflows/deploy-smb.yml`
- Paste the template contents
- Commit directly to your desired branch

## Required GitHub Secrets

Before the workflow can run successfully, you must configure the following secrets in your GitHub repository:

### Setting Up Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each of the following secrets:

### Required Secrets

| Secret Name | Description | Example |
|------------|-------------|---------|
| `SMB_SERVER` | The SMB server hostname or IP address | `files.example.com` or `192.168.1.100` |
| `SMB_SHARE` | The name of the SMB share to mount | `webfiles` or `public_html` |
| `SMB_USERNAME` | Username for SMB authentication | `deploy_user` |
| `SMB_PASSWORD` | Password for SMB authentication | `your_secure_password` |

### Example Configuration

If your SMB share is located at:
```
\\files.example.com\webfiles
```

Your secrets should be:
- **SMB_SERVER**: `files.example.com`
- **SMB_SHARE**: `webfiles`
- **SMB_USERNAME**: Your SMB username
- **SMB_PASSWORD**: Your SMB password

## Workflow Triggers

The workflow runs automatically when:
- Changes are pushed to the `main` branch that affect files in `www/`
- You can also trigger it manually from the GitHub Actions tab

## How It Works

1. **Checkout**: Checks out the repository code
2. **Verify**: Confirms the `www/` directory exists
3. **Mount**: Creates a mount point and mounts the SMB share using macOS's `mount_smbfs`
4. **Sync**: Uses `rsync` to copy files from `www/` to the mounted share
   - Preserves file attributes
   - Deletes files in the destination that don't exist in source
   - Excludes `.git*` and `.DS_Store` files
5. **Unmount**: Safely unmounts the SMB share
6. **Cleanup**: Removes the mount point

## Manual Triggering

To manually trigger the deployment:

1. Go to the **Actions** tab in your GitHub repository
2. Select **Deploy to SMB Share** from the workflow list
3. Click **Run workflow**
4. Select the branch (typically `main`)
5. Click **Run workflow**

## Troubleshooting

### Mount Failures

If the workflow fails to mount the SMB share:
- Verify all secrets are set correctly
- Check that the SMB server is accessible from GitHub's macOS runners
- Ensure the SMB share permissions allow the specified user to write
- Check if the server requires specific SMB protocol versions

### Sync Issues

If files aren't syncing correctly:
- Check the workflow logs for rsync errors
- Verify the destination has sufficient space
- Ensure the SMB user has write permissions on the share

### Network Issues

GitHub Actions runners need network access to your SMB server:
- If your SMB server is behind a firewall, you may need to whitelist GitHub's runner IP ranges
- Consider using a VPN or tunneling solution if direct access isn't possible

## Security Notes

- **Never commit credentials**: All sensitive information is stored in GitHub Secrets
- **Use strong passwords**: Ensure SMB credentials are secure
- **Limit permissions**: The SMB user should have only the necessary permissions
- **Consider environment-specific secrets**: Use different secrets for different environments

## Customization

### Change Sync Behavior

Edit the `rsync` command in the workflow to modify sync behavior:

```yaml
# Don't delete files from destination
rsync -av www/ ~/smb_mount/

# Only sync specific subdirectories
rsync -av www/html/ ~/smb_mount/

# Add more exclusions
rsync -av --delete \
  --exclude='.git*' \
  --exclude='.DS_Store' \
  --exclude='*.tmp' \
  www/ ~/smb_mount/
```

### Change Trigger Conditions

Modify the `on:` section to change when the workflow runs:

```yaml
# Run on all pushes to main
on:
  push:
    branches:
      - main

# Run on releases
on:
  release:
    types: [published]

# Run on schedule (e.g., daily at midnight UTC)
on:
  schedule:
    - cron: '0 0 * * *'
```

## Alternative: Using CIFS/SMB v1

If your SMB server requires SMB v1 (legacy), modify the mount command:

```yaml
mount_smbfs -o vers=1.0 "${SMB_URL}" ~/smb_mount
```

## Alternative: Using smbutil

For more control over authentication:

```yaml
# Create credentials file
echo "username=${SMB_USERNAME}" > ~/.smbcredentials
echo "password=${SMB_PASSWORD}" >> ~/.smbcredentials
chmod 600 ~/.smbcredentials

# Mount with credentials file
mount_smbfs -N -f 0755 -d 0755 \
  //${SMB_SERVER}/${SMB_SHARE} ~/smb_mount
```
