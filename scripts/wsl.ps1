# Run a command in the MiniServe WSL environment from Windows PowerShell.
#
#   .\scripts\wsl.ps1 pytest -q
#   .\scripts\wsl.ps1 miniserve serve --port 8000
#   .\scripts\wsl.ps1 python bench/bandwidth.py
#
# It maps this repo's Windows path to its /mnt/<drive>/... path and runs scripts/dev.sh there,
# which activates the venv on WSL's disk and cd's into the repo.
$repo = Split-Path -Parent $PSScriptRoot
$drive = $repo.Substring(0, 1).ToLower()
$wslRepo = "/mnt/$drive" + $repo.Substring(2).Replace('\', '/')
wsl -d Ubuntu-24.04 -u root -- bash "$wslRepo/scripts/dev.sh" @args
