# Streams this laptop's audio (Spotify) to the Boots Raspberry Pi speaker.
# Starts the sender in the background (no window) unless it is already running.
#
# Run it:   right-click > Run with PowerShell
# At login: make a shortcut in the Startup folder (Win+R, shell:startup) whose
#           target is:
#   powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File "<full path to this file>"

$py = Join-Path $env:USERPROFILE "boots-audio-venv\Scripts\pythonw.exe"
$sender = Join-Path $PSScriptRoot "laptop_audio_sender.py"

$running = Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -like "*laptop_audio_sender.py*" }
if ($running) {
    Write-Host "Boots audio sender is already running (PID $($running.ProcessId))."
    exit 0
}

Start-Process -FilePath $py -ArgumentList "`"$sender`"" -WindowStyle Hidden
Write-Host "Boots audio sender started."
