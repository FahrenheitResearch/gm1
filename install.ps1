# GM1 (gpuwm-cm1) installer for Windows 10 and 11.
#
# GM1 runs inside WSL2 (Ubuntu in Windows) on your NVIDIA graphics card. Open PowerShell (not Command Prompt)
# and run:
#
#   irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1 | iex
#
# It checks the Windows NVIDIA driver, installs WSL2 with Ubuntu when it is missing (Windows asks for
# permission; a computer without WSL then needs a restart), lets you choose Ubuntu's user name and password in
# this window, installs GM1 inside Ubuntu with install.sh, and adds a "GM1" shortcut to the Start menu and the
# desktop, and a "GM1 check" shortcut to the Start menu. GM1 starts GM1 and opens it in your browser; GM1 check
# checks the computer and saves gm1-report.txt on the desktop for a help request. Running the same line again
# updates GM1.
#
# Uninstall (your runs in Ubuntu's ~/gm1-runs are kept; WSL and Ubuntu stay installed):
#
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1))) -Uninstall
#
# Settings, as parameters of the line above or as environment variables set first in the same PowerShell window
# (irm | iex cannot pass parameters). For example, to install into the WSL distribution Ubuntu-24.04:
#
#   $env:GM1_DISTRO = 'Ubuntu-24.04'; irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1 | iex
#
# GM1_DISTRO (the WSL distribution, default Ubuntu), GM1_VERSION, GM1_WHEEL (a .whl file on Windows),
# GM1_INSTALL_SH (a local install.sh instead of the published one), GM1_DESKTOP_SHORTCUT=0 (Start menu only),
# GM1_SKIP_DOCTOR=1, GM1_NO_MODIFY_PATH=1, GM1_UNINSTALL=1.
#
# Everything runs inside functions and nothing ends the PowerShell session, so it is safe to pipe into
# Invoke-Expression: a failure prints what to do and returns to the prompt.

param(
    [switch]$Uninstall,
    [switch]$NoDesktopShortcut,
    [string]$Distro = ''
)

function Invoke-GM1Installer {
    param([bool]$Uninstall, [bool]$NoDesktopShortcut, [string]$Distro)

    Set-StrictMode -Version 2.0
    # Continue, not Stop: Windows PowerShell 5.1 turns a native program's stderr (wsl.exe's warnings about
    # .wslconfig, uv's progress) into errors, and Stop would end the install on the first one. Each step checks
    # its own result instead, and the cmdlets that must succeed say -ErrorAction Stop.
    $ErrorActionPreference = 'Continue'

    $RepoRaw = 'https://raw.githubusercontent.com/FahrenheitResearch/gm1/main'
    $InstallShUrl = "$RepoRaw/install.sh"
    $OneLiner = "irm $RepoRaw/install.ps1 | iex"
    $NvidiaDownload = 'https://www.nvidia.com/Download/index.aspx'
    $MinDriver = 580
    $ShortcutName = 'GM1.lnk'
    $CheckShortcutName = 'GM1 check.lnk'
    # GM1's own files on Windows: the shortcuts' icon and the GM1 check script.
    $AppFolder = Join-Path $env:LOCALAPPDATA 'GM1'
    # What install.sh removes from uv's download cache on uninstall (used when install.sh cannot be fetched).
    $CachedPackages = 'gpuwm-cm1 gpuwm gpuwm-data cupy-cuda13x cuda-toolkit cuda-pathfinder nvidia-cublas nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-nvjitlink imageio-ffmpeg'

    function Say([string]$Text) { Write-Host $Text }
    function Step([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Cyan }
    function Warn([string]$Text) { Write-Host ''; Write-Host "!!  $Text" -ForegroundColor Yellow }
    function Stop-WithHelp([string]$Text) {
        Write-Host ''
        Write-Host "GM1 was not installed: $Text" -ForegroundColor Red
        Write-Host "When it is fixed, run the same line again:  $OneLiner"
    }

    # wsl.exe writes UTF-16 unless WSL_UTF8 is set (newer WSL); strip the NULs either way.
    function Invoke-Wsl([string[]]$Arguments) {
        $previous = $env:WSL_UTF8
        $env:WSL_UTF8 = '1'
        $output = & wsl.exe @Arguments 2>$null
        $code = $LASTEXITCODE
        $env:WSL_UTF8 = $previous
        $lines = @($output | ForEach-Object { ([string]$_) -replace "`0", '' } | ForEach-Object { $_.Trim() } |
                   Where-Object { $_ -ne '' })
        return [pscustomobject]@{ Code = $code; Lines = $lines }
    }

    function Get-WslExe {
        $path = Join-Path $env:WINDIR 'System32\wsl.exe'
        if (Test-Path $path) { return $path }
        $command = Get-Command wsl.exe -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
        return $null
    }

    function Get-StartShortcut { return (Join-Path ([Environment]::GetFolderPath('Programs')) $ShortcutName) }
    function Get-DesktopShortcut { return (Join-Path ([Environment]::GetFolderPath('Desktop')) $ShortcutName) }
    function Get-CheckShortcut { return (Join-Path ([Environment]::GetFolderPath('Programs')) $CheckShortcutName) }

    # ------------------------------------------------------------ the Windows NVIDIA driver

    function Get-NvidiaDriver {
        $candidates = @(
            (Join-Path $env:WINDIR 'System32\nvidia-smi.exe'),
            (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
        )
        $command = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($command) { $candidates = @($command.Source) + $candidates }
        foreach ($smi in $candidates) {
            if (-not (Test-Path $smi)) { continue }
            try {
                $answer = & $smi --query-gpu=driver_version,name --format=csv,noheader 2>$null | Select-Object -First 1
            } catch { $answer = $null }
            if ($answer) {
                $text = [string]$answer
                $comma = $text.IndexOf(',')
                if ($comma -lt 0) { return [pscustomobject]@{ Version = $text.Trim(); Name = 'NVIDIA GPU' } }
                return [pscustomobject]@{ Version = $text.Substring(0, $comma).Trim(); Name = $text.Substring($comma + 1).Trim() }
            }
        }
        # No nvidia-smi: Windows' own record of the card (DriverVersion 32.0.15.8142 is NVIDIA 581.42).
        try {
            $card = Get-CimInstance Win32_VideoController -ErrorAction Stop |
                Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1
        } catch { $card = $null }
        if ($card -and $card.DriverVersion) {
            $digits = ($card.DriverVersion -replace '\.', '')
            if ($digits.Length -ge 5) {
                $tail = $digits.Substring($digits.Length - 5)
                return [pscustomobject]@{ Version = ($tail.Substring(0, 3) + '.' + $tail.Substring(3)); Name = $card.Name }
            }
        }
        return $null
    }

    function Test-Driver {
        $driver = Get-NvidiaDriver
        if (-not $driver) {
            Warn 'No NVIDIA graphics driver was found on Windows. GM1 installs anyway, but cannot run until it is fixed.'
            Say "    Fix: install the newest NVIDIA driver for your card (R$MinDriver or newer) from"
            Say "    $NvidiaDownload"
            Say '    then restart the computer. Install it on Windows only: WSL uses the Windows driver.'
            return
        }
        $major = 0
        [void][int]::TryParse(($driver.Version.Split('.')[0]), [ref]$major)
        if ($major -lt $MinDriver) {
            Warn "The NVIDIA driver on Windows is $($driver.Version); GM1 needs R$MinDriver or newer (CUDA 13)."
            Say "    Fix: update the Windows NVIDIA driver from $NvidiaDownload"
            Say '    then restart the computer. Never install an NVIDIA driver inside WSL: WSL uses the Windows one.'
            Say '    GM1 installs anyway.'
        } else {
            Say "NVIDIA driver $($driver.Version) on Windows, $($driver.Name)."
        }
    }

    # ------------------------------------------------------------ WSL2 and Ubuntu

    # True when Ubuntu was installed into a working WSL and the install can go on now; false when Windows must
    # restart first (WSL itself was missing) or the install did not work.
    function Install-WslUbuntu([bool]$WslPresent) {
        if ($WslPresent) {
            Step 'Installing Ubuntu in WSL (it downloads about 1 GB)'
            & wsl.exe --install -d Ubuntu --no-launch | Out-Host
            if ($LASTEXITCODE -ne 0) {
                Start-Process -FilePath (Get-WslExe) -ArgumentList @('--install', '-d', 'Ubuntu', '--no-launch') -Verb RunAs -Wait -ErrorAction Stop
            }
            return $true
        }
        Step 'Installing WSL2 with Ubuntu (Windows asks for permission)'
        Say 'This needs an administrator: click Yes when Windows asks. It downloads about 1 GB.'
        try {
            Start-Process -FilePath 'wsl.exe' -ArgumentList @('--install', '-d', 'Ubuntu') -Verb RunAs -Wait -ErrorAction Stop
        } catch {
            Stop-WithHelp 'Windows did not allow the WSL install. Open PowerShell as administrator (right-click it, "Run as administrator").'
            return $false
        }
        Say ''
        Say 'WSL2 and Ubuntu are installed. To finish:'
        Say '  1. Restart the computer.'
        Say '  2. After the restart Ubuntu opens by itself (or open "Ubuntu" from the Start menu) and asks for a'
        Say '     new user name and password (any you like; the password is not shown while you type it).'
        Say '  3. Close the Ubuntu window, open PowerShell and run the same line again:'
        Say "     $OneLiner"
        return $false
    }

    # Ubuntu's first start, in this window: it asks for a user name and a password, then shows its prompt.
    function Initialize-DistroUser([string]$Name) {
        Step "Setting up $Name (its first start)"
        Say "$Name now asks for a new user name and a password, here in this window. Choose any you like: small"
        Say 'letters for the name, and a password you will remember (it is not shown while you type it).'
        Say 'When it shows a prompt ending in $, type the word  exit  and press Enter: the GM1 install then goes on.'
        Say ''
        & wsl.exe -d $Name
        Say ''
    }

    function Select-Distro([string[]]$Names, [string]$Wanted) {
        if ($Wanted) {
            if ($Names -contains $Wanted) { return $Wanted }
            return $null
        }
        if ($Names -contains 'Ubuntu') { return 'Ubuntu' }
        $ubuntu = @($Names | Where-Object { $_ -like 'Ubuntu*' } | Sort-Object)
        if ($ubuntu.Count -gt 0) { return $ubuntu[-1] }
        return $null
    }

    function Get-DistroNames {
        $list = Invoke-Wsl @('--list', '--quiet')
        if ($list.Code -eq 0) { return @($list.Lines) }
        return @()
    }

    function Get-DistroVersion([string]$Name) {
        $list = Invoke-Wsl @('--list', '--verbose')
        foreach ($line in $list.Lines) {
            $fields = @(($line -replace '^\*', '').Trim() -split '\s+')
            if ($fields.Count -ge 3 -and $fields[0] -eq $Name) { return $fields[-1] }
        }
        return ''
    }

    # ------------------------------------------------------------ install.sh inside the distribution

    function Get-InstallSh {
        if ($env:GM1_INSTALL_SH) {
            if (-not (Test-Path $env:GM1_INSTALL_SH)) { throw "GM1_INSTALL_SH=$($env:GM1_INSTALL_SH) is not a file." }
            return [IO.File]::ReadAllText((Resolve-Path $env:GM1_INSTALL_SH).Path)
        }
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $content = (Invoke-WebRequest -UseBasicParsing -Uri $InstallShUrl -ErrorAction Stop).Content
        if ($content -is [byte[]]) { $content = [Text.Encoding]::UTF8.GetString($content) }
        return [string]$content
    }

    function Invoke-InstallSh([string]$Name, [string]$Script, [string[]]$Arguments) {
        # Copy the script in through stdin (no Windows path to translate), without Windows line ends or a
        # byte-order mark.
        $OutputEncoding = New-Object Text.UTF8Encoding $false
        $Script | & wsl.exe -d $Name --cd '~' --exec sh -c "tr -d '\r' | sed '1s/^\xef\xbb\xbf//' > ~/.gm1-install.sh"
        if ($LASTEXITCODE -ne 0) { throw 'Could not copy the installer into WSL.' }
        # The settings travel into WSL through WSLENV; /p turns a Windows path into a Linux one.
        $previousWslenv = $env:WSLENV
        $env:GM1_FROM_WINDOWS = '1'
        $names = @()
        foreach ($setting in @('GM1_VERSION', 'GM1_FIND_LINKS/p', 'GM1_WHEEL/p', 'GM1_CONSTRAINTS/p', 'GM1_REINSTALL',
                               'GM1_SKIP_DOCTOR', 'GM1_NO_MODIFY_PATH', 'GM1_FROM_WINDOWS')) {
            $plain = $setting.Split('/')[0]
            if ([Environment]::GetEnvironmentVariable($plain)) { $names += $setting }
        }
        if ($previousWslenv) { $names += $previousWslenv }
        $env:WSLENV = $names -join ':'
        try {
            $command = 'sh ~/.gm1-install.sh ' + ($Arguments -join ' ') + '; rc=$?; rm -f ~/.gm1-install.sh; exit $rc'
            # Out-Host: shown to the person, not returned with the exit code; read as UTF-8, as Linux writes it.
            $previousConsole = [Console]::OutputEncoding
            try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false } catch { }
            & wsl.exe -d $Name --cd '~' --exec bash -lc $command | Out-Host
            $code = $LASTEXITCODE
            try { [Console]::OutputEncoding = $previousConsole } catch { }
            return $code
        } finally {
            $env:WSLENV = $previousWslenv
            Remove-Item Env:GM1_FROM_WINDOWS -ErrorAction SilentlyContinue
        }
    }

    # ------------------------------------------------------------ the shortcuts

    # GM1's icon (a storm on a weather radar), drawn here so that no picture file has to be downloaded.
    function Write-GM1Icon([string]$Path) {
        Add-Type -AssemblyName System.Drawing
        $size = 48
        $bitmap = New-Object System.Drawing.Bitmap $size, $size
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        try {
            $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
            $graphics.Clear([System.Drawing.Color]::Transparent)
            $shape = New-Object System.Drawing.Drawing2D.GraphicsPath
            $r = 18
            $shape.AddArc(1, 1, $r, $r, 180, 90); $shape.AddArc(47 - $r, 1, $r, $r, 270, 90)
            $shape.AddArc(47 - $r, 47 - $r, $r, $r, 0, 90); $shape.AddArc(1, 47 - $r, $r, $r, 90, 90)
            $shape.CloseFigure()
            $graphics.FillPath((New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, 18, 32, 58))), $shape)
            foreach ($blob in @(
                    @(46, 160, 67, 9, 14, 29, 24), @(46, 160, 67, 23, 11, 17, 17), @(240, 200, 40, 14, 18, 21, 17),
                    @(240, 200, 40, 25, 15, 12, 11), @(220, 50, 40, 20, 21, 11, 10), @(220, 50, 40, 28, 18, 7, 6),
                    @(46, 160, 67, 9, 31, 11, 10), @(18, 32, 58, 13, 33, 6, 6))) {
                $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, $blob[0], $blob[1], $blob[2]))
                $graphics.FillEllipse($brush, $blob[3], $blob[4], $blob[5], $blob[6])
            }
        } finally { $graphics.Dispose() }
        # An .ico with one 48-pixel, 32-bit image (BGRA rows from the bottom up, then an empty mask).
        $rect = New-Object System.Drawing.Rectangle 0, 0, $size, $size
        $data = $bitmap.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
            [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
        $raw = New-Object byte[] ($data.Stride * $size)
        [Runtime.InteropServices.Marshal]::Copy($data.Scan0, $raw, 0, $raw.Length)
        $stride = $data.Stride
        $bitmap.UnlockBits($data)
        $bitmap.Dispose()
        $pixels = $size * $size * 4
        $mask = $size * 8
        $out = New-Object IO.MemoryStream
        $writer = New-Object IO.BinaryWriter $out
        $writer.Write([UInt16]0); $writer.Write([UInt16]1); $writer.Write([UInt16]1)
        $writer.Write([Byte]$size); $writer.Write([Byte]$size); $writer.Write([Byte]0); $writer.Write([Byte]0)
        $writer.Write([UInt16]1); $writer.Write([UInt16]32); $writer.Write([UInt32](40 + $pixels + $mask)); $writer.Write([UInt32]22)
        $writer.Write([UInt32]40); $writer.Write([Int32]$size); $writer.Write([Int32](2 * $size))
        $writer.Write([UInt16]1); $writer.Write([UInt16]32); $writer.Write([UInt32]0); $writer.Write([UInt32]$pixels)
        $writer.Write([Int32]0); $writer.Write([Int32]0); $writer.Write([UInt32]0); $writer.Write([UInt32]0)
        for ($row = $size - 1; $row -ge 0; $row--) { $writer.Write($raw, $row * $stride, $size * 4) }
        $writer.Write((New-Object byte[] $mask))
        $writer.Flush()
        [IO.File]::WriteAllBytes($Path, $out.ToArray())
    }

    # The icon both shortcuts use: GM1's own, or WSL's when it could not be drawn.
    function Get-Icon {
        $icon = Join-Path $AppFolder 'gm1.ico'
        try {
            if (-not (Test-Path $AppFolder)) { New-Item -ItemType Directory -Path $AppFolder -Force -ErrorAction Stop | Out-Null }
            Write-GM1Icon $icon
            return "$icon,0"
        } catch {
            return "$(Get-WslExe),0"
        }
    }

    function New-GM1Shortcut([string]$Path, [string]$Name, [string]$Icon) {
        $wsl = Get-WslExe
        $folder = Split-Path -Parent $Path
        if (-not (Test-Path $folder)) { New-Item -ItemType Directory -Path $folder -Force -ErrorAction Stop | Out-Null }
        $inside = 'export PATH=$HOME/.local/bin:$PATH; gpuwm-cm1 gui || { echo; echo GM1 stopped: read the lines above. Press Enter to close this window.; read _; }'
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut($Path)
        $link.TargetPath = $wsl
        $link.Arguments = "-d $Name --cd ~ --exec bash -lc `"$inside`""
        $link.WorkingDirectory = $env:USERPROFILE
        $link.Description = 'GM1: storm simulations on your NVIDIA graphics card'
        $link.IconLocation = $Icon
        $link.Save()
    }

    # GM1 check: the doctor's check in a window, and its report saved on the desktop and copied, for a help request.
    function New-CheckShortcut([string]$Path, [string]$Name, [string]$Icon) {
        if (-not (Test-Path $AppFolder)) { New-Item -ItemType Directory -Path $AppFolder -Force -ErrorAction Stop | Out-Null }
        $script = Join-Path $AppFolder 'gm1-check.ps1'
        $lines = @(
            '# GM1 check, written by the GM1 installer: checks this computer for GM1 and saves gm1-report.txt on the',
            '# desktop (also copied) for a help request.',
            "`$distro = '$Name'",
            "`$inside = 'export PATH=`$HOME/.local/bin:`$PATH; '",
            'try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false } catch { }',
            '$env:WSL_UTF8 = ''1''',
            'Write-Host ''GM1 check: can this computer run GM1?'' -ForegroundColor Cyan',
            'Write-Host ''''',
            '& wsl.exe -d $distro --cd ''~'' --exec bash -lc ($inside + ''gpuwm-cm1 doctor --plain'') | Out-Host',
            '$lines = & wsl.exe -d $distro --cd ''~'' --exec bash -lc ($inside + ''gpuwm-cm1 doctor --report 2>&1'') 2>&1',
            '$text = (@($lines) | ForEach-Object { ([string]$_) -replace "`0", '''' }) -join "`r`n"',
            '$file = Join-Path ([Environment]::GetFolderPath(''Desktop'')) ''gm1-report.txt''',
            '[IO.File]::WriteAllText($file, $text + "`r`n", (New-Object Text.UTF8Encoding $false))',
            '$copied = $true',
            'try { Set-Clipboard -Value $text -ErrorAction Stop } catch { $copied = $false }',
            'Write-Host ''''',
            'Write-Host "Saved the report on your desktop: $file" -ForegroundColor Cyan',
            'if ($copied) { Write-Host ''It is also copied: paste it into your message with Ctrl+V, or attach the file.'' }',
            'Write-Host ''Anything marked FIX above has the line under it saying what to do.''',
            '[void](Read-Host ''Press Enter to close this window'')'
        )
        [IO.File]::WriteAllText($script, ($lines -join "`r`n") + "`r`n", (New-Object Text.UTF8Encoding $true))
        $powershell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
        $folder = Split-Path -Parent $Path
        if (-not (Test-Path $folder)) { New-Item -ItemType Directory -Path $folder -Force -ErrorAction Stop | Out-Null }
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut($Path)
        $link.TargetPath = $powershell
        $link.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
        $link.WorkingDirectory = $env:USERPROFILE
        $link.Description = 'GM1 check: checks this computer for GM1 and saves gm1-report.txt on the desktop for a help request'
        $link.IconLocation = $Icon
        $link.Save()
    }

    function Remove-GM1Shortcuts {
        foreach ($path in @((Get-StartShortcut), (Get-DesktopShortcut), (Get-CheckShortcut))) {
            if (Test-Path $path) {
                try { Remove-Item -LiteralPath $path -Force -ErrorAction Stop; Say "Removed the shortcut $path" }
                catch { Warn "Could not remove the shortcut $path ($($_.Exception.Message))." }
            }
        }
        if (Test-Path $AppFolder) { Remove-Item -LiteralPath $AppFolder -Recurse -Force -ErrorAction SilentlyContinue }
    }

    # ------------------------------------------------------------ the steps

    if (-not $Distro -and $env:GM1_DISTRO) { $Distro = $env:GM1_DISTRO }
    if ($env:GM1_UNINSTALL -eq '1') { $Uninstall = $true }
    if ($env:GM1_DESKTOP_SHORTCUT -eq '0') { $NoDesktopShortcut = $true }

    Say 'GM1 installer for Windows'

    if ($Uninstall) {
        Step 'Uninstalling GM1: its shortcuts, then GM1 inside WSL'
        # The shortcuts first, so they go even when the step inside WSL cannot run.
        Remove-GM1Shortcuts
        $name = $null
        if (Get-WslExe) { $name = Select-Distro (Get-DistroNames) $Distro }
        if ($name) {
            $script = $null
            try { $script = Get-InstallSh } catch { $script = $null }
            if ($script) {
                $code = Invoke-InstallSh $name $script @('--uninstall')
            } else {
                # install.sh could not be downloaded: uv removes GM1 and its downloaded packages directly. (No
                # double quotes: Windows PowerShell 5.1 does not pass them through to wsl.exe intact.)
                $inline = 'u=$(command -v uv || echo none); [ -x $HOME/.local/bin/uv ] && u=$HOME/.local/bin/uv; if [ $u != none ] && $u tool uninstall gpuwm-cm1; then $u cache clean ' + $CachedPackages + ' >/dev/null 2>&1; echo GM1 is uninstalled.; else echo GM1 is not installed with uv here.; fi'
                & wsl.exe -d $name --cd '~' --exec bash -lc $inline | Out-Host
                $code = $LASTEXITCODE
            }
            if ($code -ne 0) { Warn "The uninstall inside $name did not finish (see above)." }
        }
        Say ''
        Say 'GM1 is uninstalled. WSL and Ubuntu stay installed, with your runs in ~/gm1-runs inside Ubuntu.'
        if ($name) {
            Say "To remove Ubuntu too, and everything in it including your storms, type in PowerShell:  wsl --unregister $name"
        }
        return
    }

    Step 'Checking Windows'
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        Stop-WithHelp 'GM1 needs a 64-bit Intel or AMD computer (x64); this one is not.'
        return
    }
    $build = [Environment]::OSVersion.Version.Build
    if ($build -lt 19044) {
        Stop-WithHelp 'GM1 needs Windows 11, or Windows 10 version 21H2 or newer, for the graphics card in WSL2. Update Windows (Settings, Windows Update).'
        return
    }
    Test-Driver

    Step 'Checking WSL2 and Ubuntu'
    $wslExe = Get-WslExe
    $wslPresent = $false
    $names = @()
    if ($wslExe) {
        $status = Invoke-Wsl @('--status')
        $wslVersion = Invoke-Wsl @('--version')
        $list = Invoke-Wsl @('--list', '--quiet')
        $wslPresent = ($status.Code -eq 0) -or ($wslVersion.Code -eq 0) -or ($list.Code -eq 0)
        if ($list.Code -eq 0) { $names = $list.Lines }
    }
    $name = Select-Distro $names $Distro
    if (-not $name) {
        if ($Distro) {
            Stop-WithHelp "There is no WSL distribution called $Distro (wsl --list shows the ones there are)."
            return
        }
        if (-not (Install-WslUbuntu $wslPresent)) { return }
        $name = Select-Distro (Get-DistroNames) ''
        if (-not $name) {
            Say ''
            Say 'Ubuntu is installed. One more step before GM1:'
            Say '  1. Open "Ubuntu" from the Start menu. It asks for a new user name and password (any you like;'
            Say '     the password is not shown while you type it).'
            Say '  2. Close the Ubuntu window, open PowerShell again and run the same line:'
            Say "     $OneLiner"
            return
        }
    }
    $version = Get-DistroVersion $name
    if ($version -eq '1') {
        Step "Switching $name to WSL2 (WSL 1 cannot use the graphics card; this takes a few minutes)"
        $run = Invoke-Wsl @('--set-version', $name, '2')
        if ($run.Code -ne 0) {
            Stop-WithHelp "Windows could not switch $name to WSL2. In PowerShell as administrator, run: wsl --set-version $name 2"
            return
        }
    }
    $user = Invoke-Wsl @('-d', $name, '--exec', 'id', '-u')
    if ($user.Code -ne 0) {
        Stop-WithHelp "$name did not start. Open it from the Start menu once, or run: wsl --update"
        return
    }
    if ($user.Lines.Count -gt 0 -and $user.Lines[0] -eq '0') {
        # No user of its own yet: Ubuntu's first start, here, then on with the install.
        Initialize-DistroUser $name
        $user = Invoke-Wsl @('-d', $name, '--exec', 'id', '-u')
        if ($user.Code -ne 0 -or ($user.Lines.Count -gt 0 -and $user.Lines[0] -eq '0')) {
            Stop-WithHelp "$name has no user of its own yet. Open `"$name`" from the Start menu, choose a user name and password, and close it."
            return
        }
    }
    Say "Using WSL2 with $name."

    Step "Installing GM1 inside $name"
    try {
        $script = Get-InstallSh
    } catch {
        Stop-WithHelp "Could not download the installer ($InstallShUrl): $($_.Exception.Message). Check the internet connection."
        return
    }
    $code = Invoke-InstallSh $name $script @()
    if ($code -ne 0) {
        Stop-WithHelp "The install inside $name stopped (see the lines above)."
        return
    }

    Step 'Adding the GM1 shortcuts'
    $icon = Get-Icon
    $made = @()
    $start = Get-StartShortcut
    New-GM1Shortcut $start $name $icon
    $made += $start
    $desktop = Get-DesktopShortcut
    if ($NoDesktopShortcut) {
        if (Test-Path $desktop) { Remove-Item -LiteralPath $desktop -Force -ErrorAction SilentlyContinue }
    } else {
        New-GM1Shortcut $desktop $name $icon
        $made += $desktop
    }
    $check = Get-CheckShortcut
    New-CheckShortcut $check $name $icon
    $made += $check
    foreach ($path in $made) { Say "Shortcut: $path" }

    Say ''
    Write-Host 'GM1 is installed.' -ForegroundColor Green
    if ($NoDesktopShortcut) {
        Say 'Open it: Start menu, GM1. A window opens that runs GM1, and your browser opens the GM1 page.'
    } else {
        Say 'Open it: Start menu, GM1 (or the GM1 icon on the desktop). A window opens that runs GM1, and your'
        Say 'browser opens the GM1 page.'
    }
    Say 'Closing that window never stops a running storm: open GM1 again to see it. Shutting down or restarting'
    Say 'Windows stops a storm; for long storms, set Windows not to sleep.'
    Say 'Something wrong? Start menu, GM1 check: it checks this computer and saves gm1-report.txt on your desktop'
    Say 'to attach when you ask for help.'
}

try {
    Invoke-GM1Installer -Uninstall $Uninstall.IsPresent -NoDesktopShortcut $NoDesktopShortcut.IsPresent -Distro $Distro
} catch {
    Write-Host ''
    Write-Host "GM1 installer stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Run the same line again. If it stops again, ask for help with the lines above.'
}
