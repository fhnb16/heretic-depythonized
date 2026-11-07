# Builds app.ico from app-256.png (Windows, PowerShell 7).
# Small sizes are stored as 32-bit BITMAPINFOHEADER DIBs with an AND mask and only the 256px
# image as PNG: rc.exe (used by tauri-build on Windows) rejects PNG entries below 256px with
# "error RC2176: old DIB".
param(
    [string]$Source = (Join-Path $PSScriptRoot "app-256.png"),
    [string]$Output = (Join-Path $PSScriptRoot "app.ico"),
    [int[]]$Sizes = @(16, 24, 32, 48, 64)
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$sourceImage = [System.Drawing.Bitmap]::FromFile((Resolve-Path $Source).Path)
$images = [System.Collections.Generic.List[object]]::new()

foreach ($size in $Sizes) {
    $bitmap = [System.Drawing.Bitmap]::new($size, $size, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
    $graphics.Clear([System.Drawing.Color]::Transparent)
    $graphics.DrawImage($sourceImage, 0, 0, $size, $size)
    $graphics.Dispose()

    $stream = [System.IO.MemoryStream]::new()
    $writer = [System.IO.BinaryWriter]::new($stream)
    # BITMAPINFOHEADER; the height counts the XOR and AND bitmaps.
    $writer.Write([uint32]40)
    $writer.Write([int32]$size)
    $writer.Write([int32]($size * 2))
    $writer.Write([uint16]1)
    $writer.Write([uint16]32)
    $writer.Write([uint32]0)
    $writer.Write([uint32]0)
    $writer.Write([int32]0)
    $writer.Write([int32]0)
    $writer.Write([uint32]0)
    $writer.Write([uint32]0)
    # XOR bitmap: BGRA, bottom-up.
    for ($y = $size - 1; $y -ge 0; $y--) {
        for ($x = 0; $x -lt $size; $x++) {
            $c = $bitmap.GetPixel($x, $y)
            $writer.Write([byte]$c.B)
            $writer.Write([byte]$c.G)
            $writer.Write([byte]$c.R)
            $writer.Write([byte]$c.A)
        }
    }
    # AND mask: 1 bit per pixel, rows padded to 4 bytes; transparent pixels set.
    $rowBytes = [int]([math]::Ceiling($size / 32) * 4)
    for ($y = $size - 1; $y -ge 0; $y--) {
        $row = [byte[]]::new($rowBytes)
        for ($x = 0; $x -lt $size; $x++) {
            if ($bitmap.GetPixel($x, $y).A -lt 128) { $row[[int][math]::Floor($x / 8)] = $row[[int][math]::Floor($x / 8)] -bor (0x80 -shr ($x % 8)) }
        }
        $writer.Write($row)
    }
    $writer.Flush()
    $images.Add([pscustomobject]@{ Size = $size; Bytes = $stream.ToArray() })
    $bitmap.Dispose()
}
$sourceImage.Dispose()
$images.Add([pscustomobject]@{ Size = 256; Bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $Source).Path) })

$out = [System.IO.MemoryStream]::new()
$w = [System.IO.BinaryWriter]::new($out)
$w.Write([uint16]0)
$w.Write([uint16]1)
$w.Write([uint16]$images.Count)
$offset = 6 + 16 * $images.Count
foreach ($image in $images) {
    $dimension = if ($image.Size -ge 256) { 0 } else { $image.Size }
    $w.Write([byte]$dimension)
    $w.Write([byte]$dimension)
    $w.Write([byte]0)
    $w.Write([byte]0)
    $w.Write([uint16]1)
    $w.Write([uint16]32)
    $w.Write([uint32]$image.Bytes.Length)
    $w.Write([uint32]$offset)
    $offset += $image.Bytes.Length
}
foreach ($image in $images) { $w.Write($image.Bytes) }
$w.Flush()
[System.IO.File]::WriteAllBytes($Output, $out.ToArray())
Write-Host "Wrote $Output ($($images.Count) images, $($out.Length) bytes)"
