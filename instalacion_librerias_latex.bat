@echo off
setlocal EnableDelayedExpansion

REM === CONFIGURACIÓN ===
set TEXFILE=estructura_tesis.tex
set TEXNAME=estructura_tesis
set DEST=utils
set MAX_INTENTOS=10

REM === Crear carpeta de salida si no existe ===
if not exist "%DEST%" (
    mkdir "%DEST%"
)

REM === Intentar compilar y detectar paquetes faltantes ===
set /a intentos=0

:reintentar
set /a intentos+=1
echo --- Intento de compilación #%intentos% ---
pdflatex -interaction=nonstopmode -halt-on-error %TEXFILE% > salida.log

set FOUND=

REM === Detectar errores de paquete faltante ===
for /f "tokens=*" %%A in ('findstr /C:"LaTeX Error: File `" salida.log') do (
    for /f "tokens=3 delims=`\"" %%B in ("%%A") do (
        set "PKGNAME=%%~B"
        set "PKGNAME=!PKGNAME:.sty=!"
        echo → Instalando paquete: !PKGNAME!
        mpm --install !PKGNAME! > nul 2>&1
        set FOUND=1
    )
)

if defined FOUND (
    if !intentos! lss %MAX_INTENTOS% goto :reintentar
)

REM === Segunda pasada para generar .bcf y .toc ===
echo --- Segunda pasada con pdflatex ---
pdflatex -interaction=nonstopmode -halt-on-error %TEXFILE% > nul

REM === Ejecutar Biber después de generar .bcf ===
echo --- Ejecutando Biber ---
biber %TEXNAME% > nul

REM === Dos pasadas adicionales para referencias cruzadas y TOC ===
echo --- Tercera pasada con pdflatex ---
pdflatex -interaction=nonstopmode -halt-on-error %TEXFILE% > nul

echo --- Cuarta pasada con pdflatex ---
pdflatex -interaction=nonstopmode -halt-on-error %TEXFILE% > nul

REM === Mover auxiliares a carpeta utils ===
echo --- Moviendo archivos auxiliar
