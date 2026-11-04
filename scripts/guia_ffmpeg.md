# Guia de Instalacion de FFmpeg en Windows

FFmpeg es necesario para la extraccion y conversion de pistas sonoras desde archivos de video y enlaces web.

## Instalacion con Winget

Abre una terminal de PowerShell como administrador o usuario estandar y ejecuta:

```powershell
winget install Gyan.FFmpeg
```

Cierra y vuelve a abrir tu terminal para que el sistema actualice la variable PATH.

## Comprobacion

Verifica que el binario responda ejecutando:

```powershell
ffmpeg -version
```
