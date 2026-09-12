@echo off
rem 启动 Label Studio（本地文件服务）。
rem
rem 路径全部相对脚本所在目录解析，不再硬编码某台机器的绝对路径：
rem   LABEL_STUDIO_DOC_ROOT   本地图片根目录，默认 <repo>\data\raw_images
rem   LABEL_STUDIO_PYTHON     可用的 python 解释器，默认项目 venv 里的 python.exe
rem
rem 注意：label-studio 依赖的 django-environ 使用了 Python 3.12 起移除的
rem pkgutil.find_loader，在 Python 3.14 下无法启动。如需标注功能，
rem 请在 Python <= 3.12 的环境里安装 label-studio，并用 LABEL_STUDIO_PYTHON 指定。

setlocal
set "REPO=%~dp0.."
if "%LABEL_STUDIO_DOC_ROOT%"=="" set "LABEL_STUDIO_DOC_ROOT=%REPO%\data\raw_images"
if "%LABEL_STUDIO_PYTHON%"=="" set "LABEL_STUDIO_PYTHON=%REPO%\drone_rescue_env\Scripts\python.exe"
if not exist "%LABEL_STUDIO_DOC_ROOT%" mkdir "%LABEL_STUDIO_DOC_ROOT%"

set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%LABEL_STUDIO_DOC_ROOT%

echo [run_label_studio] python   : %LABEL_STUDIO_PYTHON%
echo [run_label_studio] doc root : %LABEL_STUDIO_DOC_ROOT%

"%LABEL_STUDIO_PYTHON%" -c "from label_studio.server import main; main()" %*
endlocal
