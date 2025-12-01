@echo off
setlocal EnableDelayedExpansion

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    set "IS_ADMIN=1"
) else (
    set "IS_ADMIN=0"
)

:: 1. Define work_path as the directory where this bat file is located
set "work_path=%~dp0"
set "work_path=%work_path:~0,-1%"

echo ========================================
echo Open WebUI Setup and Launch Script
echo ========================================
echo Work Path: %work_path%
if "%IS_ADMIN%"=="1" (
    echo Running as: Administrator
) else (
    echo Running as: Standard User
)
echo.

:: 2. Check if open-webui directory exists
if exist "%work_path%\open-webui\" (
    echo [INFO] Already installed, directly execute application
) else (
    echo [INFO] open-webui directory not found, checking for zip file...
    
    :: 2.2. Check if open-webui.zip exists
    if exist "%work_path%\open-webui.zip" (
        echo [INFO] Found open-webui.zip, starting installation...
        
        :: 2.2.1.1. Extract open-webui.zip
        echo [INFO] Extracting open-webui.zip...
        powershell -Command "Expand-Archive -Path '%work_path%\open-webui.zip' -DestinationPath '%work_path%\open-webui' -Force"
        if errorlevel 1 (
            echo [ERROR] Failed to extract open-webui.zip
            pause
            exit /b 1
        )
        echo [INFO] Extraction completed
        echo.
    ) else (
        :: 2.2.2. open-webui.zip not found
        echo [ERROR] The open-webui.zip is missing
        echo [ERROR] Please place open-webui.zip in the same directory as this script
        pause
        exit /b 1
    )
)

:: 2.5. Check Python, uv, and virtual environments
echo.
echo ========================================
echo Environment Setup
echo ========================================
echo.

:: 2.5.1. Check if Python 3.11 is installed
echo [INFO] Checking for Python 3.11...
python --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Python command not found in PATH.
    goto :PYTHON_NOT_FOUND
)

:: Python exists, check version
python --version 2>&1 | findstr /C:"Python 3.11" >nul
if errorlevel 1 (
    :: Display current version
    for /f "tokens=*" %%V in ('python --version 2^>^&1') do set "PYTHON_VER=%%V"
    echo [WARNING] Python 3.11 not found. Current version: !PYTHON_VER!
    goto :PYTHON_NOT_FOUND
)

:: Python 3.11 is installed
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "PYTHON_VER=%%V"
echo [INFO] Found Python version: !PYTHON_VER!
echo [INFO] Python 3.11 is already installed
goto :SKIP_PYTHON_INSTALL

:PYTHON_NOT_FOUND
echo.
    
    echo [INFO] Python 3.11 needs to be installed or is not detected in PATH.
    echo [INFO] You have four options:
    echo.
    echo   1. Install Python for all users (Recommended)
    if "%IS_ADMIN%"=="0" (
        echo      - Requires administrator privileges ^(will restart^)
    ) else (
        echo      - Using administrator privileges
    )
    echo      - Better system integration
    echo.
    echo   2. Install Python for current user only
    echo      - No admin rights required
    echo      - Works only for your account
    echo.
    echo   3. Skip Python installation (Python already installed manually)
    echo      - Continue to next step
    echo      - Use this if you have already installed Python 3.11
    echo.
    echo   4. Cancel and exit
    echo.
    set /p "INSTALL_CHOICE=Please choose (1/2/3/4): "
    
    if "!INSTALL_CHOICE!"=="1" (
        if "%IS_ADMIN%"=="0" (
            echo [INFO] Restarting with administrator privileges...
            powershell -Command "Start-Process '%~f0' -Verb RunAs"
            exit /b 0
        )
        echo [INFO] Installing Python for all users...
        set "INSTALL_FOR_ALL_USERS=1"
        goto :INSTALL_PYTHON
    )
    
    if "!INSTALL_CHOICE!"=="2" (
        echo [INFO] Installing Python for current user...
        set "INSTALL_FOR_ALL_USERS=0"
        goto :INSTALL_PYTHON
    )
    
    if "!INSTALL_CHOICE!"=="3" (
        echo [INFO] Skipping Python installation...
        echo [WARNING] This script requires Python 3.11 to run properly.
        echo [WARNING] If you encounter errors, please install Python 3.11 and add it to PATH.
        echo.
        goto :SKIP_PYTHON_INSTALL
    )
    
    if "!INSTALL_CHOICE!"=="4" (
        echo [INFO] Installation cancelled.
        echo [INFO] Please download and install Python 3.11 from: https://www.python.org/downloads/
        echo [INFO] After installation, run this script again.
        pause
        exit /b 1
    )
    
    :: Invalid choice
    echo [ERROR] Invalid choice. Please run the script again.
    pause
    exit /b 1
    
    :INSTALL_PYTHON
    echo [INFO] Attempting to download and install Python 3.11...
    echo.
    
    :: Detect Windows architecture
    if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
        set "PYTHON_ARCH=amd64"
        set "PYTHON_INSTALLER=python-3.11.9-amd64.exe"
    )
    if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
        set "PYTHON_ARCH=arm64"
        set "PYTHON_INSTALLER=python-3.11.9-arm64.exe"
    )
    if not "%PROCESSOR_ARCHITECTURE%"=="AMD64" if not "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
        set "PYTHON_ARCH=win32"
        set "PYTHON_INSTALLER=python-3.11.9.exe"
    )
    
    set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/!PYTHON_INSTALLER!"
    set "PYTHON_INSTALLER_PATH=%work_path%\!PYTHON_INSTALLER!"
    
    echo [INFO] Detected architecture: !PYTHON_ARCH!
    echo [INFO] Downloading Python 3.11.9 from !PYTHON_URL!...
    echo [INFO] This may take a few minutes...
    
    :: Download Python installer using PowerShell
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '!PYTHON_URL!' -OutFile '!PYTHON_INSTALLER_PATH!' -UseBasicParsing}"
    
    if errorlevel 1 (
        echo [ERROR] Failed to download Python installer
        echo [INFO] Please download and install Python 3.11 manually from: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    
    echo [INFO] Download completed successfully
    
    if "!INSTALL_FOR_ALL_USERS!"=="1" (
        echo [INFO] Installing Python 3.11 for all users...
        "!PYTHON_INSTALLER_PATH!" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0
    ) else (
        echo [INFO] Installing Python 3.11 for current user...
        "!PYTHON_INSTALLER_PATH!" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0
    )
    
    :: Wait for installation to complete
    echo [INFO] Waiting for installation to complete...
    :: Use ping to wait 15 seconds (doesn't require PATH)
    ping 127.0.0.1 -n 16 >nul 2>&1
    
    :: Clean up installer
    if exist "!PYTHON_INSTALLER_PATH!" (
        del "!PYTHON_INSTALLER_PATH!"
        echo [INFO] Cleaned up installer file
    )
    
    :: Refresh environment variables
    echo [INFO] Refreshing environment variables...
    call :RefreshEnv
    
    :: Verify installation
    python --version 2>&1 | findstr /C:"Python 3.11" >nul
    if errorlevel 1 (
        echo [ERROR] Python 3.11 installation verification failed
        echo [INFO] The installation completed but Python is not yet available in PATH
        echo [INFO] Please try one of the following:
        echo.
        echo   Option 1: Close this window and run the script again
        echo   Option 2: Open a new command prompt and run the script
        echo   Option 3: Log out and log back in to refresh environment
        echo.
        pause
        exit /b 1
    )
    
    echo [INFO] Python 3.11 installed successfully!
    echo.

:SKIP_PYTHON_INSTALL
:: 2.5.2. Check if uv is installed
echo [INFO] Checking for uv...
uv --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] uv not found, installing uv...
    
    :: Check if python is available
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python is not available in PATH. Cannot install uv.
        echo [INFO] Please ensure Python 3.11 is installed and added to PATH.
        echo [INFO] Then run this script again.
        pause
        exit /b 1
    )
    
    python -m pip install uv
    if errorlevel 1 (
        echo [ERROR] Failed to install uv
        pause
        exit /b 1
    )
    echo [INFO] uv installed successfully
) else (
    echo [INFO] uv is already installed
)

:: 2.5.3. Create venv for backend
echo [INFO] Setting up backend virtual environment...

:: Verify Python is available before creating venv
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not available in PATH. Cannot create virtual environment.
    echo [INFO] Please ensure Python 3.11 is installed and added to PATH.
    echo [INFO] Then run this script again.
    pause
    exit /b 1
)

cd /d "%work_path%\open-webui\backend"
if not exist "venv_open_webui\" (
    echo [INFO] Virtual environment not found, creating new one...
    python -m venv venv_open_webui
    if errorlevel 1 (
        echo [ERROR] Failed to create venv_open_webui
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists, will update dependencies
)

echo [INFO] Checking and installing backend requirements...
call venv_open_webui\Scripts\activate.bat
if not errorlevel 1 (
    :: Set npm environment variables to handle SSL certificate issues
    set "NPM_CONFIG_STRICT_SSL=false"
    set "NODE_TLS_REJECT_UNAUTHORIZED=0"
    
    if exist "requirements.txt" (
        echo [INFO] Installing from requirements.txt...
        pip install -r requirements.txt
    ) else (
        if exist "requirement.txt" (
            echo [INFO] Installing from requirement.txt...
            pip install -r requirement.txt
        ) else (
            echo [WARNING] No requirements.txt or requirement.txt found in backend
        )
    )
    call deactivate
    echo [INFO] Backend setup completed
) else (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

:: 2.5.4. Create venv for km
echo [INFO] Setting up km virtual environment...
cd /d "%work_path%\open-webui\km"
if not exist "venv_km\" (
    echo [INFO] Virtual environment not found, creating new one...
    python -m venv venv_km
    if errorlevel 1 (
        echo [ERROR] Failed to create venv_km
        echo [INFO] Please ensure Python 3.11 is properly installed.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists, will update dependencies
)

echo [INFO] Checking and installing km requirements...
call venv_km\Scripts\activate.bat
if not errorlevel 1 (
    if exist "requirements.txt" (
        echo [INFO] Installing from requirements.txt...
        pip install -r requirements.txt
    ) else (
        if exist "requirement.txt" (
            echo [INFO] Installing from requirement.txt...
            pip install -r requirement.txt
        ) else (
            echo [WARNING] No requirements.txt or requirement.txt found in km
        )
    )
    call deactivate
    echo [INFO] km setup completed
) else (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

cd /d "%work_path%"
echo.
echo [INFO] Environment setup completed successfully!

:: 3. Get environment variables from user
:GET_ENV_VARS
echo.
echo ========================================
echo Environment Configuration
echo ========================================
echo.

:: 3.1. LLM Model Setting
echo ===== LLM Model Setting =====
echo.

:INPUT_LLM_API_PORT
set /p "LLM_API_PORT=LLM_API_PORT (Hint: The port you start the LLM Model. e.g. 13141): "
if "%LLM_API_PORT%"=="" (
    echo [ERROR] LLM_API_PORT cannot be empty
    goto :INPUT_LLM_API_PORT
)

:INPUT_LLM_MODEL_NAME
echo.
set /p "LLM_MODEL_NAME=LLM_MODEL_NAME (Hint: You can find it in http://localhost:%LLM_API_PORT%/models.): "
if "%LLM_MODEL_NAME%"=="" (
    echo [ERROR] LLM_MODEL_NAME cannot be empty
    goto :INPUT_LLM_MODEL_NAME
)

:: 3.1.3. Verify LLM API connection
echo.
echo [INFO] Testing LLM API connection...
echo [INFO] Endpoint: http://localhost:%LLM_API_PORT%/v1/chat/completions
echo [INFO] Model: %LLM_MODEL_NAME%
echo [INFO] Please wait, this may take a few seconds...

:: Refresh environment variables before testing (in case Python was just installed)
call :RefreshEnv

:: Check if PowerShell is available using full path
set "PS_PATH=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_PATH%" (
    echo [WARNING] PowerShell not found. Skipping API connection test.
    echo [INFO] You can manually verify the connection later.
    echo.
    goto :SKIP_LLM_ERROR
)

:: Use PowerShell with full path and create a script file to avoid command line length issues
set "TEMP_SCRIPT=%TEMP%\llm_test_%RANDOM%.ps1"
(
    echo $ErrorActionPreference = 'Stop'
    echo try {
    echo     $body = @{
    echo         model = '%LLM_MODEL_NAME%'
    echo         messages = @(@{ role = 'user'; content = 'What is the capital of France?' }^)
    echo         temperature = 0
    echo         max_tokens = 2
    echo         cache_prompt = $true
    echo     } ^| ConvertTo-Json -Depth 10
    echo     $response = Invoke-RestMethod -Uri 'http://localhost:%LLM_API_PORT%/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 30
    echo     exit 0
    echo } catch {
    echo     exit 1
    echo }
) > "%TEMP_SCRIPT%"

"%PS_PATH%" -ExecutionPolicy Bypass -NoProfile -File "%TEMP_SCRIPT%" 2>nul
set LLM_TEST_ERROR=%ERRORLEVEL%
del "%TEMP_SCRIPT%" 2>nul

:: If error 9009 (command not found), refresh environment and retry once
if !LLM_TEST_ERROR! EQU 9009 (
    echo [INFO] Environment may need refresh, retrying...
    :: Use ping to wait 2 seconds (doesn't require PATH)
    ping 127.0.0.1 -n 3 >nul 2>&1
    call :RefreshEnv
    set "TEMP_SCRIPT=%TEMP%\llm_test_retry_%RANDOM%.ps1"
    (
        echo $ErrorActionPreference = 'Stop'
        echo try {
        echo     $body = @{
        echo         model = '%LLM_MODEL_NAME%'
        echo         messages = @(@{ role = 'user'; content = 'What is the capital of France?' }^)
        echo         temperature = 0
        echo         max_tokens = 2
        echo         cache_prompt = $true
        echo     } ^| ConvertTo-Json -Depth 10
        echo     $response = Invoke-RestMethod -Uri 'http://localhost:%LLM_API_PORT%/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 30
        echo     exit 0
        echo } catch {
        echo     exit 1
        echo }
    ) > "%TEMP_SCRIPT%"
    "%PS_PATH%" -ExecutionPolicy Bypass -NoProfile -File "%TEMP_SCRIPT%" 2>nul
    set LLM_TEST_ERROR=%ERRORLEVEL%
    del "%TEMP_SCRIPT%" 2>nul
)

echo.
if !LLM_TEST_ERROR! EQU 0 (
    echo [SUCCESS] LLM API connection verified successfully!
    echo.
    goto :SKIP_LLM_ERROR
)

echo [ERROR] ========================================
echo [ERROR] Failed to connect to LLM API
echo [ERROR] ========================================
echo [ERROR] Endpoint: http://localhost:%LLM_API_PORT%/v1/chat/completions
echo [ERROR] Model: %LLM_MODEL_NAME%
echo [ERROR] Error Code: !LLM_TEST_ERROR!
if !LLM_TEST_ERROR! EQU 9009 (
    echo.
    echo [WARNING] Error 9009: Command not found
    echo [INFO] This usually happens when environment variables are not yet updated.
    echo [INFO] The API test was skipped. You can manually verify the connection later.
    echo [INFO] The script will continue without API verification.
    echo.
    goto :SKIP_LLM_ERROR
)
echo.
echo [INFO] Please check:
echo        1. LLM server is running on port %LLM_API_PORT%
echo        2. Model name is correct (check http://localhost:%LLM_API_PORT%/v1/models)
echo        3. Server is accessible from this machine
echo        4. Firewall or antivirus is not blocking the connection
echo.
echo [INFO] To see detailed error, run this command manually:
echo        powershell -Command "Invoke-RestMethod -Uri 'http://localhost:%LLM_API_PORT%/v1/chat/completions' -Method Post -ContentType 'application/json' -Body '{\"model\":\"%LLM_MODEL_NAME%\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}'"
echo.
set /p "RETRY_LLM=Do you want to re-enter LLM settings? (y/n): "
if /i "!RETRY_LLM!"=="y" (
    echo.
    goto :INPUT_LLM_API_PORT
)
echo [WARNING] Continuing without verification...
echo.

:SKIP_LLM_ERROR

:: 3.2. Embedding Model Setting
echo.
echo ===== Embedding Model Setting =====
echo.

:INPUT_EMBEDDING_URL
set /p "EMBEDDING_URL=EMBEDDING_URL (Hint: The endpoint you start the Embedding Model. e.g. http://localhost:13142/v1): "
if "%EMBEDDING_URL%"=="" (
    echo [ERROR] EMBEDDING_URL cannot be empty
    goto :INPUT_EMBEDDING_URL
)

:INPUT_EMBEDDING_MODEL_NAME
echo.
set /p "EMBEDDING_MODEL_NAME=EMBEDDING_MODEL_NAME (Hint: You can find it in %EMBEDDING_URL%.): "
if "%EMBEDDING_MODEL_NAME%"=="" (
    echo [ERROR] EMBEDDING_MODEL_NAME cannot be empty
    goto :INPUT_EMBEDDING_MODEL_NAME
)

:: 3.2.3. Verify Embedding API connection
echo.
echo [INFO] Testing Embedding API connection...
:: Check if EMBEDDING_URL ends with slash
set "EMBEDDING_TEST_URL=%EMBEDDING_URL%"
if "!EMBEDDING_TEST_URL:~-1!"=="/" (
    set "EMBEDDING_TEST_URL=!EMBEDDING_TEST_URL!embeddings"
) else (
    set "EMBEDDING_TEST_URL=!EMBEDDING_TEST_URL!/embeddings"
)
echo [INFO] Endpoint: !EMBEDDING_TEST_URL!
echo [INFO] Model: %EMBEDDING_MODEL_NAME%
echo [INFO] Please wait, this may take a few seconds...

:: Refresh environment variables before testing (in case Python was just installed)
call :RefreshEnv

:: Check if PowerShell is available using full path
set "PS_PATH=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_PATH%" (
    echo [WARNING] PowerShell not found. Skipping API connection test.
    echo [INFO] You can manually verify the connection later.
    echo.
    goto :SKIP_EMBEDDING_ERROR
)

:: Use PowerShell with full path and create a script file to avoid command line length issues
set "TEMP_SCRIPT=%TEMP%\embedding_test_%RANDOM%.ps1"
(
    echo $ErrorActionPreference = 'Stop'
    echo try {
    echo     $body = @{
    echo         model = '%EMBEDDING_MODEL_NAME%'
    echo         input = 'What is the capital of France?'
    echo     } ^| ConvertTo-Json -Depth 10
    echo     $response = Invoke-RestMethod -Uri '!EMBEDDING_TEST_URL!' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 30
    echo     exit 0
    echo } catch {
    echo     exit 1
    echo }
) > "%TEMP_SCRIPT%"

"%PS_PATH%" -ExecutionPolicy Bypass -NoProfile -File "%TEMP_SCRIPT%" 2>nul
set EMBEDDING_TEST_ERROR=%ERRORLEVEL%
del "%TEMP_SCRIPT%" 2>nul

:: If error 9009 (command not found), refresh environment and retry once
if !EMBEDDING_TEST_ERROR! EQU 9009 (
    echo [INFO] Environment may need refresh, retrying...
    :: Use ping to wait 2 seconds (doesn't require PATH)
    ping 127.0.0.1 -n 3 >nul 2>&1
    call :RefreshEnv
    set "TEMP_SCRIPT=%TEMP%\embedding_test_retry_%RANDOM%.ps1"
    (
        echo $ErrorActionPreference = 'Stop'
        echo try {
        echo     $body = @{
        echo         model = '%EMBEDDING_MODEL_NAME%'
        echo         input = 'What is the capital of France?'
        echo     } ^| ConvertTo-Json -Depth 10
        echo     $response = Invoke-RestMethod -Uri '!EMBEDDING_TEST_URL!' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 30
        echo     exit 0
        echo } catch {
        echo     exit 1
        echo }
    ) > "%TEMP_SCRIPT%"
    "%PS_PATH%" -ExecutionPolicy Bypass -NoProfile -File "%TEMP_SCRIPT%" 2>nul
    set EMBEDDING_TEST_ERROR=%ERRORLEVEL%
    del "%TEMP_SCRIPT%" 2>nul
)

echo.
if !EMBEDDING_TEST_ERROR! EQU 0 (
    echo [SUCCESS] Embedding API connection verified successfully!
    echo.
    goto :SKIP_EMBEDDING_ERROR
)

echo [ERROR] ========================================
echo [ERROR] Failed to connect to Embedding API
echo [ERROR] ========================================
echo [ERROR] Endpoint: !EMBEDDING_TEST_URL!
echo [ERROR] Model: %EMBEDDING_MODEL_NAME%
echo [ERROR] Error Code: !EMBEDDING_TEST_ERROR!
if !EMBEDDING_TEST_ERROR! EQU 9009 (
    echo.
    echo [WARNING] Error 9009: Command not found
    echo [INFO] This usually happens when environment variables are not yet updated.
    echo [INFO] The API test was skipped. You can manually verify the connection later.
    echo [INFO] The script will continue without API verification.
    echo.
    goto :SKIP_EMBEDDING_ERROR
)
echo.
echo [INFO] Please check:
echo        1. Embedding server is running
echo        2. EMBEDDING_URL is correct: %EMBEDDING_URL%
echo        3. Model name is correct
echo        4. Server is accessible from this machine
echo        5. Firewall or antivirus is not blocking the connection
echo.
echo [INFO] To see detailed error, run this command manually:
echo        powershell -Command "Invoke-RestMethod -Uri '!EMBEDDING_TEST_URL!' -Method Post -ContentType 'application/json' -Body '{\"model\":\"%EMBEDDING_MODEL_NAME%\",\"input\":\"test\"}'"
echo.
set /p "RETRY_EMBEDDING=Do you want to re-enter Embedding settings? (y/n): "
if /i "!RETRY_EMBEDDING!"=="y" (
    echo.
    goto :INPUT_EMBEDDING_URL
)
echo [WARNING] Continuing without verification...
echo.

:SKIP_EMBEDDING_ERROR

:: 3.3. KM Setting
echo.
echo ===== KM Setting =====
echo.

:INPUT_API_PORT
set /p "API_PORT=API_PORT (Hint: The port you want to start KM. Should be unused. e.g. 18299): "
if "%API_PORT%"=="" (
    echo [ERROR] API_PORT cannot be empty
    goto :INPUT_API_PORT
)
:: Check if port is in use
where netstat >nul 2>&1
if not errorlevel 1 (
    netstat -ano 2>nul | findstr /C:":%API_PORT% " >nul 2>&1
    if not errorlevel 1 (
        echo [WARNING] Port %API_PORT% is already in use!
        set /p "CONTINUE_API_PORT=Do you want to use this port anyway? (y/n): "
        if /i not "!CONTINUE_API_PORT!"=="y" goto :INPUT_API_PORT
    )
)

:INPUT_MAX_TOKENS_PER_GROUP
echo.
set /p "MAX_TOKENS_PER_GROUP=MAX_TOKENS_PER_GROUP (Hint: The token length each group, should be smaller than context size of LLM. e.g. 13000): "
if "%MAX_TOKENS_PER_GROUP%"=="" (
    echo [ERROR] MAX_TOKENS_PER_GROUP cannot be empty
    goto :INPUT_MAX_TOKENS_PER_GROUP
)

:INPUT_LLM_GGUF
echo.
set /p "LLM_GGUF=LLM_GGUF (Hint: The file name of GGUF LLM weight file. e.g. Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf): "
if "%LLM_GGUF%"=="" (
    echo [ERROR] LLM_GGUF cannot be empty
    goto :INPUT_LLM_GGUF
)

:INPUT_LLM_MODEL_DIR
echo.
set /p "LLM_MODEL_DIR=LLM_MODEL_DIR (Hint: The path of GGUF LLM weight file.. e.g. C:\Program Files\models): "
if "%LLM_MODEL_DIR%"=="" (
    echo [ERROR] LLM_MODEL_DIR cannot be empty
    goto :INPUT_LLM_MODEL_DIR
)

:: 3.4. OpenWebUI Setting
echo.
echo ===== OpenWebUI Setting =====
echo.

:INPUT_PORT
set /p "PORT=PORT (Hint: The port you want to start OpenWebUI. Should be unused. e.g. 8080): "
if "%PORT%"=="" (
    echo [ERROR] PORT cannot be empty
    goto :INPUT_PORT
)
:: Check if port is in use
where netstat >nul 2>&1
if not errorlevel 1 (
    netstat -ano 2>nul | findstr /C:":%PORT% " >nul 2>&1
    if not errorlevel 1 (
        echo [WARNING] Port %PORT% is already in use!
        set /p "CONTINUE_PORT=Do you want to use this port anyway? (y/n): "
        if /i not "!CONTINUE_PORT!"=="y" goto :INPUT_PORT
    )
)

:: 4. Set environment variables
echo.
echo ========================================
echo Setting Environment Variables
echo ========================================
echo.

set "OPENAI_API_BASE_URL=http://localhost:%LLM_API_PORT%/v1"
set "OPEN_WEBUI_DIR=%work_path%\open-webui\backend\data\parse_txt"
set "KM_RESULT_DIR=%work_path%\open-webui\km"
set "KM_SELF_RAG_API_BASE_URL=http://localhost:%API_PORT%"

echo LLM_API_PORT=%LLM_API_PORT%
echo LLM_MODEL_NAME=%LLM_MODEL_NAME%
echo EMBEDDING_URL=%EMBEDDING_URL%
echo EMBEDDING_MODEL_NAME=%EMBEDDING_MODEL_NAME%
echo API_PORT=%API_PORT%
echo MAX_TOKENS_PER_GROUP=%MAX_TOKENS_PER_GROUP%
echo LLM_GGUF=%LLM_GGUF%
echo LLM_MODEL_DIR=%LLM_MODEL_DIR%
echo OPENAI_API_BASE_URL=%OPENAI_API_BASE_URL%
echo OPEN_WEBUI_DIR=%OPEN_WEBUI_DIR%
echo KM_RESULT_DIR=%KM_RESULT_DIR%
echo KM_SELF_RAG_API_BASE_URL=%KM_SELF_RAG_API_BASE_URL%
echo PORT=%PORT%
echo.

:: 5. Launch services
echo ========================================
echo Starting Services
echo ========================================
echo.

:: Create log directory
if not exist "%work_path%\logs" mkdir "%work_path%\logs"

:: 5.1. Start Open WebUI backend
echo [INFO] Starting Open WebUI backend on port %PORT%...
start "Open WebUI Backend" cmd /k "cd /d "%work_path%\open-webui\backend" && call venv_open_webui\Scripts\activate.bat && set "VIRTUAL_ENV=%work_path%\open-webui\backend\venv_open_webui" && set "LLM_API_PORT=%LLM_API_PORT%" && set "LLM_MODEL_NAME=%LLM_MODEL_NAME%" && set "EMBEDDING_URL=%EMBEDDING_URL%" && set "EMBEDDING_MODEL_NAME=%EMBEDDING_MODEL_NAME%" && set "API_PORT=%API_PORT%" && set "MAX_TOKENS_PER_GROUP=%MAX_TOKENS_PER_GROUP%" && set "LLM_GGUF=%LLM_GGUF%" && set "LLM_MODEL_DIR=%LLM_MODEL_DIR%" && set "OPENAI_API_BASE_URL=%OPENAI_API_BASE_URL%" && set "OPEN_WEBUI_DIR=%OPEN_WEBUI_DIR%" && set "KM_RESULT_DIR=%KM_RESULT_DIR%" && set "KM_SELF_RAG_API_BASE_URL=%KM_SELF_RAG_API_BASE_URL%" && set "NPM_CONFIG_STRICT_SSL=false" && set "NODE_TLS_REJECT_UNAUTHORIZED=0" && uv run --no-project uvicorn open_webui.main:app --port %PORT% --host 0.0.0.0 --reload"

:: Wait a moment before starting the second service
:: Use ping to wait 3 seconds (doesn't require PATH)
ping 127.0.0.1 -n 4 >nul 2>&1

:: 5.2. Start KM service
echo [INFO] Starting KM service on port %API_PORT%...
start "KM Service" cmd /k "cd /d "%work_path%\open-webui\km" && call venv_km\Scripts\activate.bat && set "VIRTUAL_ENV=%work_path%\open-webui\km\venv_km" && set "LLM_API_PORT=%LLM_API_PORT%" && set "LLM_MODEL_NAME=%LLM_MODEL_NAME%" && set "EMBEDDING_URL=%EMBEDDING_URL%" && set "EMBEDDING_MODEL_NAME=%EMBEDDING_MODEL_NAME%" && set "API_PORT=%API_PORT%" && set "MAX_TOKENS_PER_GROUP=%MAX_TOKENS_PER_GROUP%" && set "LLM_GGUF=%LLM_GGUF%" && set "LLM_MODEL_DIR=%LLM_MODEL_DIR%" && set "OPENAI_API_BASE_URL=%OPENAI_API_BASE_URL%" && set "OPEN_WEBUI_DIR=%OPEN_WEBUI_DIR%" && set "KM_RESULT_DIR=%KM_RESULT_DIR%" && set "KM_SELF_RAG_API_BASE_URL=%KM_SELF_RAG_API_BASE_URL%" && uv run --no-project api.py"

echo.
echo ========================================
echo Services Started Successfully!
echo ========================================
echo.
echo Open WebUI: http://localhost:%PORT%
echo KM Service: http://localhost:%API_PORT%
echo.
echo Two console windows have been opened for each service.
echo Close those windows to stop the services.
echo.
pause
exit /b 0

:: Function to refresh environment variables
:RefreshEnv
:: Update PATH from registry
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SysPATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "UserPATH=%%b"
set "PATH=%SysPATH%;%UserPATH%"
exit /b 0

