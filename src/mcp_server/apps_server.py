import subprocess
import time
import webbrowser

from mcp.server import MCPServer

# Deliberately small allow-list: launch_app/close_app can only ever act on
# apps named here, never an arbitrary executable path. Extend this list to
# add more apps.
#
# "launch" is the command that starts the app; "process" is the name it
# actually runs under once started -- these can differ. Modern Windows
# Calculator, for example, is launched via calc.exe, but that immediately
# hands off to a separate process, CalculatorApp.exe, which is what
# tasklist/taskkill actually see -- found by testing, not assumed.
#
# "browser" is special-cased: it opens the OS default browser rather than a
# specific .exe, so it has no trackable process name and cannot be closed
# by name (process=None).
ALLOWED_APPS = {
    "notepad": {"launch": "notepad.exe", "process": "notepad.exe"},
    "calculator": {"launch": "calc.exe", "process": "CalculatorApp.exe"},
    "browser": {"launch": "browser", "process": None},
}

mcp = MCPServer(
    "AppsTools", "0.1.0",
    f"Launch/close a small allow-list of Windows applications: {', '.join(ALLOWED_APPS)}",
)


@mcp.tool()
def list_running_apps() -> list[str]:
    """List which allow-listed apps (by name, e.g. 'notepad') are currently running. 'browser' is never reported here -- there's no reliable single process name for 'the default browser'."""
    running = []
    for name, app in ALLOWED_APPS.items():
        if app["process"] is None:
            continue
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {app['process']}"], capture_output=True, text=True
        )
        if app["process"].lower() in result.stdout.lower():
            running.append(name)
    return running


@mcp.tool()
def launch_app(name: str) -> str:
    """Launch an allow-listed application by name. Allowed names: notepad, calculator, browser (opens the default web browser to a blank page)."""
    if name not in ALLOWED_APPS:
        raise ValueError(f"'{name}' is not an allowed app. Allowed: {list(ALLOWED_APPS)}")
    launch = ALLOWED_APPS[name]["launch"]
    if launch == "browser":
        webbrowser.open("about:blank")
        return "Opened the default browser"
    subprocess.Popen([launch])
    return f"Launched {name}"


@mcp.tool()
def close_app(name: str) -> str:
    """Close a running allow-listed application by name (graceful close only -- will not force-kill, so some apps may not actually close). Allowed names: notepad, calculator. 'browser' cannot be closed this way."""
    process = ALLOWED_APPS.get(name, {}).get("process")
    if process is None:
        raise ValueError(f"'{name}' cannot be closed this way. Closeable apps: notepad, calculator")

    result = subprocess.run(["taskkill", "/IM", process], capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"Could not close {name}: {result.stderr.strip() or result.stdout.strip()}")

    # taskkill without /F can report success ("signal sent") even when the
    # process ignores it and keeps running -- confirmed by testing against
    # the modern Calculator app. Verify it's actually gone before claiming
    # success, rather than trusting the exit code alone.
    time.sleep(1)
    check = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {process}"], capture_output=True, text=True)
    if process.lower() in check.stdout.lower():
        raise ValueError(
            f"{name} did not actually close -- it may have ignored the graceful close "
            "request (this tool will not force-terminate it)"
        )
    return f"Closed {name}"


if __name__ == "__main__":
    mcp.run()
