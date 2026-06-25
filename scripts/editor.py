#!/usr/bin/env python3
"""Launch Isaac Sim GUI for scene editing (importing CAD, configuring environments)."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

print("[INFO] Isaac Sim GUI is running. Use File > Import to load CAD files.")
print("[INFO] Close the window or press Ctrl+C to exit.")

while app.is_running():
    app.update()

app.close()
