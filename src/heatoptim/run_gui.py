# from tkinter import Tk
from gui.main_window import SimulationConfigGUI


def main():
    # Create the root Tkinter window
    # root = Tk()
    # Set up the main GUI window
    app = SimulationConfigGUI()
    # Run the Tkinter main loop
    app.mainloop()


if __name__ == "__main__":
    main()
