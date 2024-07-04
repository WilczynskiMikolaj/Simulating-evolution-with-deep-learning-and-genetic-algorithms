class Settings:
    """
    A class to store all settings for the simulation.

    Attributes
    ----------
    screen_width : int
        The width of the simulation screen.
    screen_height : int
        The height of the simulation screen.
    bg_color : tuple
        The background color of the simulation screen in RGB format.
    """

    def __init__(self):
        """
        Initializes the settings for the simulation.
        """
        self.screen_width = 856
        self.screen_height = 788
        self.bg_color = (0, 109, 91)
