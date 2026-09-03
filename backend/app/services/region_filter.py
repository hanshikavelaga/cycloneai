def classify_region(latitude: float, longitude: float) -> str:
    """
    Evaluates latitude and longitude coordinates to classify
    the storm's location relative to the North Indian Ocean basin.
    
    Bounding Box:
      Latitude:  0.0 to 35.0 degrees North
      Longitude: 40.0 to 110.0 degrees East (Bay of Bengal & Arabian Sea)
    """
    if (0.0 <= latitude <= 35.0) and (40.0 <= longitude <= 110.0):
        return "NORTH_INDIAN_OCEAN"
    else:
        return "OTHER"
