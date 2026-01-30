#### Manual Processing Steps

1. ruse run_brainreg.py to run brainreg and register all data to allen 10um atlas
2. open downampled_standard_2.tiff in napari and manually label fiber tips with a single point in a point layer
3. save out these points (in atlas space) as .csv files in subject/fiber_coordinates
4. GridMaze.preprocessing code will then save fiber info as .json files in the processed data