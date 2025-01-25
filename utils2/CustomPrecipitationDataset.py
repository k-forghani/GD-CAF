import os
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class CustomPrecipitationDataset(Dataset):
    def __init__(self, dataset_path, time_steps, future_look, delimiter='/', fast_dev_run=False, cell_path='', cell_cutoff=100):
        self.tile_positions = None
        self.pics_width = None  # Total x length of input data (whole europe)
        self.pics_height = None
        self.grid_data = None
        self.total_length = None
        self.cell_shapes = None
        self.graph_size = None

        self.dataset_path = dataset_path  # path to the netCDF files
        self.past_steps = time_steps  # Number of past observations
        self.future_look = future_look  # Number of future observations
        self.delimiter = delimiter
        self.cell_path = cell_path  # Randomly placed cells, position file
        self.cell_cutoff = cell_cutoff  # At this number stop reading in more cells
        self.fast_dev_run = fast_dev_run  # When testing do not load every year
        self.vocal = True

        # Read files
        self.count_data_rows()

    def count_data_rows(self):
        """
        Read .nc datafiles to count the total number of rows, adjusted for the new dataset format
        :return:
        """
        # List all subdirectories (folders) in the dataset path
        folders = [folder for folder in Path(self.dataset_path).glob("*") if folder.is_dir()]

        n = 0

        # Check if folders exist and open the relevant files to extract grid dimensions
        if folders:
            # Assuming new dataset structure - using the first folder's data for grid size extraction
            f1 = folders[0] / "data_stream-oper_stepType-accum.nc"  # Replace with the appropriate filename
            f2 = folders[0] / "data_stream-oper_stepType-instant.nc"  # Replace with the appropriate filename
            
            # Opening the .nc files using xarray
            nc1 = xr.open_dataset(f1, engine="netcdf4")
            nc2 = xr.open_dataset(f2, engine="netcdf4")
            
            # Merging the datasets and extracting the grid dimensions
            nc = xr.merge([nc1, nc2], join="exact")
            self.pics_height = nc.variables['tp'][:].shape[1]  # Height from the 'tp' variable (time, latitude, longitude)
            self.pics_width = nc.variables['tp'][:].shape[2]   # Width from the 'tp' variable

        # Count the total number of pictures across all folders
        for folder in folders:
            f1 = folder / "data_stream-oper_stepType-accum.nc"  # Replace with the appropriate filename
            f2 = folder / "data_stream-oper_stepType-instant.nc"  # Replace with the appropriate filename
            
            # Open the files and merge datasets to count the number of time steps (pictures)
            nc1 = xr.open_dataset(f1, engine="netcdf4")
            nc2 = xr.open_dataset(f2, engine="netcdf4")
            nc = xr.merge([nc1, nc2], join="exact")

            # Add the number of time steps from the 'tp' variable
            n += nc.variables['tp'][:].shape[0]

            # If fast development mode is enabled, stop after the first month (for debugging purposes)
            month = folder.name.split('_')[2]  # Assuming folder name contains a month part
            if month == '01' and self.fast_dev_run:
                break

        # Store the total number of images (time steps)
        self.total_length = n

        # Load a random tile (if necessary) - keep this functionality if required
        self.load_random_tile()

        # Print out the total number of images and grids
        print(f'Total number of images: {n}')
        print(f'Nr. Images/grids: {n / self.graph_size}')


    def load_random_tile(self):
        """
        Load random tile positions from the cell position file and adapt them to the dataset coordinates
        """
        self.tile_positions = []

        with open(self.cell_path) as file:
            lines = [line.rstrip() for line in file]

            for line in lines:
                # Example: 178 91 80 40 : x, y , edge_x, edge_y
                content = line.split(' ')

                square = {
                    'x': int(content[0]),  # Longitude index
                    'y': int(content[1]),  # Latitude index
                    'edge_x': int(content[2]),  # Width of the tile
                    'edge_y': int(content[3])   # Height of the tile
                }
                self.tile_positions.append(square)

                if len(self.tile_positions) == self.cell_cutoff:
                    print(f'Cut off at {self.cell_cutoff} tiles.')
                    break

        # Ensure valid tile positions within dataset dimensions
        self.tile_positions = [
            tile for tile in self.tile_positions
            if tile['x'] + tile['edge_x'] <= self.pics_width
            and tile['y'] + tile['edge_y'] <= self.pics_height
        ]

        # Set number of grids
        self.graph_size = len(self.tile_positions)

        # Set cell shapes (height, width) from the first tile
        if self.graph_size > 0:
            self.cell_shapes = (self.tile_positions[0]['edge_y'], self.tile_positions[0]['edge_x'])
        
        print(f"Loaded {self.graph_size} tiles.")

    def iterate_random_cells(self, nc, grid_data, from_counter):
        """
        Extracts grid data from the dataset based on predefined tile positions.

        :param nc: Xarray dataset (merged precipitation, wind, temperature, etc.)
        :param grid_data: Numpy array to store extracted data
        :param from_counter: Starting index for grid_data
        :return: Updated grid_data
        """
        for i in range(len(self.tile_positions)):
            # Extract the relevant variable (e.g., total precipitation 'tp')
            if 'tp' in nc.variables:
                raw_data = nc['tp'].values  # Convert xarray data to numpy array
            else:
                raise ValueError("Variable 'tp' not found in dataset.")

            # Get tile position
            square = self.tile_positions[i]

            # Ensure tile is within valid dataset dimensions
            y_start, y_end = square['y'], square['y'] + square['edge_y']
            x_start, x_end = square['x'], square['x'] + square['edge_x']

            if y_end > raw_data.shape[1] or x_end > raw_data.shape[2]:
                print(f"Skipping tile {i}: Out of dataset bounds.")
                continue  # Skip tiles that exceed dataset size

            print(
                "Processing tile:",
                grid_data[from_counter: from_counter + raw_data.shape[0], i, :, :].shape,
                raw_data[:, y_start:y_end, x_start:x_end].shape,
                grid_data.shape,
                raw_data.shape
            )

            # Store extracted tile data
            grid_data[from_counter: from_counter + raw_data.shape[0], i, :, :] = \
                raw_data[:, y_start:y_end, x_start:x_end]

        return grid_data

    def load_grid(self):
        """
        Load all .nc files, extract disjoint regions, and convert them into a numpy array.
        """
        # Read dataset folders
        folders = [folder for folder in Path(self.dataset_path).glob("*") if folder.is_dir()]
        
        # Initialize tracking index
        from_counter = 0  

        # Grid cell dimensions
        grid_y, grid_x = self.cell_shapes  

        # Ensure dimensions are multiples of 4 (optional)
        self.pics_width -= self.pics_width % 4
        self.pics_height -= self.pics_height % 4

        # Initialize numpy array for storing extracted data
        grid_data = np.zeros((self.total_length, self.graph_size, grid_y, grid_x), dtype='float32')
        print(f'Initialized grid data array: {grid_data.shape}')

        for folder in folders:
            month = folder.name.split('_')[2]  # Extract month from folder name

            # Load dataset files
            f1 = folder / "data_stream-oper_stepType-accum.nc"
            f2 = folder / "data_stream-oper_stepType-instant.nc"

            nc1 = xr.open_dataset(f1, engine="netcdf4")
            nc2 = xr.open_dataset(f2, engine="netcdf4")
            nc = xr.merge([nc1, nc2], join="exact")

            # Validate the existence of 'tp' variable before accessing it
            if 'tp' not in nc.variables:
                print(f"Skipping {folder.name}: 'tp' variable not found.")
                continue  

            number_of_images = nc['tp'].shape[0]
            print(f'Processing {folder.name}, TP shape: {nc["tp"].shape}')

            # Load data into grid
            grid_data = self.iterate_random_cells(nc, grid_data, from_counter)

            # Update time index
            from_counter += number_of_images

            # Fast stop mode (Only process January)
            if month == '01' and self.fast_dev_run:
                break

        self.grid_data = grid_data  # Store processed grid data

    def __getitem__(self, idx):
        """
        Return a specific observation and label from the dataset.
        """
        # Extract past steps (observations) and the future look (label)
        x = self.grid_data[idx: idx + self.past_steps, :, :]
        y = self.grid_data[idx + self.past_steps + self.future_look - 1]

        return x, y

    def __len__(self):
        """
        Return the total number of possible samples.
        """
        return self.grid_data.shape[0] - self.past_steps - self.future_look + 1

    def normalize(self, local=False):
        """
        Normalize the grid data using min-max normalization.
        """
        # Print data range before normalization for debugging
        print(f'NORMALIZE: {np.amin(self.grid_data)} {np.amax(self.grid_data)}')
        print(f'Grid data shape: {self.grid_data.shape}')

        # Perform normalization
        self.grid_data = self._normalize(self.grid_data, local)

    @staticmethod
    def _normalize(input_data, local=False):
        """
        Perform min-max normalization on the input data.

        Formula: x = (x - min) / (max - min)
        """
        if local:
            # Normalize per local min/max
            min_vals, max_vals = np.amin(input_data), np.amax(input_data)
        else:
            # Use a fixed global max value (adjust this if needed based on your dataset)
            min_vals, max_vals = 0, 0.03651024401187897  # Global max value

        # Print normalization parameters for debugging
        print(f'Normalization {input_data.shape}: x = (x - {min_vals}) / ({max_vals} - {min_vals})')

        # Apply normalization
        return (input_data - min_vals) / (max_vals - min_vals)

