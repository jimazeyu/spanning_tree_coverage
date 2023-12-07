import tkinter as tk
import heapq
import cv2
import numpy as np
import os

class GridMapGenerator:
    def __init__(self, master):
        """
        Initialize the application.
        - master: The main tkinter window.
        Creates UI elements including input fields for rows and columns and buttons for generating the grid and planning the path.
        """
        self.master = master
        self.grid = []
        self.buttons = []
        self.start_point = [0, 0]

        # Input fields and labels for grid dimensions
        tk.Label(master, text="Rows:").grid(row=0, column=0)
        self.rows_entry = tk.Entry(master)
        self.rows_entry.grid(row=0, column=1)

        tk.Label(master, text="Columns:").grid(row=1, column=0)
        self.cols_entry = tk.Entry(master)
        self.cols_entry.grid(row=1, column=1)

        # Buttons for generating the grid and planning the path
        generate_button = tk.Button(master, text="Generate", command=self.generate_grid)
        generate_button.grid(row=2, column=0, columnspan=2)
        plan_button = tk.Button(master, text="Plan Path", command=self.plan_path)
        plan_button.grid(row=2, column=2, columnspan=2)

    def generate_grid(self):
        """
        Generate the grid based on user input.
        - Creates a grid of buttons where each button represents a cell in the grid.
        - Binds left click to set or remove obstacles, and right click to set the start point.
        """
        rows = int(self.rows_entry.get())
        cols = int(self.cols_entry.get())
        
        # Clear any existing grid buttons
        for row_buttons in self.buttons:
            for button in row_buttons:
                button.destroy()
        
        # Configure window row and column weights
        for i in range(rows):
            self.master.grid_rowconfigure(i+3, weight=1, minsize=20)
        for j in range(cols):
            self.master.grid_columnconfigure(j, weight=1, minsize=20)
        
        self.grid = [[0] * cols for _ in range(rows)]
        self.buttons = []
        
        # Create grid buttons
        button_size = 20
        for i in range(rows):
            row_buttons = []
            for j in range(cols):
                button = tk.Button(self.master, bg='white', width=button_size, height=button_size)
                button.grid(row=i+3, column=j, sticky='nsew', padx=1, pady=1)
                button.bind('<Button-1>', self.toggle_obstacle(i, j))
                button.bind('<Button-3>', self.set_start_point(i, j))
                row_buttons.append(button)
            self.buttons.append(row_buttons)

    def toggle_obstacle(self, i, j):
        """
        Toggle an obstacle in the grid.
        - i, j: Grid cell coordinates.
        Changes the color of the button to represent an obstacle or a free cell.
        """
        def command(event=None):
            if self.grid[i][j] == 0:
                self.grid[i][j] = 1
                self.buttons[i][j]['bg'] = 'black'
            else:
                self.grid[i][j] = 0
                self.buttons[i][j]['bg'] = 'white'
        return command

    def set_start_point(self, i, j):
        """
        Set the start point for path planning.
        - i, j: Grid cell coordinates.
        Changes the color of the start cell.
        """
        def command(event=None):
            if self.start_point:
                pi, pj = self.start_point
                self.buttons[pi][pj]['bg'] = 'white'
            self.start_point = (i, j)
            self.buttons[i][j]['bg'] = 'green'
        return command

    def is_connected(self, start):
        """
        Check if the grid is fully connected (all non-obstacle cells are reachable).
        - start: Starting cell coordinates for the connection check.
        Uses depth-first search (DFS) to explore the grid.
        """
        rows, cols = len(self.grid), len(self.grid[0])
        stack = [start]
        visited = set()
        
        # Perform DFS to check connectivity
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited:
                visited.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and self.grid[rr][cc] == 0 and (rr, cc) not in visited:
                        stack.append((rr, cc))
        
        # Check if all non-obstacle cells were visited
        for r in range(rows):
            for c in range(cols):
                if self.grid[r][c] == 0 and (r, c) not in visited:
                    return False
        return True

    def dfs_visit_order(self, start, path_from):
        """
        Determine the visit order of cells using depth-first search (DFS).
        - start: Starting cell coordinates for DFS.
        - path_from: Dictionary mapping each cell to the cell it was visited from.
        Returns a dictionary where keys are cell coordinates and values are the order in which cells were visited.
        """
        visit_order = {}
        order = 1
        stack = [start]

        while stack:
            node = stack.pop()
            if node not in visit_order:
                visit_order[node] = order
                order += 1

                for next_node in [neigh for neigh in path_from if path_from[neigh] == node]:
                    stack.append(next_node)

        return visit_order

    def draw_path(self, visit_order, path_from):
        """
        Draw the path on the grid using OpenCV.
        - visit_order: Dictionary with the visit order of each cell.
        - path_from: Dictionary mapping each cell to the cell it was visited from.
        Visualizes the grid, obstacles, and the path.
        """
        rows, cols = len(self.grid), len(self.grid[0])
        cell_size = 80
        img = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

        # Draw grid cells and obstacles
        for r in range(rows):
            for c in range(cols):
                center = (c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)
                if self.grid[r][c] == 1:
                    cv2.rectangle(img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), (0, 0, 255), -1)
                else:
                    cv2.rectangle(img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), (255, 255, 255), 1)

        # Draw the path lines
        for point, from_point in path_from.items():
            if from_point is not None:
                start = (from_point[1] * cell_size + cell_size // 2, from_point[0] * cell_size + cell_size // 2)
                end = (point[1] * cell_size + cell_size // 2, point[0] * cell_size + cell_size // 2)
                cv2.line(img, start, end, (0, 255, 0), 2)

        # Draw the visit order
        for point, order in visit_order.items():
            center = (point[1] * cell_size + cell_size // 2, point[0] * cell_size + cell_size // 2)
            cv2.putText(img, str(order), center, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Path", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def generate_hamiltonian_path(self, sub_grid, path_from):
        """
        Generate a Hamiltonian path in the subdivided grid.
        - sub_grid: A 2x larger grid subdivided from the original grid.
        - path_from: Dictionary mapping each cell to the cell it was visited from.
        Returns a list of cell coordinates representing the Hamiltonian path.
        """
        # Create a set of bi-directional edges
        path_set_bi = set()
        for k, v in path_from.items():
            if v is not None:
                path_set_bi.add((k, v))
                path_set_bi.add((v, k))
        
        rows, cols = len(self.grid), len(self.grid[0])

        # Map large grid obstacles to the subdivided grid
        for r in range(rows):
            for c in range(cols):
                if self.grid[r][c] == 1:
                    sub_grid[r*2][c*2] = sub_grid[r*2+1][c*2] = sub_grid[r*2][c*2+1] = sub_grid[r*2+1][c*2+1] = 1

        # Generate the Hamiltonian path
        hamiltonian_path = []
        stack = [(self.start_point[0] * 2, self.start_point[1] * 2)]
        visited = set()
        
        while stack:
            r, c = stack[-1]
            if (r, c) not in visited:
                visited.add((r, c))
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    # Check if the next cell is within the subdivided grid and not an obstacle
                    if 0 <= rr < len(sub_grid) and 0 <= cc < len(sub_grid[0]) and sub_grid[rr][cc] == 0 and (rr, cc) not in visited:
                        # If both from and to cells are inside the same cell in the original grid, check if the path is valid
                        if r // 2 == rr // 2 and c // 2 == cc // 2:
                            outside = (r // 2, c // 2)
                            if c % 2 == 0 and cc % 2 == 0 \
                            and path_set_bi.__contains__((outside, (outside[0], outside[1] - 1))):
                                continue
                            if c % 2 == 1 and cc % 2 == 1 \
                            and path_set_bi.__contains__((outside, (outside[0], outside[1] + 1))):
                                continue
                            if r % 2 == 0 and rr % 2 == 0 \
                            and path_set_bi.__contains__((outside, (outside[0] - 1, outside[1]))):
                                continue
                            if r % 2 == 1 and rr % 2 == 1 \
                            and path_set_bi.__contains__((outside, (outside[0] + 1, outside[1]))):
                                continue
                        # If both from and to cells are inside different cells in the original grid, check if the path is valid
                        else:
                            if (r // 2, c // 2) != (rr // 2, cc // 2) and not path_set_bi.__contains__(((r // 2, c // 2), (rr // 2, cc // 2))):
                                continue
                        stack.append((rr, cc))
                        hamiltonian_path.append((r, c))
                        hamiltonian_path.append((rr, cc))
                        break
                else:
                    stack.pop()
            else:
                stack.pop()

        return hamiltonian_path

    def draw_hamiltonian_path(self, path):
        """
        Draw the Hamiltonian path on the subdivided grid using OpenCV.
        - path: List of cell coordinates representing the Hamiltonian path.
        Visualizes the path on a grid, highlighting the route taken.
        """
        rows, cols = len(self.grid) * 2, len(self.grid[0]) * 2
        cell_size = 20
        img = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

        # Draw the subdivided grid cells
        for r in range(rows):
            for c in range(cols):
                center = (c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)
                cv2.rectangle(img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), (255, 255, 255), 1)

        # Draw the Hamiltonian path lines
        for i in range(1, len(path)):
            start = (path[i-1][1] * cell_size + cell_size // 2, path[i-1][0] * cell_size + cell_size // 2)
            end = (path[i][1] * cell_size + cell_size // 2, path[i][0] * cell_size + cell_size // 2)
            cv2.line(img, start, end, (0, 255, 0), 2)

        cv2.imshow("Hamiltonian Path", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def subdivide_grid(self):
        """
        Subdivide each cell of the original grid into four smaller cells.
        This is used to create a finer grid for the Hamiltonian path.
        Returns a new grid with double the number of rows and columns.
        """
        sub_rows, sub_cols = len(self.grid) * 2, len(self.grid[0]) * 2
        sub_grid = [[0 for _ in range(sub_cols)] for _ in range(sub_rows)]

        # Copy the state of each cell in the original grid to the four corresponding cells in the subdivided grid
        for r in range(len(self.grid)):
            for c in range(len(self.grid[0])):
                sub_grid[r*2][c*2] = self.grid[r][c]
                sub_grid[r*2+1][c*2] = self.grid[r][c]
                sub_grid[r*2][c*2+1] = self.grid[r][c]
                sub_grid[r*2+1][c*2+1] = self.grid[r][c]

        return sub_grid

    def plan_path(self):
        """
        Main function to plan the path.
        First checks if the start point is blocked or if the map is not connected.
        Then, it generates a minimum spanning tree (MST) and determines a Hamiltonian path based on the MST.
        Finally, it draws the paths and saves the Hamiltonian path to a file.
        """
        if self.grid[self.start_point[0]][self.start_point[1]] == 1:
            print("Start point is blocked, cannot plan path.")
            return
        if not self.is_connected(self.start_point):
            print("Map is not connected, cannot plan path.")
            return

        rows, cols = len(self.grid), len(self.grid[0])
        mst = set()
        path_from = {}
        visit_order = {}
        order = 1
        edges = [(0, self.start_point, None)]  # Priority queue for MST generation

        # Generate the MST
        while edges and len(mst) < rows * cols:
            weight, (r, c), from_cell = heapq.heappop(edges)
            if (r, c) in mst:
                continue
            mst.add((r, c))
            if from_cell is not None:
                path_from[(r, c)] = from_cell
            visit_order[(r, c)] = order
            order += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in mst and self.grid[rr][cc] == 0:
                    heapq.heappush(edges, (1, (rr, cc), (r, c)))

        # Draw the path based on the MST
        visit_order = self.dfs_visit_order(self.start_point, path_from)
        self.draw_path(visit_order, path_from)

        # Generate and draw the Hamiltonian path
        sub_grid = self.subdivide_grid()
        hamiltonian_path = self.generate_hamiltonian_path(sub_grid, path_from)
        self.draw_hamiltonian_path(hamiltonian_path)

        # Save the Hamiltonian path to a file
        os.makedirs("output", exist_ok=True)
        with open("output/path.txt", "w") as f:
            for r, c in hamiltonian_path:
                f.write(f"{r},{c}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = GridMapGenerator(root)
    root.geometry("800x600")
    root.mainloop()