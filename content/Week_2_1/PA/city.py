import numpy as np
import matplotlib.pyplot as plt


class Plan:
    def __init__(self, coordinates, side_length):
        self.coordinates = coordinates
        self.side_length = side_length
        self.triangles = None
        self.shared_sides = None
        self.bar_coordinates = None
        self.kapsalon_coordinates = None
        self.refinement_number = 0
        self.check_initialization()

    def try_triangles(self, triangles=None, triangle_id=None):
        if triangles is None:
            assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Provide as keyword argument triangles=[[],[],...], or \n'
                +'--> Define as attribute triangles.\n')
            triangles = self.triangles
        if triangle_id is None:
            triangle_id = range(len(triangles))
        self.plot_triangles(triangles=triangles, triangle_id=triangle_id)
        self.check_triangles(triangles=triangles, triangle_id=triangle_id)

    def get_bar_coordinates(self):
        assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Provide as keyword argument triangles=[[],[],...], or \n'
                +'--> Define as attribute triangles.\n')
        
        # YOUR_CODE_HERE

        return bar_coordinates
    
    def get_kapsalon_coordinates(self):
        assert self.shared_sides is not None, (
                'NO SIDES DEFINED!\n'
                +'--> Provide as keyword argument sides=[[[],[]],...], or \n'
                +'--> Define as attribute shared_sides.\n')
        
        # YOUR_CODE_HERE

        return kapsalon_coordinates

    def get_triangle_area(self, triangle_id, triangles):
        triangle = triangles[triangle_id]
        a = self.coordinates[triangle[0]]
        b = self.coordinates[triangle[1]]
        c = self.coordinates[triangle[2]]
        return 0.5*np.abs(np.cross(b-a, c-a))

    def plot_everything(self):
        fig = self.plot_coordinates()
        ax = fig.get_axes()[0]
        if self.bar_coordinates is not None:
            ax.scatter(self.bar_coordinates[:, 0],
                       self.bar_coordinates[:, 1],
                       label='Bars',
                       marker='s',
                       color='red')
        if self.kapsalon_coordinates is not None:
            ax.scatter(self.kapsalon_coordinates[:, 0],
                       self.kapsalon_coordinates[:, 1],
                       label='Kapsalon Shops',
                       marker='^',
                       color='orange')
        ax.legend()
        ax.set_title('My Plan for the City')
        return fig

    def plot_coordinates(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.coordinates[:, 0],
                   self.coordinates[:, 1],
                   label='Student Houses',
                   color='blue')
        ax.set_title('Triangle Vertices')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        text_offset = 0.1
        for i in range(self.coordinates.shape[0]):
            ax.text(self.coordinates[i, 0] - text_offset,
                    self.coordinates[i, 1] - text_offset,
                    str(i),
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=12, color='black')
        
        ax.grid(True)
        ax.axis('equal')
        # ax.set_xlim(0, 8)
        # ax.set_ylim(0, 6)
        return fig
    
    def plot_triangles(self, triangles=None, triangle_id=None):
        if triangles is None:
            assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Provide as keyword argument triangles=[[],[],...], or \n'
                +'--> Define as attribute triangles.\n')
            triangles = self.triangles
        if triangle_id is None:
            triangle_id = range(len(triangles))
        fig = self.plot_coordinates()
        ax = fig.get_axes()[0]
        if isinstance(triangle_id, int):
            ax.set_title(f'Triangle {triangle_id} (vertices {triangles[triangle_id]})')
            triangle_id = [triangle_id]
        else:
            ax.set_title(f'Triangles plotted: {triangle_id}')
            
        for t_id in triangle_id:
            triangle = triangles[t_id]
            for i in range(3):
                start = self.coordinates[triangle[i]]
                end = self.coordinates[triangle[(i+1)%3]]
                ax.plot([start[0], end[0]], [start[1], end[1]],
                    color='black', linestyle='--', linewidth=2)
                ax.fill(self.coordinates[triangle, 0], 
                    self.coordinates[triangle, 1], 
                    color='green', alpha=0.1)
        return fig

    def plot_shared_sides(self, sides=None, side_id=None):
        if sides is None:
            assert self.shared_sides is not None, (
                'NO SIDES DEFINED!\n'
                +'--> Provide as keyword argument sides=[[[],[]],...], or \n'
                +'--> Define as attribute shared_sides.\n')
            sides = self.shared_sides
        if side_id is None:
            side_id = range(len(sides))
        elif isinstance(side_id, int):
            side_id = [side_id]

        unique_triangles = list(set([triangle for side in sides for triangle in side[1]]))

        fig = self.plot_triangles(triangle_id=unique_triangles)
        ax = fig.get_axes()[0]

        if isinstance(side_id, int):
            ax.set_title((f'Side {side_id} (vertices {sides[side_id][0]}, '
                          +f'triangles {sides[side_id][0]})'))
        else:
            ax.set_title((f'Sides plotted: {side_id} and '
                         +f'triangles: {unique_triangles}'))

        for s_id in side_id:
            side = sides[s_id]
            start = self.coordinates[side[0][0]]
            end = self.coordinates[side[0][1]]
            ax.plot([start[0], end[0]], [start[1], end[1]],
                color='red', linestyle='--', linewidth=4)
        return fig

    def check_shared_sides(self):
        assert (
             isinstance(self.shared_sides, list)
        ), (
             'Arg shared_sides should be a list of lists'
        )
        for i in range(len(self.shared_sides)):
            assert (
                isinstance(self.shared_sides[i][0], list)
            ), (
                f'First item in arg shared_sides, item {i}, is not a list'
            )
            assert (
                isinstance(self.shared_sides[i][1], list)
            ), (
                f'Second item in arg shared_sides, item {i}, is not a list'
            )
            assert (
                len(self.shared_sides[i][0]) == 2
            ), (
                f'First item in arg shared_sides, item {i}, must have 2 elements'
            )
            assert (
                len(self.shared_sides[i][1]) == 2
            ), (
                f'Second item in arg shared_sides, item {i}, must have 2 elements'
            )
            self.check_sides_in_triangles(i,
                                          self.shared_sides[i][0],
                                          self.shared_sides[i][1])
        print('The sides you provided seem to be defined correctly!')
        if len(self.shared_sides) != 10:
            print('WARNING: You should have 10 shared sides'
                           +f' --> currently only {len(self.shared_sides)}')

    def check_sides_in_triangles(self, side, vertices, triangle_id):
        triangle_1 = self.triangles[triangle_id[0]]
        triangle_2 = self.triangles[triangle_id[1]]
        common_vertices = set(triangle_1).intersection(set(triangle_2))
        assert len(common_vertices) == 2, (
            f'Side {side}: Triangles {triangle_id[0]} and {triangle_id[1]} do not share exactly two vertices'
        )
        assert vertices[0] in triangle_1 and vertices[0] in triangle_2, (
            f'Vertex {vertices[0]} is not shared by triangles {triangle_id[0]} and {triangle_id[1]}'
        )
        assert vertices[1] in triangle_1 and vertices[1] in triangle_2, (
            f'Vertex {vertices[1]} is not shared by triangles {triangle_id[0]} and {triangle_id[1]}'
        )
        return True


    def check_initialization(self):
        assert (
             isinstance(self.coordinates, np.ndarray)
        ), (
             'Arg coordinates should be a numpy array'
        )
        assert (
            self.coordinates.shape[1] == 2
        ), (
            'Arg coordinates should have 2 columns'
        )
        assert (
            self.coordinates.shape[0] == 11
        ), (
            'Arg coordinates should have 11 rows'
        )

    def check_triangles(self, triangles=None, triangle_id=None, report=True):
        tol = self.side_length/20
        if triangles is None:
            assert self.triangles is not None, (
                'NO TRIANGLES DEFINED!\n'
                +'--> Provide as keyword argument triangles=[[],[],...], or \n'
                +'--> Define as attribute triangles.\n')
            triangles = self.triangles
        if triangle_id is None:
            triangle_id = range(len(triangles))
        assert (
             isinstance(triangles, list)
        ), (
             'Arg triangles should be a list of lists'
        )
        assert (
             isinstance(triangles[0], list)
        ), (
             'First item in arg triangles is not a list'
        )
        for i, tri in enumerate(triangles):
            assert (
                len(tri) == 3
            ), (
                f'Problem with triangle at row {i}: must have 3 vertices'
            )

        if self.refinement_number == 0:
            if len(triangles) != 10:
                print('WARNING: You should have 10 triangles'
                            +f' --> currently only {len(triangles)}')
        
        height = np.sqrt(self.side_length**2 - (self.side_length/2)**2)
        area = self.side_length*height/2
        for i in range(len(triangles)):
            area_i = self.get_triangle_area(i, triangles)
            assert (
                np.isclose(area_i, area, atol=tol)
            ), (
                f'Triangle {i} does not have the correct area:\n'
                +f'  {area_i:.3e} instead of {area:.3e} (tol = {tol:.2e})'
            )
            for j in range(3):
                side_length = np.linalg.norm(self.coordinates[tri[j]] - self.coordinates[tri[(j+1)%3]])
                assert (
                    np.isclose(side_length, self.side_length, atol=tol)
                ), (
                    f'One of the sides of triangle {i} has incorrect length'
                )
        if report:
            print('All triangles seem to be defined correctly!')